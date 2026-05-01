from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm
from .data import load_instruction_labels
from .paths import DATA_DIR, FEATURES_DIR, ensure_project_dirs


SPLIT_FRACTIONS = {
    "train": 0.6,
    "valid_seen": 0.2,
    "valid_unseen": 0.2,
}


def _as_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def filter_confident_labels(
    labels: pd.DataFrame,
    *,
    negative_max: float,
    positive_min: float,
) -> pd.DataFrame:
    """Keep only labels with strong annotator agreement."""
    mask = labels["ambiguous_mean"].le(negative_max) | labels["ambiguous_mean"].ge(positive_min)
    filtered = labels[mask].copy()
    filtered["ambiguous"] = filtered["ambiguous_mean"].ge(positive_min).astype(int)
    return filtered.reset_index(drop=True)


def _candidate_targets(row: pd.Series) -> list[int]:
    starts: list[int] = []
    start = _as_int(row.get("subgoal_start"))
    end = _as_int(row.get("subgoal_end"))
    idx = _as_int(row.get("subgoal_idx"))
    # try the midpoint first because it is usually closest to the action.
    if start is not None and end is not None:
        starts.append(max(0, (start + end) // 2))
    if start is not None:
        starts.append(max(0, start))
    if end is not None:
        starts.append(max(0, end))
    if idx is not None:
        starts.append(max(0, idx))
    starts.append(0)
    return list(dict.fromkeys(starts))


def _existing_image_path(
    *,
    image_roots: Iterable[Path],
    split: str,
    task_id: str,
    trial_id: str,
    image_name: str,
) -> Path | None:
    rel = Path(split) / task_id / trial_id / "raw_images" / image_name
    names = [image_name]
    # some extracted subsets switch between png and jpg.
    if image_name.lower().endswith(".png"):
        names.append(f"{Path(image_name).stem}.jpg")
    elif image_name.lower().endswith(".jpg"):
        names.append(f"{Path(image_name).stem}.png")
    for root in image_roots:
        for name in names:
            rel = Path(split) / task_id / trial_id / "raw_images" / name
            candidates = [root / rel, root / "full_2.1.0" / rel]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
    return None


def build_folder_alfred_index(json_data: str | Path) -> pd.DataFrame:
    """Index ALFRED JSON trajectories by their folder task/trial IDs."""
    root = Path(json_data)
    rows: list[dict[str, object]] = []
    for traj_json in tqdm(sorted(root.rglob("traj_data.json")), desc="Indexing ALFRED trajectories"):
        trajectory_dir = traj_json.parent
        rel = trajectory_dir.relative_to(root)
        parts = rel.parts
        if len(parts) >= 3:
            split, task_id, trial_id = parts[-3], parts[-2], parts[-1]
        elif len(parts) == 2:
            split, task_id, trial_id = parts[0], parts[1], parts[1]
        else:
            continue
        if split not in {"train", "valid_seen", "valid_unseen"}:
            continue
        rows.append(
            {
                "split": split,
                "task_id": task_id,
                "trial_id": trial_id,
                "trajectory_dir": str(trajectory_dir),
                "traj_json": str(traj_json),
            }
        )
    return pd.DataFrame(rows)


def select_midpoint_frame(
    *,
    row: pd.Series,
    images: list[dict[str, object]],
    image_roots: Iterable[Path],
) -> tuple[Path | None, str | None, int | None]:
    """Choose an existing raw image nearest the subgoal midpoint/start/end."""
    if not images:
        return None, None, None

    split = str(row.get("split_alfred") if pd.notna(row.get("split_alfred")) else row.get("split"))
    task_id = str(row["task_id"])
    trial_id = str(row["trial_id"])
    targets = _candidate_targets(row)

    ranked: list[tuple[int, int, dict[str, object]]] = []
    for order, image in enumerate(images):
        low_idx = _as_int(image.get("low_idx"))
        high_idx = _as_int(image.get("high_idx"))
        image_idx = low_idx if low_idx is not None else high_idx
        if image_idx is None:
            image_idx = order
        distance = min(abs(image_idx - target) for target in targets)
        ranked.append((distance, order, image))

    for _, _, image in sorted(ranked, key=lambda item: (item[0], item[1])):
        image_name = image.get("image_name")
        if not image_name:
            continue
        image_path = _existing_image_path(
            image_roots=image_roots,
            split=split,
            task_id=task_id,
            trial_id=trial_id,
            image_name=str(image_name),
        )
        if image_path:
            selected_idx = _as_int(image.get("low_idx"))
            return image_path, str(image_name), selected_idx
    return None, None, None


def attach_existing_images(
    labels: pd.DataFrame,
    *,
    json_data: str | Path,
    image_roots: Iterable[str | Path],
) -> pd.DataFrame:
    """Attach an existing raw image path to each label row when possible."""
    image_root_paths = [Path(root) for root in image_roots]
    index = build_folder_alfred_index(json_data)
    if index.empty:
        raise ValueError(f"No ALFRED traj_data.json files found under {json_data}")

    merged = labels.merge(index, on=["task_id", "trial_id"], how="left", suffixes=("", "_alfred"))
    rows: list[dict[str, object]] = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Selecting clean ALFRED frames"):
        traj_json = row.get("traj_json")
        if pd.isna(traj_json):
            continue
        try:
            images = json.loads(Path(traj_json).read_text(encoding="utf-8")).get("images", [])
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        image_path, image_name, selected_idx = select_midpoint_frame(
            row=row,
            images=images,
            image_roots=image_root_paths,
        )
        if image_path is None:
            continue

        record = row.to_dict()
        record["image_path"] = str(image_path)
        record["selected_image_name"] = image_name
        record["selected_low_idx"] = selected_idx
        record["frame_selection"] = "nearest_subgoal_midpoint"
        rows.append(record)

    return pd.DataFrame(rows)


def _desired_pairs_by_split(target_rows: int) -> dict[str, int]:
    pairs: dict[str, int] = {}
    for split, fraction in SPLIT_FRACTIONS.items():
        split_rows = int(target_rows * fraction)
        pairs[split] = max(1, split_rows // 2)
    return pairs


def balance_after_matching(
    manifest: pd.DataFrame,
    *,
    target_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    """Balance labels inside each split after image validation."""
    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    desired_pairs = _desired_pairs_by_split(target_rows) if target_rows else {}

    for split in ("train", "valid_seen", "valid_unseen"):
        split_df = manifest[manifest["split"].astype(str).eq(split)]
        negative = split_df[split_df["ambiguous"].eq(0)]
        positive = split_df[split_df["ambiguous"].eq(1)]
        # the smaller class sets the balanced size.
        pair_count = min(len(negative), len(positive))
        if target_rows:
            pair_count = min(pair_count, desired_pairs.get(split, pair_count))
        if pair_count == 0:
            continue

        negative_sample = negative.sample(n=pair_count, random_state=int(rng.integers(0, 2**31 - 1)))
        positive_sample = positive.sample(n=pair_count, random_state=int(rng.integers(0, 2**31 - 1)))
        split_balanced = pd.concat([negative_sample, positive_sample], ignore_index=True)
        split_balanced = split_balanced.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 2**31 - 1)),
        )
        selected.append(split_balanced)

    if not selected:
        raise RuntimeError("No balanced rows available after filtering and image matching.")
    return pd.concat(selected, ignore_index=True)


def sample_after_matching(
    manifest: pd.DataFrame,
    *,
    target_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    """Sample rows by split without forcing class balance."""
    if not target_rows or len(manifest) <= target_rows:
        return manifest.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    selected: list[pd.DataFrame] = []
    desired_rows = {
        split: int(target_rows * fraction)
        for split, fraction in SPLIT_FRACTIONS.items()
    }
    assigned = sum(desired_rows.values())
    if assigned < target_rows:
        # put any rounding leftover into train.
        desired_rows["train"] += target_rows - assigned

    for split in ("train", "valid_seen", "valid_unseen"):
        split_df = manifest[manifest["split"].astype(str).eq(split)]
        if split_df.empty:
            continue
        rows = min(len(split_df), desired_rows.get(split, len(split_df)))
        selected.append(split_df.sample(n=rows, random_state=int(rng.integers(0, 2**31 - 1))))

    if not selected:
        raise RuntimeError("No rows available after filtering and image matching.")
    sampled = pd.concat(selected)
    if len(sampled) < target_rows:
        remaining = manifest.index.difference(sampled.index)
        extra_count = min(target_rows - len(sampled), len(remaining))
        if extra_count:
            sampled = pd.concat(
                [
                    sampled,
                    manifest.loc[remaining].sample(
                        n=extra_count,
                        random_state=int(rng.integers(0, 2**31 - 1)),
                    ),
                ]
            )

    return sampled.sample(
        frac=1.0,
        random_state=int(rng.integers(0, 2**31 - 1)),
    ).reset_index(drop=True)


def build_clean_manifest(
    *,
    output: str | Path,
    csv: str | Path | None = None,
    json_data: str | Path = DATA_DIR / "json_2.1.0",
    image_roots: Iterable[str | Path] = (
        DATA_DIR / "full_2.1.0",
        DATA_DIR / "alfred_subset_1500",
        DATA_DIR / "alfred_subset_300",
        DATA_DIR / "alfred_subset_300_natural",
    ),
    negative_max: float = 0.25,
    positive_min: float = 0.75,
    target_rows: int | None = 1500,
    seed: int = 42,
    no_download: bool = False,
    balance: bool = True,
) -> Path:
    ensure_project_dirs()
    labels = load_instruction_labels(csv, download=not no_download)
    confident = filter_confident_labels(
        labels,
        negative_max=negative_max,
        positive_min=positive_min,
    )
    matched = attach_existing_images(
        confident,
        json_data=json_data,
        image_roots=image_roots,
    )
    if matched.empty:
        raise RuntimeError("No clean labels could be matched to existing ALFRED raw images.")

    selected = (
        balance_after_matching(matched, target_rows=target_rows, seed=seed)
        if balance
        else sample_after_matching(matched, target_rows=target_rows, seed=seed)
    )
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)

    print(f"Labels before filtering: {len(labels):,}")
    print(f"Labels after confidence filtering: {len(confident):,}")
    print(f"Rows with existing images: {len(matched):,}")
    print(f"Rows written: {len(selected):,}")
    print(selected.groupby(["split", "ambiguous"]).size().unstack(fill_value=0))
    print(f"Manifest: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a clean balanced Sensor-VLM manifest.")
    parser.add_argument("--output", default=FEATURES_DIR / "clean-balanced-1500_manifest.csv")
    parser.add_argument("--csv", default=None)
    parser.add_argument("--json-data", default=DATA_DIR / "json_2.1.0")
    parser.add_argument(
        "--image-root",
        action="append",
        dest="image_roots",
        default=None,
        help="Raw-image root to search. Can be passed multiple times.",
    )
    parser.add_argument("--negative-max", type=float, default=0.25)
    parser.add_argument("--positive-min", type=float, default=0.75)
    parser.add_argument("--target-rows", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Preserve natural class skew after image matching; training remains class-weighted.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    image_roots = args.image_roots or [
        DATA_DIR / "full_2.1.0",
        DATA_DIR / "alfred_subset_1500",
        DATA_DIR / "alfred_subset_300",
        DATA_DIR / "alfred_subset_300_natural",
    ]
    build_clean_manifest(
        output=args.output,
        csv=args.csv,
        json_data=args.json_data,
        image_roots=image_roots,
        negative_max=args.negative_max,
        positive_min=args.positive_min,
        target_rows=args.target_rows,
        seed=args.seed,
        no_download=args.no_download,
        balance=not args.no_balance,
    )


if __name__ == "__main__":
    main()
