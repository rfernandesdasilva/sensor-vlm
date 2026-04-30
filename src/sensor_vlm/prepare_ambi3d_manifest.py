from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from .paths import DATA_DIR, FEATURES_DIR, ensure_project_dirs


AMBI3D_BASE_URL = "https://huggingface.co/datasets/jiayuttkx/ambi3d/resolve/main"
AMBI3D_FILES = {
    "train": "ambi3d_train.json",
    "test": "ambi3d_test.json",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def download_ambi3d(output_dir: str | Path | None = None, *, force: bool = False) -> dict[str, Path]:
    """Download Ambi3D train/test JSON files from Hugging Face."""
    root = Path(output_dir) if output_dir else DATA_DIR / "ambi3d"
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, filename in AMBI3D_FILES.items():
        output = root / filename
        if force or not output.exists():
            response = requests.get(f"{AMBI3D_BASE_URL}/{filename}", timeout=120)
            response.raise_for_status()
            output.write_bytes(response.content)
        paths[split] = output
    return paths


def _read_json_records(path: str | Path) -> list[dict[str, object]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    raise ValueError(f"Could not find row records in {path}")


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if value is None or pd.isna(value):
        return []
    return [value]


def _evidence_text(row: pd.Series) -> str:
    object_names = [str(value) for value in _as_list(row.get("object_names")) if str(value).strip()]
    unique_names = list(dict.fromkeys(object_names))
    ambiguity_type = str(row.get("ambiguity_type") or "").strip()
    parts = []
    if unique_names:
        parts.append("Objects: " + ", ".join(unique_names))
    if ambiguity_type:
        parts.append(f"Ambiguity type: {ambiguity_type}")
    return ". ".join(parts)


def _column_or_default(df: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in df:
        return df[column]
    return pd.Series([default] * len(df), index=df.index)


def load_ambi3d_records(
    *,
    train_json: str | Path | None = None,
    test_json: str | Path | None = None,
    download: bool = True,
) -> pd.DataFrame:
    """Load Ambi3D rows and preserve the source train/test split."""
    if train_json is None or test_json is None:
        paths = download_ambi3d() if download else {
            "train": DATA_DIR / "ambi3d" / AMBI3D_FILES["train"],
            "test": DATA_DIR / "ambi3d" / AMBI3D_FILES["test"],
        }
        train_json = train_json or paths["train"]
        test_json = test_json or paths["test"]

    rows: list[dict[str, object]] = []
    for source_split, path in (("train", train_json), ("test", test_json)):
        for record in _read_json_records(path):
            record = dict(record)
            record["source_split"] = source_split
            rows.append(record)
    return pd.DataFrame(rows)


def assign_sensor_vlm_splits(
    df: pd.DataFrame,
    *,
    valid_fraction: float = 0.1,
    seed: int = 42,
) -> pd.Series:
    """Map Ambi3D train/test into train/valid_seen/valid_unseen for existing training code."""
    splits = pd.Series("train", index=df.index, dtype=object)
    test_mask = df["source_split"].astype(str).eq("test")
    splits[test_mask] = "valid_unseen"

    train_indices = df.index[~test_mask]
    scenes = sorted(df.loc[train_indices, "scene_id"].astype(str).unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(scenes)
    valid_scene_count = max(1, int(len(scenes) * valid_fraction))
    valid_scenes = set(scenes[:valid_scene_count])
    valid_mask = ~test_mask & df["scene_id"].astype(str).isin(valid_scenes)
    splits[valid_mask] = "valid_seen"
    return splits


def discover_scene_views(view_roots: Iterable[str | Path], scene_id: str, *, max_views: int | None = None) -> list[str]:
    """Find local rendered scene images whose path or filename contains the Ambi3D scene id."""
    matches: list[Path] = []
    for root in [Path(path) for path in view_roots]:
        if not root.exists():
            continue
        scene_dirs = [root / scene_id] if (root / scene_id).exists() else []
        scene_dirs.extend(path for path in root.rglob(scene_id) if path.is_dir())

        candidates: list[Path] = []
        for scene_dir in dict.fromkeys(scene_dirs):
            candidates.extend(scene_dir.rglob("*"))
        candidates.extend(root.rglob(f"*{scene_id}*"))

        for candidate in dict.fromkeys(candidates):
            if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
                matches.append(candidate)
    deduped = [str(path) for path in sorted(dict.fromkeys(matches))]
    return deduped[:max_views] if max_views else deduped


def write_required_scene_list(manifest: pd.DataFrame, output: str | Path) -> Path:
    """Write the unique Ambi3D/ScanNet scene ids needed for visual baselines."""
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene_ids = sorted(manifest["scene_id"].astype(str).unique())
    output_path.write_text("\n".join(scene_ids) + "\n", encoding="utf-8")
    return output_path


def build_ambi3d_manifest(
    *,
    output: str | Path = FEATURES_DIR / "ambi3d_manifest.csv",
    train_json: str | Path | None = None,
    test_json: str | Path | None = None,
    view_roots: Iterable[str | Path] = (),
    valid_fraction: float = 0.1,
    seed: int = 42,
    max_rows: int | None = None,
    max_views: int | None = 4,
    scene_list_output: str | Path | None = None,
    no_download: bool = False,
) -> Path:
    """Normalize Ambi3D into the manifest columns used by Sensor-VLM caches."""
    ensure_project_dirs()
    raw = load_ambi3d_records(train_json=train_json, test_json=test_json, download=not no_download)
    manifest = pd.DataFrame(
        {
            "dataset": "ambi3d",
            "scene_id": raw["scene_id"].astype(str),
            "instruction_id": raw["instruction_id"].astype(str),
            "instruction": raw["question"].astype(str),
            "ambiguous": raw["answer"].astype(int),
            "ambiguity_type": _column_or_default(raw, "ambiguity_type", "").fillna("").astype(str),
            "object_id": _column_or_default(raw, "object_id", []).map(json.dumps),
            "object_names": _column_or_default(raw, "object_names", []).map(json.dumps),
            "source_split": raw["source_split"].astype(str),
        }
    )
    manifest["split"] = assign_sensor_vlm_splits(manifest, valid_fraction=valid_fraction, seed=seed)
    manifest["evidence_text"] = raw.apply(_evidence_text, axis=1)

    view_roots = list(view_roots)
    if view_roots:
        view_lists = [
            discover_scene_views(view_roots, scene_id, max_views=max_views)
            for scene_id in manifest["scene_id"].astype(str)
        ]
        manifest["image_paths"] = ["|".join(paths) for paths in view_lists]
        manifest["image_path"] = [paths[0] if paths else "" for paths in view_lists]

    if max_rows and len(manifest) > max_rows:
        sampled: list[pd.DataFrame] = []
        groups = list(manifest.groupby(["split", "ambiguous"], group_keys=False))
        per_group = max(1, max_rows // len(groups))
        for _, group in groups:
            sampled.append(group.sample(n=min(len(group), per_group), random_state=seed))
        selected = pd.concat(sampled)
        if len(selected) < max_rows:
            remaining = manifest.index.difference(selected.index)
            extra_count = min(max_rows - len(selected), len(remaining))
            if extra_count:
                selected = pd.concat(
                    [selected, manifest.loc[remaining].sample(n=extra_count, random_state=seed)],
                    ignore_index=True,
                )
        manifest = selected.head(max_rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    print(f"Rows written: {len(manifest):,}")
    print(manifest.groupby(["split", "ambiguous"]).size().unstack(fill_value=0))
    if scene_list_output:
        scene_list = write_required_scene_list(manifest, scene_list_output)
        print(f"Required scenes: {manifest['scene_id'].nunique():,}")
        print(f"Scene list: {scene_list}")
    if "image_path" in manifest:
        print(f"Rows with scene views: {manifest['image_path'].astype(str).ne('').sum():,}")
    print(f"Manifest: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare an Ambi3D manifest for Sensor-VLM.")
    parser.add_argument("--output", default=FEATURES_DIR / "ambi3d_manifest.csv")
    parser.add_argument("--train-json", default=None)
    parser.add_argument("--test-json", default=None)
    parser.add_argument("--view-root", action="append", dest="view_roots", default=None)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-views", type=int, default=4)
    parser.add_argument("--scene-list-output", default=None)
    parser.add_argument("--no-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    build_ambi3d_manifest(
        output=args.output,
        train_json=args.train_json,
        test_json=args.test_json,
        view_roots=args.view_roots or [],
        valid_fraction=args.valid_fraction,
        seed=args.seed,
        max_rows=args.max_rows,
        max_views=args.max_views,
        scene_list_output=args.scene_list_output,
        no_download=args.no_download,
    )


if __name__ == "__main__":
    main()
