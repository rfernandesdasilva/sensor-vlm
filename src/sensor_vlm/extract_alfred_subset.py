from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pandas as pd
import py7zr

from .data import load_instruction_labels
from .paths import ARTIFACTS_DIR, DATA_DIR, ensure_project_dirs


DEFAULT_ARCHIVE = DATA_DIR / "full_2.1.0.7z"
DEFAULT_OUTPUT_DIR = DATA_DIR / "alfred_subset"
DEFAULT_MANIFEST = ARTIFACTS_DIR / "features" / "dialfred_alfred_subset_manifest.csv"
SEVEN_ZIP_EXE = Path(r"C:\Program Files\AMD\CIM\Bin64\7z.exe")


def archive_names(archive_path: str | Path) -> set[str]:
    """Load archive headers without extracting the full dataset."""
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        return set(archive.getnames())


def split_to_alfred(split: str) -> str:
    if split == "valid_seen":
        return "valid_seen"
    if split == "valid_unseen":
        return "valid_unseen"
    return "train"


def candidate_image_paths(row: pd.Series, max_offset: int = 8) -> list[str]:
    split = split_to_alfred(str(row["split"]))
    base = f"full_2.1.0/{split}/{row['task_id']}/{row['trial_id']}/raw_images"
    starts: list[int] = []
    for key in ("subgoal_start", "subgoal_idx"):
        value = row.get(key)
        if pd.isna(value):
            continue
        try:
            starts.append(max(0, int(float(value))))
        except (TypeError, ValueError):
            continue
    starts.append(0)

    candidates: list[str] = []
    for start in starts:
        for offset in range(max_offset + 1):
            candidates.append(f"{base}/{start + offset:09d}.jpg")
        for offset in range(1, max_offset + 1):
            idx = start - offset
            if idx >= 0:
                candidates.append(f"{base}/{idx:09d}.jpg")
    return list(dict.fromkeys(candidates))


def balanced_sample(labels: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Prefer train/valid_seen/valid_unseen and both labels in a tiny subset."""
    per_bucket = max(1, max_rows // 6)
    buckets: list[pd.DataFrame] = []
    for split in ("train", "valid_seen", "valid_unseen"):
        split_df = labels[labels["split"].astype(str).eq(split)]
        for label in (0, 1):
            bucket = split_df[split_df["ambiguous"].eq(label)].head(per_bucket)
            buckets.append(bucket)
    sampled = pd.concat(buckets, ignore_index=True).drop_duplicates(
        subset=["instruction", "split", "task_id", "trial_id"]
    )
    if len(sampled) < max_rows:
        remainder = labels.drop(sampled.index, errors="ignore").head(max_rows - len(sampled))
        sampled = pd.concat([sampled, remainder], ignore_index=True)
    return sampled.head(max_rows).reset_index(drop=True)


def natural_sample(labels: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    """Keep split sizes even while preserving the natural label skew within each split."""
    per_split = max(1, max_rows // 3)
    buckets: list[pd.DataFrame] = []
    for split in ("train", "valid_seen", "valid_unseen"):
        split_df = labels[labels["split"].astype(str).eq(split)]
        buckets.append(split_df.head(per_split))
    sampled = pd.concat(buckets, ignore_index=True).drop_duplicates(
        subset=["instruction", "split", "task_id", "trial_id"]
    )
    if len(sampled) < max_rows:
        remainder = labels.drop(sampled.index, errors="ignore").head(max_rows - len(sampled))
        sampled = pd.concat([sampled, remainder], ignore_index=True)
    return sampled.head(max_rows).reset_index(drop=True)


def build_subset_manifest(
    *,
    archive_path: str | Path,
    max_rows: int,
    sampling: str = "balanced",
) -> tuple[pd.DataFrame, list[str]]:
    labels = load_instruction_labels()
    if sampling == "natural":
        labels = natural_sample(labels, max_rows=max_rows)
    else:
        labels = balanced_sample(labels, max_rows=max_rows)
    names = archive_names(archive_path)
    rows: list[dict[str, object]] = []
    files_to_extract: list[str] = []

    for _, row in labels.iterrows():
        selected = next((path for path in candidate_image_paths(row) if path in names), None)
        if selected is None:
            continue
        local_path = str(Path(*Path(selected).parts[1:]))
        record = row.to_dict()
        record["archive_image_path"] = selected
        record["image_path"] = local_path
        rows.append(record)
        files_to_extract.append(selected)

    return pd.DataFrame(rows), files_to_extract


def extract_files_with_7z(
    *,
    archive_path: str | Path,
    files_to_extract: list[str],
    output_dir: str | Path,
    list_file: str | Path,
) -> None:
    if not SEVEN_ZIP_EXE.exists():
        raise FileNotFoundError(f"7-Zip executable not found: {SEVEN_ZIP_EXE}")
    list_path = Path(list_file)
    list_path.parent.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(files_to_extract), encoding="utf-8")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            str(SEVEN_ZIP_EXE),
            "x",
            str(archive_path),
            f"@{list_path}",
            f"-o{output_dir}",
            "-y",
        ],
        check=True,
    )


def command_extract(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    manifest, files_to_extract = build_subset_manifest(
        archive_path=args.archive,
        max_rows=args.max_rows,
        sampling=args.sampling,
    )
    if manifest.empty:
        raise RuntimeError("No matching ALFRED images found in the archive for the sampled DialFRED rows.")

    list_file = ARTIFACTS_DIR / "features" / "alfred_subset_files.txt"
    extract_files_with_7z(
        archive_path=args.archive,
        files_to_extract=files_to_extract,
        output_dir=args.output_dir,
        list_file=list_file,
    )

    output_root = Path(args.output_dir)
    manifest["image_path"] = manifest["archive_image_path"].map(lambda path: str(output_root / path))
    output_manifest = Path(args.manifest)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    print(f"Extracted {len(files_to_extract)} ALFRED frames.")
    print(f"Manifest: {output_manifest}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a small ALFRED image subset for Sensor-VLM.")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--max-rows", type=int, default=60)
    parser.add_argument(
        "--sampling",
        choices=["balanced", "natural"],
        default="balanced",
        help="balanced gives 50/50 labels per split; natural preserves DialFRED label skew.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    command_extract(args)


if __name__ == "__main__":
    main()

