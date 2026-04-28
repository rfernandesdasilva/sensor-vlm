from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm


@dataclass
class AlfredTrajectory:
    split: str
    task_id: str
    trial_id: str
    trajectory_dir: Path
    traj_json: Path


def iter_traj_jsons(alfred_data_dir: str | Path) -> list[Path]:
    root = Path(alfred_data_dir)
    return sorted(root.rglob("traj_data.json"))


def build_alfred_index(alfred_data_dir: str | Path) -> pd.DataFrame:
    """Index ALFRED trajectories by task_id and trial_id."""
    rows: list[dict[str, object]] = []
    for traj_json in tqdm(iter_traj_jsons(alfred_data_dir), desc="Indexing ALFRED trajectories"):
        try:
            data = json.loads(traj_json.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        trajectory_dir = traj_json.parent
        split = trajectory_dir.parts[-3] if len(trajectory_dir.parts) >= 3 else ""
        task_id = str(data.get("task_id") or trajectory_dir.parent.name)
        trial_id = str(data.get("trial_id") or trajectory_dir.name)
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


def _load_traj_images(traj_json: str | Path) -> list[dict[str, object]]:
    data = json.loads(Path(traj_json).read_text(encoding="utf-8"))
    return data.get("images", [])


def select_image_for_subgoal(
    traj_json: str | Path,
    trajectory_dir: str | Path,
    *,
    subgoal_idx: int | float | str | None = None,
    subgoal_start: int | float | str | None = None,
) -> Path | None:
    """Select a raw ALFRED frame for a DialFRED subgoal if raw images are present."""
    images = _load_traj_images(traj_json)
    if not images:
        return None

    selected = None
    if subgoal_idx is not None and str(subgoal_idx) != "nan":
        try:
            high_idx = int(float(subgoal_idx))
            selected = next((img for img in images if int(img.get("high_idx", -1)) == high_idx), None)
        except ValueError:
            selected = None

    if selected is None and subgoal_start is not None and str(subgoal_start) != "nan":
        try:
            low_idx = int(float(subgoal_start))
            selected = min(images, key=lambda img: abs(int(img.get("low_idx", 0)) - low_idx))
        except ValueError:
            selected = None

    selected = selected or images[0]
    image_name = selected.get("image_name")
    if not image_name:
        return None

    image_path = Path(trajectory_dir) / "raw_images" / str(image_name)
    return image_path if image_path.exists() else None


def link_dialfred_to_alfred(
    dialfred_labels: pd.DataFrame,
    alfred_data_dir: str | Path,
    *,
    output_manifest: str | Path | None = None,
) -> pd.DataFrame:
    """Attach ALFRED trajectory metadata and raw image paths to DialFRED labels."""
    index = build_alfred_index(alfred_data_dir)
    if index.empty:
        raise ValueError(f"No ALFRED traj_data.json files found under {alfred_data_dir}")

    merged = dialfred_labels.merge(
        index,
        on=["task_id", "trial_id"],
        how="left",
        suffixes=("", "_alfred"),
    )
    image_paths: list[str | None] = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="Selecting ALFRED frames"):
        if pd.isna(row.get("traj_json")):
            image_paths.append(None)
            continue
        image_path = select_image_for_subgoal(
            row["traj_json"],
            row["trajectory_dir"],
            subgoal_idx=row.get("subgoal_idx"),
            subgoal_start=row.get("subgoal_start"),
        )
        image_paths.append(str(image_path) if image_path else None)

    merged["image_path"] = image_paths
    if output_manifest:
        output = Path(output_manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output, index=False)
    return merged

