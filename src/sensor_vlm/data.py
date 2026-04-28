from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .paths import DATA_DIR


DIALFRED_URL = (
    "https://raw.githubusercontent.com/xfgao/DialFRED/main/data/"
    "dialfred_human_qa.csv"
)

DIALFRED_COLUMNS = [
    "split",
    "task_id",
    "trial_id",
    "room_type",
    "task_type",
    "subgoal_start",
    "subgoal_end",
    "num_actions",
    "subgoal_idx",
    "instruction",
    "verb",
    "noun1",
    "noun2",
    "question_type",
    "question",
    "answer",
    "necessary",
]

SPLIT_ALIASES = {
    "training": "train",
    "train": "train",
    "valid_seen": "valid_seen",
    "validation_seen": "valid_seen",
    "valid_unseen": "valid_unseen",
    "validation_unseen": "valid_unseen",
}


def download_dialfred(
    output_path: str | Path | None = None,
    *,
    force: bool = False,
    url: str = DIALFRED_URL,
) -> Path:
    """Download the DialFRED human QA CSV if it is not already cached."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(output_path) if output_path else DATA_DIR / "dialfred_human_qa.csv"
    if output.exists() and not force:
        return output

    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output.write_bytes(response.content)
    return output


def _read_dialfred_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read DialFRED CSV across the header variants seen in public mirrors."""
    path = Path(csv_path)
    df = pd.read_csv(path, low_memory=False)
    if set(DIALFRED_COLUMNS).issubset(df.columns):
        return df[DIALFRED_COLUMNS].copy()

    df = pd.read_csv(path, header=None, names=DIALFRED_COLUMNS, low_memory=False)
    if str(df.iloc[0]["split"]).strip().lower() in {"split", "data splits"}:
        df = df.iloc[1:].reset_index(drop=True)
    return df


def normalize_necessary(value: object) -> int:
    """Map DialFRED's necessary column to 1 = ambiguous, 0 = not ambiguous."""
    text = str(value).strip().lower()
    return 1 if text in {"true", "1", "yes", "y"} else 0


def _unique_nonempty(values: Iterable[object]) -> list[str]:
    cleaned = {
        str(value).strip()
        for value in values
        if pd.notna(value) and str(value).strip() and str(value).strip() != "nan"
    }
    return sorted(cleaned)


def load_dialfred(csv_path: str | Path | None = None, *, download: bool = True) -> pd.DataFrame:
    """Load and clean DialFRED human QA rows."""
    path = Path(csv_path) if csv_path else DATA_DIR / "dialfred_human_qa.csv"
    if download and not path.exists():
        path = download_dialfred(path)

    df = _read_dialfred_csv(path)
    df["instruction"] = df["instruction"].astype(str).str.strip()
    df = df[df["instruction"].ne("") & df["instruction"].ne("nan")].copy()
    df["split"] = (
        df["split"].astype(str).str.strip().str.lower().map(SPLIT_ALIASES).fillna(df["split"])
    )
    df["ambiguous"] = df["necessary"].map(normalize_necessary).astype(int)
    df["question_type"] = df["question_type"].astype(str).str.strip().str.lower()
    return df.reset_index(drop=True)


def prepare_instruction_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate annotation rows into one label row per instruction and split."""
    grouped = (
        df.groupby(["instruction", "split"], dropna=False)
        .agg(
            ambiguous_mean=("ambiguous", "mean"),
            num_annotations=("ambiguous", "count"),
            task_id=("task_id", "first"),
            trial_id=("trial_id", "first"),
            room_type=("room_type", "first"),
            task_type=("task_type", "first"),
            subgoal_start=("subgoal_start", "first"),
            subgoal_end=("subgoal_end", "first"),
            subgoal_idx=("subgoal_idx", "first"),
            question_types=("question_type", _unique_nonempty),
            questions=("question", _unique_nonempty),
            answers=("answer", _unique_nonempty),
        )
        .reset_index()
    )
    grouped["ambiguous"] = (grouped["ambiguous_mean"] > 0.5).astype(int)
    grouped["ambiguity_type"] = grouped["question_types"].map(
        lambda values: values[0] if values else "none"
    )
    return grouped


def load_instruction_labels(
    csv_path: str | Path | None = None, *, download: bool = True
) -> pd.DataFrame:
    """Load DialFRED and return instruction-level labels ready for training."""
    return prepare_instruction_labels(load_dialfred(csv_path, download=download))


def print_dataset_summary(df: pd.DataFrame) -> None:
    """Print a compact summary matching the proof-of-concept notebook."""
    total = len(df)
    ambiguous = int(df["ambiguous"].sum())
    print(f"Rows: {total:,}")
    print(f"Ambiguous: {ambiguous:,}")
    print(f"Not ambiguous: {total - ambiguous:,}")
    print(f"Ratio: {df['ambiguous'].mean():.2%} ambiguous")
    if "split" in df:
        print(f"Splits: {df['split'].value_counts().to_dict()}")

