from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .blip2_features import Blip2FeatureExtractor, Blip2Features


DEFAULT_TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class MultimodalFeatureRecord:
    instruction: str
    label: int | None
    feature: np.ndarray
    caption: str
    caption_ambiguity_score: float
    vqa_text: str


class TextEmbedder:
    """Small wrapper around Sentence Transformers for consistent encoding."""

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, *, device: str | None = None):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: list[str], *, batch_size: int = 128) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, show_progress_bar=len(texts) > batch_size, batch_size=batch_size),
            dtype=np.float32,
        )


def flatten_vqa_answers(vqa_answers: dict[str, str]) -> str:
    return " ".join(f"{question} {answer}" for question, answer in vqa_answers.items()).strip()


def multimodal_feature_vector(
    *,
    instruction_embedding: np.ndarray,
    image_embedding: np.ndarray,
    caption_vqa_embedding: np.ndarray,
    caption_similarity: float,
    caption_ambiguity_score: float,
) -> np.ndarray:
    scalars = np.asarray([caption_similarity, caption_ambiguity_score], dtype=np.float32)
    return np.concatenate(
        [
            image_embedding.astype(np.float32).reshape(-1),
            instruction_embedding.astype(np.float32).reshape(-1),
            caption_vqa_embedding.astype(np.float32).reshape(-1),
            scalars,
        ]
    )


def build_single_multimodal_feature(
    *,
    image_path: str | Path,
    instruction: str,
    blip2: Blip2FeatureExtractor,
    text_embedder: TextEmbedder,
) -> tuple[np.ndarray, Blip2Features, str]:
    blip_features = blip2.extract_features(image_path)
    vqa_text = flatten_vqa_answers(blip_features.vqa_answers)
    caption_context = f"{blip_features.caption} {vqa_text}".strip()
    instruction_embedding = text_embedder.encode([instruction])[0]
    caption_vqa_embedding = text_embedder.encode([caption_context])[0]
    feature = multimodal_feature_vector(
        instruction_embedding=instruction_embedding,
        image_embedding=blip_features.qformer_pooled,
        caption_vqa_embedding=caption_vqa_embedding,
        caption_similarity=blip_features.caption_similarity,
        caption_ambiguity_score=blip_features.caption_ambiguity_score,
    )
    return feature, blip_features, vqa_text


def save_feature_cache(
    output_path: str | Path,
    *,
    features: np.ndarray,
    labels: np.ndarray | None = None,
    splits: np.ndarray | None = None,
    metadata: pd.DataFrame | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"features": features.astype(np.float32)}
    if labels is not None:
        payload["labels"] = labels.astype(np.int64)
    if splits is not None:
        payload["splits"] = splits.astype(str)
    np.savez_compressed(output, **payload)
    if metadata is not None:
        metadata.to_json(output.with_suffix(".metadata.json"), orient="records", indent=2)
    return output


def load_feature_cache(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def build_text_baseline_cache(
    labels_df: pd.DataFrame,
    output_path: str | Path,
    *,
    text_columns: list[str] | None = None,
    text_model_name: str = DEFAULT_TEXT_MODEL,
    batch_size: int = 128,
) -> Path:
    embedder = TextEmbedder(text_model_name)
    columns = text_columns or ["instruction"]
    missing = set(columns) - set(labels_df.columns)
    if missing:
        raise ValueError(f"Dataframe is missing text columns: {sorted(missing)}")
    texts = (
        labels_df[columns]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .tolist()
    )
    embeddings = embedder.encode(texts, batch_size=batch_size)
    return save_feature_cache(
        output_path,
        features=embeddings,
        labels=labels_df["ambiguous"].to_numpy(),
        splits=labels_df["split"].astype(str).to_numpy(),
        metadata=labels_df,
    )


def build_multimodal_cache_from_manifest(
    manifest: pd.DataFrame,
    output_path: str | Path,
    *,
    image_column: str = "image_path",
    instruction_column: str = "instruction",
    label_column: str = "ambiguous",
    split_column: str = "split",
    text_model_name: str = DEFAULT_TEXT_MODEL,
    blip2_model_name: str = "Salesforce/blip2-opt-2.7b",
) -> Path:
    """Extract and cache multimodal features for rows containing image paths."""
    required = {image_column, instruction_column}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    text_embedder = TextEmbedder(text_model_name)
    blip2 = Blip2FeatureExtractor(blip2_model_name)
    records: list[MultimodalFeatureRecord] = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting BLIP-2 features"):
        feature, blip_features, vqa_text = build_single_multimodal_feature(
            image_path=row[image_column],
            instruction=str(row[instruction_column]),
            blip2=blip2,
            text_embedder=text_embedder,
        )
        label = int(row[label_column]) if label_column in manifest.columns and pd.notna(row[label_column]) else None
        records.append(
            MultimodalFeatureRecord(
                instruction=str(row[instruction_column]),
                label=label,
                feature=feature,
                caption=blip_features.caption,
                caption_ambiguity_score=blip_features.caption_ambiguity_score,
                vqa_text=vqa_text,
            )
        )

    features = np.stack([record.feature for record in records])
    labels = (
        np.asarray([record.label for record in records], dtype=np.int64)
        if all(record.label is not None for record in records)
        else None
    )
    splits = manifest[split_column].astype(str).to_numpy() if split_column in manifest.columns else None
    metadata = manifest.copy()
    metadata["blip2_caption"] = [record.caption for record in records]
    metadata["caption_ambiguity_score"] = [record.caption_ambiguity_score for record in records]
    metadata["vqa_text"] = [record.vqa_text for record in records]

    return save_feature_cache(output_path, features=features, labels=labels, splits=splits, metadata=metadata)


def _split_image_paths(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def build_multiview_cache_from_manifest(
    manifest: pd.DataFrame,
    output_path: str | Path,
    *,
    image_paths_column: str = "image_paths",
    instruction_column: str = "instruction",
    label_column: str = "ambiguous",
    split_column: str = "split",
    text_model_name: str = DEFAULT_TEXT_MODEL,
    blip2_model_name: str = "Salesforce/blip2-opt-2.7b",
    max_views: int | None = None,
) -> Path:
    """Extract BLIP-2 features for multiple views and mean-pool them per row."""
    required = {image_paths_column, instruction_column}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    text_embedder = TextEmbedder(text_model_name)
    blip2 = Blip2FeatureExtractor(blip2_model_name)
    features: list[np.ndarray] = []
    captions: list[str] = []
    ambiguity_scores: list[float] = []
    view_counts: list[int] = []

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting multi-view BLIP-2 features"):
        image_paths = _split_image_paths(row[image_paths_column])
        if max_views:
            image_paths = image_paths[:max_views]
        if not image_paths:
            raise ValueError(f"Row has no image paths in column {image_paths_column}: {row.to_dict()}")

        row_features: list[np.ndarray] = []
        row_captions: list[str] = []
        row_scores: list[float] = []
        for image_path in image_paths:
            feature, blip_features, _ = build_single_multimodal_feature(
                image_path=image_path,
                instruction=str(row[instruction_column]),
                blip2=blip2,
                text_embedder=text_embedder,
            )
            row_features.append(feature)
            row_captions.append(blip_features.caption)
            row_scores.append(blip_features.caption_ambiguity_score)

        features.append(np.mean(np.stack(row_features), axis=0).astype(np.float32))
        captions.append(" | ".join(row_captions))
        ambiguity_scores.append(float(np.mean(row_scores)))
        view_counts.append(len(row_features))

    labels = manifest[label_column].to_numpy(dtype=np.int64) if label_column in manifest.columns else None
    splits = manifest[split_column].astype(str).to_numpy() if split_column in manifest.columns else None
    metadata = manifest.copy()
    metadata["blip2_captions"] = captions
    metadata["caption_ambiguity_score"] = ambiguity_scores
    metadata["view_count"] = view_counts

    return save_feature_cache(
        output_path,
        features=np.stack(features),
        labels=labels,
        splits=splits,
        metadata=metadata,
    )


def load_torch_blip2_feature(path: str | Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu")

