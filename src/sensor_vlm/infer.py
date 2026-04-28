from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .blip2_features import Blip2FeatureExtractor
from .features import TextEmbedder, build_single_multimodal_feature
from .model import load_binary_checkpoint, predict_probability


def clarification_question(instruction: str, caption: str, probability: float) -> str:
    """Template clarification grounded in the scene caption and model confidence."""
    if probability < 0.5:
        return "No clarification needed."

    lower = instruction.lower()
    if any(token in lower for token in ("that", "it", "there", "this", "those", "them")):
        return f"In the scene described as '{caption}', which object or location do you mean?"
    if "left" in lower or "right" in lower:
        return f"In the scene described as '{caption}', from whose perspective should I interpret the direction?"
    return f"In the scene described as '{caption}', what extra detail should I use to complete the instruction?"


def predict_image_instruction(
    *,
    image_path: str | Path,
    instruction: str,
    checkpoint_path: str | Path,
    threshold: float = 0.5,
) -> dict[str, object]:
    model, _ = load_binary_checkpoint(checkpoint_path)
    blip2 = Blip2FeatureExtractor()
    text_embedder = TextEmbedder()
    feature, blip_features, _ = build_single_multimodal_feature(
        image_path=image_path,
        instruction=instruction,
        blip2=blip2,
        text_embedder=text_embedder,
    )
    probability = predict_probability(model, np.asarray(feature, dtype=np.float32))
    label = "Ambiguous" if probability >= threshold else "Not Ambiguous"
    return {
        "instruction": instruction,
        "label": label,
        "probability": probability,
        "caption": blip_features.caption,
        "caption_ambiguity_score": blip_features.caption_ambiguity_score,
        "multi_captions": blip_features.multi_captions,
        "vqa_answers": blip_features.vqa_answers,
        "clarification_question": clarification_question(
            instruction,
            blip_features.caption,
            probability,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Sensor-VLM image + instruction inference.")
    parser.add_argument("--image", required=True, help="Local image path or image URL.")
    parser.add_argument("--instruction", required=True, help="Instruction to classify.")
    parser.add_argument("--checkpoint", required=True, help="Trained multimodal MLP checkpoint.")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = predict_image_instruction(
        image_path=args.image,
        instruction=args.instruction,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
    )
    print(f"Instruction: {result['instruction']}")
    print(f"Label: {result['label']} ({result['probability']:.3f})")
    print(f"Caption: {result['caption']}")
    print(f"Caption ambiguity score: {result['caption_ambiguity_score']:.3f}")
    print(f"Clarification: {result['clarification_question']}")


if __name__ == "__main__":
    main()

