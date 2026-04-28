from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import requests
import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, Blip2Processor


DEFAULT_MODEL_NAME = "Salesforce/blip2-opt-2.7b"

DEFAULT_CAPTION_STRATEGIES: dict[str, dict[str, Any]] = {
    "greedy": {"do_sample": False, "max_new_tokens": 50},
    "beam_5": {"do_sample": False, "num_beams": 5, "max_new_tokens": 50},
    "nucleus_0_9": {"do_sample": True, "top_p": 0.9, "temperature": 0.7, "max_new_tokens": 50},
    "high_temp": {"do_sample": True, "temperature": 1.2, "max_new_tokens": 50},
    "nucleus_high_temp": {
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 1.2,
        "max_new_tokens": 50,
    },
}

DEFAULT_VQA_QUESTIONS = [
    "What is the main subject of this image?",
    "What is happening in this image?",
    "Where was this photo taken?",
    "How many people are in this image?",
    "What colors are prominent in this image?",
    "What is in the background?",
    "What emotion or mood does this image convey?",
]


@dataclass
class Blip2Features:
    caption: str
    multi_captions: dict[str, str]
    caption_similarity: float
    caption_ambiguity_score: float
    vqa_answers: dict[str, str]
    qformer_embeddings: torch.Tensor
    qformer_pooled: np.ndarray
    llm_hidden_state: torch.Tensor | None = None


def load_image(image: str | Path | Image.Image) -> Image.Image:
    """Load an RGB PIL image from a path, URL, or existing PIL image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    image_text = str(image)
    if image_text.startswith(("http://", "https://")):
        response = requests.get(image_text, stream=True, timeout=60)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")

    return Image.open(image_text).convert("RGB")


def word_overlap(text_a: str, text_b: str) -> float:
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def caption_variance(captions: list[str]) -> tuple[float, float]:
    """Return average word-overlap similarity and 1-similarity ambiguity score."""
    if len(captions) < 2:
        return 1.0, 0.0
    pairs = list(combinations(range(len(captions)), 2))
    similarities = [word_overlap(captions[i], captions[j]) for i, j in pairs]
    avg_similarity = float(sum(similarities) / len(similarities))
    return avg_similarity, 1.0 - avg_similarity


class Blip2FeatureExtractor:
    """Frozen BLIP-2 feature extractor for image captions, VQA, and embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def _processor_inputs(self, image: Image.Image, text: str | None = None) -> dict[str, torch.Tensor]:
        kwargs: dict[str, Any] = {"images": image, "return_tensors": "pt"}
        if text is not None:
            kwargs["text"] = text
        return self.processor(**kwargs).to(self.device, self.dtype)

    def generate_captions(
        self,
        image: str | Path | Image.Image,
        strategies: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, str]:
        img = load_image(image)
        inputs = self._processor_inputs(img)
        strategies = strategies or DEFAULT_CAPTION_STRATEGIES
        captions: dict[str, str] = {}

        with torch.no_grad():
            for name, params in strategies.items():
                generated_ids = self.model.generate(**inputs, **params)
                captions[name] = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0].strip()
        return captions

    def answer_questions(
        self,
        image: str | Path | Image.Image,
        questions: list[str] | None = None,
    ) -> dict[str, str]:
        img = load_image(image)
        answers: dict[str, str] = {}
        with torch.no_grad():
            for question in questions or DEFAULT_VQA_QUESTIONS:
                prompt = f"Question: {question} Answer:"
                inputs = self._processor_inputs(img, prompt)
                generated_ids = self.model.generate(**inputs, max_new_tokens=30)
                answer = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )[0].strip()
                answers[question] = answer
        return answers

    def extract_qformer_embeddings(self, image: str | Path | Image.Image) -> torch.Tensor:
        img = load_image(image)
        pixel_values = self.processor(images=img, return_tensors="pt")["pixel_values"].to(
            self.device,
            self.dtype,
        )

        with torch.no_grad():
            try:
                qformer_outputs = self.model.get_qformer_features(pixel_values=pixel_values)
                qformer_embeddings = qformer_outputs[0]
            except AttributeError:
                vision_outputs = self.model.vision_model(pixel_values=pixel_values)
                image_embeds = vision_outputs[0]
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(self.device)
                query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
                qformer_outputs = self.model.qformer(
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                )
                qformer_embeddings = qformer_outputs.last_hidden_state
        return qformer_embeddings.detach().cpu()

    def extract_llm_hidden_state(self, image: str | Path | Image.Image) -> torch.Tensor | None:
        img = load_image(image)
        inputs = self._processor_inputs(img)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        if not outputs.hidden_states:
            return None
        return outputs.hidden_states[-1][-1].detach().cpu()

    def extract_features(
        self,
        image: str | Path | Image.Image,
        *,
        include_llm_hidden_state: bool = False,
    ) -> Blip2Features:
        img = load_image(image)
        captions = self.generate_captions(img)
        avg_similarity, ambiguity_score = caption_variance(list(captions.values()))
        vqa_answers = self.answer_questions(img)
        qformer_embeddings = self.extract_qformer_embeddings(img)
        qformer_pooled = qformer_embeddings.mean(dim=1).squeeze(0).numpy()
        llm_hidden_state = self.extract_llm_hidden_state(img) if include_llm_hidden_state else None

        return Blip2Features(
            caption=captions.get("greedy", next(iter(captions.values()), "")),
            multi_captions=captions,
            caption_similarity=avg_similarity,
            caption_ambiguity_score=ambiguity_score,
            vqa_answers=vqa_answers,
            qformer_embeddings=qformer_embeddings,
            qformer_pooled=qformer_pooled,
            llm_hidden_state=llm_hidden_state,
        )

    @staticmethod
    def save_features(features: Blip2Features, output_path: str | Path) -> Path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "caption": features.caption,
                "multi_captions": features.multi_captions,
                "caption_similarity": features.caption_similarity,
                "caption_ambiguity_score": features.caption_ambiguity_score,
                "vqa_answers": features.vqa_answers,
                "qformer_embeddings": features.qformer_embeddings,
                "qformer_pooled": features.qformer_pooled,
                "llm_hidden_state": features.llm_hidden_state,
            },
            output,
        )
        return output

