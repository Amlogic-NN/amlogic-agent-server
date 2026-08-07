# -*- coding: utf-8 -*-
"""OCRBench dataset loader.

Download from Hugging Face ``echo840/OCRBench`` and convert to
multimodal format with image data (base64-encoded).

OCRBench is a comprehensive OCR evaluation benchmark with 1000 question-answer
pairs across 5 task types: Text Recognition, Scene Text-Centric VQA,
Document-Oriented VQA, Key Information Extraction, and Handwritten
Mathematical Expression Recognition.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import random
from pathlib import Path
from typing import Any

from .base import BenchmarkDataset

logger = logging.getLogger("benchmark.datasets.ocrbench")

# OCRBench question types (5 categories for stratified sampling)
OCRBENCH_QUESTION_TYPES = [
    "Regular Text Recognition",
    "Scene Text-Centric VQA",
    "Document-Oriented VQA",
    "Key Information Extraction",
    "Handwritten Mathematical Expression Recognition",
]


class OCRBenchDataset(BenchmarkDataset):
    dataset_name = "ocrbench"
    hf_repo = "echo840/OCRBench"
    hf_config = "default"
    hf_split = "test"
    default_sample_size = 100

    def __init__(self, cache_dir: str | Path = "dataset", resize: tuple[int, int] | None = None):
        super().__init__(cache_dir)
        self.resize = resize  # (width, height) or None for original size
        self._system_prompt = (
            "You are a precise OCR assistant with vision capabilities. "
            "Analyze the provided image carefully and answer the question "
            "by extracting or reading the text shown in the image.\n\n"
            "Give ONLY the exact answer text — no explanations, no extra words, "
            "no punctuation beyond what is in the image. "
            "Your response should be as concise as possible."
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self) -> None:
        if self.processed_path.exists():
            logger.info("OCRBench already downloaded and processed at %s", self.processed_path)
            return

        logger.info("Downloading OCRBench from %s (config=%s, split=%s)...",
                     self.hf_repo, self.hf_config, self.hf_split)
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "OCRBench requires 'datasets' package. Install with: pip install datasets"
            )

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        items = self._convert(ds)
        self._save_processed(items)
        logger.info("OCRBench: %d items downloaded and saved.", len(items))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, force_reload: bool = False) -> list[dict]:
        if force_reload or not self.processed_path.exists():
            self.download()

        with open(self.processed_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if self.resize is not None:
            raw = self._resize_items(raw, self.resize)
            logger.info("OCRBench: loaded %d items with resize to %dx%d.", len(raw), *self.resize)
        return raw

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self, n: int = 100, seed: int = 42) -> list[dict]:
        import json as _json

        all_items = self.load()
        sample_path = self.get_sample_path(n)

        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                indices = _json.load(f)
            logger.info("OCRBench: using fixed sample of %d items.", len(indices))
            return [all_items[i] for i in indices]

        rng = random.Random(seed)

        # Stratify by question_type
        by_type: dict[str, list[int]] = {}
        for i, item in enumerate(all_items):
            qtype = item.get("subject", item.get("question_type", "other"))
            by_type.setdefault(qtype, []).append(i)

        n_per_type = max(1, n // max(len(by_type), 1))
        selected: list[int] = []
        for qtype, idx_list in by_type.items():
            k = min(n_per_type, len(idx_list))
            selected.extend(rng.sample(idx_list, k))

        # If we didn't get enough, fill randomly
        if len(selected) < n:
            remaining = [i for i in range(len(all_items)) if i not in set(selected)]
            extra = rng.sample(remaining, min(n - len(selected), len(remaining)))
            selected.extend(extra)

        selected = sorted(selected)[:n]
        with open(sample_path, "w", encoding="utf-8") as f:
            _json.dump(selected, f)
        logger.info("OCRBench: sampled %d items, saved indices.", len(selected))
        return [all_items[i] for i in selected]

    # ------------------------------------------------------------------
    # Internal conversion
    # ------------------------------------------------------------------

    def _convert(self, ds) -> list[dict]:
        """Convert HuggingFace dataset to unified dict format.

        OCRBench fields:
        - dataset: source dataset name (e.g., "IIIT5K")
        - question: the question text (e.g., "what is written in the image?")
        - question_type: task category
        - answer: list of accepted ground truth answer strings
        - image: PIL Image
        """
        items = []
        for i in range(len(ds)):
            row = ds[i]
            question_text = row.get("question", "")
            answer = row.get("answer", [])
            question_type = row.get("question_type", "Other")
            source_dataset = row.get("dataset", "unknown")
            image = row.get("image")

            # Normalize ground truth: ensure it's a list of strings
            ground_truth: list[str] = []
            if isinstance(answer, list):
                ground_truth = [str(a) for a in answer]
            elif isinstance(answer, str):
                ground_truth = [answer]
            else:
                ground_truth = [str(answer)]

            # Build multimodal user message content
            user_content: list[dict] = []

            # Add image as base64 data URL if available
            if image is not None:
                image_b64 = self._pil_to_base64(image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                })

            # Add question text
            user_content.append({
                "type": "text",
                "text": f"{question_text}",
            })

            items.append({
                "item_id": f"ocrbench_{i:05d}",
                "subject": str(question_type),
                "source_dataset": str(source_dataset),
                "question_type": str(question_type),
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "ground_truth": ground_truth,
                "options": {},
            })

        return items

    # ------------------------------------------------------------------
    # Image encoding / resizing
    # ------------------------------------------------------------------

    @staticmethod
    def _pil_to_base64(image, resize: tuple[int, int] | None = None) -> str:
        """Convert a PIL Image to base64-encoded JPEG string.

        Args:
            image: PIL Image, file-like object, or bytes.
            resize: Optional (width, height) tuple to resize the image to.
        """
        # Normalize to PIL Image
        if hasattr(image, "save"):
            pil_image = image
        elif hasattr(image, "read"):
            from PIL import Image
            pil_image = Image.open(image)
        else:
            from PIL import Image
            pil_image = Image.open(io.BytesIO(image if isinstance(image, bytes) else bytes(image)))

        # Convert mode if needed
        if pil_image.mode in ("RGBA", "P", "LA"):
            pil_image = pil_image.convert("RGB")

        # Apply resize if requested
        if resize is not None:
            pil_image = pil_image.resize(resize)

        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _resize_items(self, items: list[dict], size: tuple[int, int]) -> list[dict]:
        """Decode base64 images in items, resize, and re-encode."""
        from PIL import Image

        for item in items:
            user_content = item["messages"][1]["content"]
            if not isinstance(user_content, list):
                continue
            for part in user_content:
                if part.get("type") != "image_url":
                    continue
                url = part["image_url"]["url"]
                if url.startswith("data:image/"):
                    header, b64_data = url.split(",", 1)
                    img_bytes = base64.b64decode(b64_data)
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    if pil_img.mode in ("RGBA", "P", "LA"):
                        pil_img = pil_img.convert("RGB")
                    pil_img = pil_img.resize(size)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="JPEG", quality=90)
                    new_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                    part["image_url"]["url"] = f"data:image/jpeg;base64,{new_b64}"
        return items

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_processed(self, items: list[dict]) -> None:
        with open(self.processed_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
