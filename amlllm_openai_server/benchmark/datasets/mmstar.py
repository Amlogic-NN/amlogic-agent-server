# -*- coding: utf-8 -*-
"""MMStar dataset loader.

Download from Hugging Face ``Lin-Chen/MMStar`` and convert to
text-based multiple-choice format with image data (base64-encoded).

MMStar is a vision-indispensable multi-modal benchmark with 1500 samples
across 6 core capability categories.
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

logger = logging.getLogger("benchmark.datasets.mmstar")

# MMStar categories (18 detailed axes grouped into 6 core capabilities)
MMSTAR_CATEGORIES = [
    "coarse_perception",
    "fine-grained_perception",
    "instance_reasoning",
    "logical_reasoning",
    "science_technology",
    "math",
]


class MMStarDataset(BenchmarkDataset):
    dataset_name = "mmstar"
    hf_repo = "Lin-Chen/MMStar"
    hf_config = "val"
    hf_split = "val"
    default_sample_size = 100

    def __init__(self, cache_dir: str | Path = "dataset", resize: tuple[int, int] | None = None):
        super().__init__(cache_dir)
        self.resize = resize  # (width, height) or None for original size
        self._system_prompt = (
            "You are a knowledgeable assistant with vision capabilities. "
            "Analyze the provided image carefully and answer the following "
            "multiple-choice question by selecting the correct option letter.\n\n"
            "You may think through the problem step by step before giving your final answer. "
            "However, your final answer MUST be placed on its own line at the very end "
            "of your response in the following format:\n"
            "\\boxed{X}\n"
            "where X is the letter of the correct answer (A, B, C, or D).\n\n"
            "Example:\n"
            "Let me analyze the image step by step. [your reasoning here...]\n"
            "Therefore, the correct answer is B.\n"
            "\\boxed{B}\n\n"
            "The answer inside \\boxed{} on the last line is what will be evaluated."
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self) -> None:
        if self.processed_path.exists():
            logger.info("MMStar already downloaded and processed at %s", self.processed_path)
            return

        logger.info("Downloading MMStar from %s (config=%s, split=%s)...",
                     self.hf_repo, self.hf_config, self.hf_split)
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "MMStar requires 'datasets' package. Install with: pip install datasets"
            )

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        items = self._convert(ds)
        self._save_processed(items)
        logger.info("MMStar: %d items downloaded and saved.", len(items))

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
            logger.info("MMStar: loaded %d items with resize to %dx%d.", len(raw), *self.resize)
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
            logger.info("MMStar: using fixed sample of %d items.", len(indices))
            return [all_items[i] for i in indices]

        rng = random.Random(seed)

        # Stratify by category
        by_category: dict[str, list[int]] = {}
        for i, item in enumerate(all_items):
            category = item.get("subject", item.get("l2_category", "other"))
            by_category.setdefault(category, []).append(i)

        n_per_category = max(1, n // max(len(by_category), 1))
        selected: list[int] = []
        for category, idx_list in by_category.items():
            k = min(n_per_category, len(idx_list))
            selected.extend(rng.sample(idx_list, k))

        # If we didn't get enough, fill randomly
        if len(selected) < n:
            remaining = [i for i in range(len(all_items)) if i not in set(selected)]
            extra = rng.sample(remaining, min(n - len(selected), len(remaining)))
            selected.extend(extra)

        selected = sorted(selected)[:n]
        with open(sample_path, "w", encoding="utf-8") as f:
            _json.dump(selected, f)
        logger.info("MMStar: sampled %d items, saved indices.", len(selected))
        return [all_items[i] for i in selected]

    # ------------------------------------------------------------------
    # Internal conversion
    # ------------------------------------------------------------------

    def _convert(self, ds) -> list[dict]:
        """Convert HuggingFace dataset to unified dict format.

        MMStar embeds options within the question text in two formats:
        1. "Question?\\nOptions: A: ..., B: ..., C: ..., D: ..."
        2. "Hint: ...\\nQuestion: ...\\nChoices:\\n(A) ...\\n(B) ...\\n(C) ...\\n(D) ..."

        We parse out the options for metadata but keep the full original
        question text (with options inline) as the user prompt.
        """
        import re

        items = []
        for i in range(len(ds)):
            row = ds[i]
            question_full = row.get("question", "")
            answer = row.get("answer", "")
            category = row.get("category", row.get("l2_category", "other"))
            l2_category = row.get("l2_category", category)
            image = row.get("image")

            options_map = self._parse_options(question_full)

            # Determine ground truth letter
            ground_truth = ""
            if answer and len(answer) == 1 and answer.isalpha():
                ground_truth = answer.upper()
            else:
                # Extract letter from answer string
                m = re.search(r'([A-Da-d])', str(answer))
                if m:
                    ground_truth = m.group(1).upper()

            if not ground_truth:
                logger.warning("Could not determine ground truth for item %d, answer=%s", i, answer)
                ground_truth = "A"

            # Build multimodal user message content
            user_content: list[dict] = []

            # Add image as base64 data URL if available
            if image is not None:
                image_b64 = self._pil_to_base64(image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                })

            # Use full question text as-is (options embedded)
            user_content.append({
                "type": "text",
                "text": f"{question_full}\nAnswer:",
            })

            items.append({
                "item_id": f"mmstar_{i:05d}",
                "subject": str(category),
                "l2_category": str(l2_category),
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "ground_truth": ground_truth,
                "options": options_map,
            })

        return items

    @staticmethod
    def _parse_options(question_text: str) -> dict[str, str]:
        """Parse options dict from MMStar question text.

        Handles two formats:
        1. "...\\nOptions: A: text, B: text, C: text, D: text"
        2. "...\\nChoices:\\n(A) text\\n(B) text\\n(C) text\\n(D) text"

        Falls back to regex extraction of A-D patterns at end of text.
        """
        import re

        # Format 1: "Options: A: ..., B: ..., C: ..., D: ..."
        m = re.search(r'Options?\s*:\s*(.+)$', question_text, re.DOTALL)
        if m:
            opts_str = m.group(1)
            # Parse "A: text, B: text" or "A: text. B: text."
            # Match: letter colon space text (non-greedy, stop at next letter: or end)
            result: dict[str, str] = {}
            for opt_m in re.finditer(r'([A-Z]):\s*(.+?)(?=\s*[,;.]?\s*[A-Z]:\s|$)', opts_str):
                letter = opt_m.group(1)
                text = opt_m.group(2).strip().rstrip('.').rstrip(',')
                result[letter] = text
            if result:
                return result

        # Format 2: "Choices:\\n(A) text\\n(B) text\\n(C) text\\n(D) text"
        m = re.search(r'Choices?\s*:\s*\n(.+)$', question_text, re.DOTALL)
        if m:
            choices_str = m.group(1)
            result = {}
            for opt_m in re.finditer(r'\(([A-Z])\)\s*(.+?)(?=\n\(|\Z)', choices_str):
                letter = opt_m.group(1)
                text = opt_m.group(2).strip()
                result[letter] = text
            if result:
                return result

        # Fallback: scan for "A: ... B: ... C: ... D: ..." at end
        m = re.search(
            r'(?:^|\n)\s*'
            r'A:\s*(.+?)(?:\s*[,;.]?\s+)B:\s*(.+?)(?:\s*[,;.]?\s+)'
            r'C:\s*(.+?)(?:\s*[,;.]?\s+)D:\s*(.+?)(?:\s*[,;.]?\s*)$',
            question_text, re.DOTALL,
        )
        if m:
            return {"A": m.group(1).strip(), "B": m.group(2).strip(),
                    "C": m.group(3).strip(), "D": m.group(4).strip()}

        # Second fallback: "(A) ... (B) ... (C) ... (D) ..."
        m = re.search(
            r'(?:^|\n)\s*'
            r'\(A\)\s*(.+?)(?:\s*\n\s*)\(B\)\s*(.+?)(?:\s*\n\s*)'
            r'\(C\)\s*(.+?)(?:\s*\n\s*)\(D\)\s*(.+?)$',
            question_text, re.DOTALL,
        )
        if m:
            return {"A": m.group(1).strip(), "B": m.group(2).strip(),
                    "C": m.group(3).strip(), "D": m.group(4).strip()}

        return {}

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
        """Decode base64 images in items, resize, and re-encode.

        Modifies each item's user message in-place: decodes the base64
        data URL -> PIL Image -> resize -> re-encode as base64 JPEG.
        """
        from PIL import Image

        for item in items:
            user_content = item["messages"][1]["content"]
            if not isinstance(user_content, list):
                continue
            for part in user_content:
                if part.get("type") != "image_url":
                    continue
                url = part["image_url"]["url"]
                # Extract base64 data from data URL
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
