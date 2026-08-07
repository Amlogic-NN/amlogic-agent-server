# -*- coding: utf-8 -*-
"""MMLU-Pro dataset loader.

Download from Hugging Face ``TIGER-Lab/MMLU-Pro`` and convert to
text-based multiple-choice format suitable for 8K context evaluation.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from .base import BenchmarkDataset

logger = logging.getLogger("benchmark.datasets.mmlu_pro")

# MMLU-Pro subjects
MMLU_PRO_SUBJECTS = [
    "biology", "business", "chemistry", "computer_science", "economics",
    "engineering", "health", "history", "law", "math", "philosophy",
    "physics", "psychology", "other",
]


class MMLUProDataset(BenchmarkDataset):
    dataset_name = "mmlu_pro"
    hf_repo = "TIGER-Lab/MMLU-Pro"
    hf_config = "default"
    hf_split = "test"

    def __init__(self, cache_dir: str | Path = "dataset"):
        super().__init__(cache_dir)
        self._system_prompt = (
            "You are a knowledgeable assistant. Answer the following multiple-choice "
            "question by selecting the correct option letter.\n\n"
            "You may think through the problem step by step before giving your final answer. "
            "However, your final answer MUST be placed on its own line at the very end "
            "of your response in the following format:\n"
            "\\boxed{X}\n"
            "where X is the letter of the correct answer (A, B, C, D, E, F, G, H, I, or J).\n\n"
            "Example:\\n"
            "Let me analyze this step by step. [your reasoning here...]\\n"
            "Therefore, the correct answer is A.\\n"
            "\\boxed{A}\n\n"
            "The answer inside \\boxed{} on the last line is what will be evaluated."
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self) -> None:
        if self.processed_path.exists():
            logger.info("MMLU-Pro already downloaded and processed at %s", self.processed_path)
            return

        logger.info("Downloading MMLU-Pro from %s...", self.hf_repo)
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "MMLU-Pro requires 'datasets' package. Install with: pip install datasets"
            )

        ds = load_dataset(self.hf_repo, self.hf_config, split=self.hf_split)
        items = self._convert(ds)
        self._save_processed(items)
        logger.info("MMLU-Pro: %d items downloaded and saved.", len(items))

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self, force_reload: bool = False) -> list[dict]:
        if force_reload or not self.processed_path.exists():
            self.download()

        with open(self.processed_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return raw

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(self, n: int = 500, seed: int = 42) -> list[dict]:
        import json as _json

        all_items = self.load()
        sample_path = self.get_sample_path(n)

        if sample_path.exists():
            with open(sample_path, "r", encoding="utf-8") as f:
                indices = _json.load(f)
            logger.info("MMLU-Pro: using fixed sample of %d items.", len(indices))
            return [all_items[i] for i in indices]

        rng = random.Random(seed)

        # Stratify by subject
        by_subject: dict[str, list[int]] = {}
        for i, item in enumerate(all_items):
            subject = item.get("subject", "other")
            by_subject.setdefault(subject, []).append(i)

        n_per_subject = max(1, n // max(len(by_subject), 1))
        selected: list[int] = []
        for subject, idx_list in by_subject.items():
            k = min(n_per_subject, len(idx_list))
            selected.extend(rng.sample(idx_list, k))

        # If we didn't get enough, fill randomly
        if len(selected) < n:
            remaining = [i for i in range(len(all_items)) if i not in set(selected)]
            extra = rng.sample(remaining, min(n - len(selected), len(remaining)))
            selected.extend(extra)

        selected = sorted(selected)[:n]
        with open(sample_path, "w", encoding="utf-8") as f:
            _json.dump(selected, f)
        logger.info("MMLU-Pro: sampled %d items, saved indices.", len(selected))
        return [all_items[i] for i in selected]

    # ------------------------------------------------------------------
    # Internal conversion
    # ------------------------------------------------------------------

    def _convert(self, ds) -> list[dict]:
        """Convert HuggingFace dataset to unified dict format."""
        items = []
        for i, row in enumerate(ds):
            question = row.get("question", "")
            options_raw = row.get("options", [])
            answer = row.get("answer", "")
            answer_index = row.get("answer_index", -1)
            subject = row.get("category", row.get("subject", "other"))
            difficulty = row.get("src", "")

            # Build options mapping
            letters = "ABCDEFGHIJ"[:len(options_raw)]
            options_map = {}
            options_text = ""
            for j, opt in enumerate(options_raw):
                letter = letters[j] if j < len(letters) else f"Opt{j}"
                options_map[letter] = str(opt)
                options_text += f"{letter}. {opt}\n"

            # Determine correct answer letter
            if answer and len(answer) == 1 and answer.isalpha() and answer.upper() in letters:
                ground_truth = answer.upper()
            elif isinstance(answer_index, int) and 0 <= answer_index < len(options_raw):
                ground_truth = letters[answer_index]
            else:
                # Fallback: try to find answer in options
                ground_truth = letters[0]  # default, will be scored as unknown
                logger.warning("Could not determine ground truth for item %d", i)

            user_content = f"{question}\n\n{options_text}Answer:"

            items.append({
                "item_id": f"mmlu_pro_{i:05d}",
                "subject": subject,
                "difficulty": str(difficulty),
                "messages": [
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "ground_truth": ground_truth,
                "options": options_map,
            })

        return items

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_processed(self, items: list[dict]) -> None:
        with open(self.processed_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
