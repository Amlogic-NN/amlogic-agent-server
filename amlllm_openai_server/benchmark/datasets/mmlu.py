# -*- coding: utf-8 -*-
"""MMLU dataset loader.

Download from Hugging Face ``cais/mmlu`` (57 subjects, each a separate config)
and convert to 5-shot text-based multiple-choice format matching the original
Hendrycks et al. (ICLR 2021) evaluation protocol.

Prompt format per subject::

    The following are multiple choice questions (with answers) about {subject}.

    Question: {dev_q1}
    A. ... B. ... C. ... D. ...
    Answer: {dev_a1}

    ... (5 dev examples total)

    Question: {test_q}
    A. ... B. ... C. ... D. ...
    Answer:
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from .base import BenchmarkDataset

logger = logging.getLogger("benchmark.datasets.mmlu")

# All 57 MMLU subjects (each is a Hugging Face config under cais/mmlu)
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
]


class MMLUDataset(BenchmarkDataset):
    """MMLU: 5-shot multiple-choice across 57 subjects.

    Uses the original evaluation format:
    - 5 few-shot examples from the dev split (with answers shown as \\boxed{X})
    - Test question without answer
    - Subject-line prefix identifying the topic
    - System message enforcing \\boxed{X} output format
    """

    dataset_name = "mmlu"
    hf_repo = "cais/mmlu"
    # No single config/split — loaded per-subject

    _SYSTEM_PROMPT = (
        "You are a knowledgeable assistant. Answer the following multiple-choice "
        "question by selecting the correct option letter (A, B, C, or D).\n\n"
        "You MUST respond with exactly the answer in the following format:\n"
        "\\boxed{X}\n"
        "where X is the letter of the correct answer.\n\n"
        "Example:\\n"
        "If the correct answer is option B, respond with:\\n"
        "\\boxed{B}\n\n"
        "Do NOT include any other text, explanation, or formatting in your response."
    )

    def __init__(self, cache_dir: str | Path = "dataset"):
        super().__init__(cache_dir)
        self._dev_cache: dict[str, list[dict]] = {}

    @property
    def dev_examples_path(self) -> Path:
        return self.cache_dir / "dev_examples.json"

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download(self) -> None:
        if self.processed_path.exists() and self.dev_examples_path.exists():
            logger.info("MMLU already downloaded and processed at %s", self.processed_path)
            return

        logger.info("Downloading MMLU from %s (%d subjects)...", self.hf_repo, len(MMLU_SUBJECTS))
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "MMLU requires 'datasets' package. Install with: pip install datasets"
            )

        all_items: list[dict] = []
        all_dev_examples: dict[str, list[dict]] = {}
        failed_subjects: list[str] = []

        for subject in MMLU_SUBJECTS:
            try:
                ds = load_dataset(self.hf_repo, subject)
                dev_examples = self._extract_dev_examples(ds, subject)
                all_dev_examples[subject] = dev_examples

                test_items = self._convert(ds, subject, dev_examples)
                all_items.extend(test_items)
                logger.debug("  %s: %d test items, %d dev examples", subject, len(test_items), len(dev_examples))
            except Exception as exc:
                logger.warning("Failed to load subject '%s': %s", subject, exc)
                failed_subjects.append(subject)

        if failed_subjects:
            logger.warning("Failed to load %d/%d subjects: %s",
                          len(failed_subjects), len(MMLU_SUBJECTS), failed_subjects)

        self._dev_cache = all_dev_examples
        self._save_processed(all_items)
        self._save_dev_examples(all_dev_examples)
        logger.info("MMLU: %d items from %d subjects downloaded and saved.",
                    len(all_items), len(MMLU_SUBJECTS) - len(failed_subjects))

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
            logger.info("MMLU: using fixed sample of %d items.", len(indices))
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
        logger.info("MMLU: sampled %d items, saved indices.", len(selected))
        return [all_items[i] for i in selected]

    # ------------------------------------------------------------------
    # Internal: dev example extraction
    # ------------------------------------------------------------------

    def _extract_dev_examples(self, ds, subject: str) -> list[dict]:
        """Extract up to 5 dev examples for few-shot prompting.

        The MMLU dataset stores ``answer`` as an integer index (0-3).
        We convert it to a letter (A-D) for display in the prompt.
        """
        dev_examples: list[dict] = []

        if "dev" in ds:
            dev_split = ds["dev"]
            for row in dev_split:
                question = row.get("question", "")
                choices = row.get("choices", [])
                answer_raw = row.get("answer", "")
                answer_idx = self._answer_to_index(answer_raw)
                answer_letter = "ABCD"[answer_idx] if 0 <= answer_idx < 4 else "A"
                dev_examples.append({
                    "question": question,
                    "choices": list(choices),
                    "answer": answer_letter,
                    "answer_idx": answer_idx,
                })
                if len(dev_examples) >= 5:
                    break

        return dev_examples

    # ------------------------------------------------------------------
    # Internal: conversion
    # ------------------------------------------------------------------

    def _convert(self, ds, subject: str, dev_examples: list[dict]) -> list[dict]:
        """Convert a single subject's test split to 5-shot prompted items.

        Each item is a single user message containing:
        1. Subject line
        2. 5 few-shot examples with answers
        3. Test question (no answer)
        """
        items: list[dict] = []

        if "test" not in ds:
            logger.warning("No 'test' split for subject '%s'", subject)
            return items

        # Build few-shot prefix (shared across all items in this subject)
        few_shot_prefix = self._build_few_shot_prefix(subject, dev_examples)

        test_split = ds["test"]
        letters = "ABCD"

        for i, row in enumerate(test_split):
            question = row.get("question", "")
            choices = row.get("choices", [])
            answer_key = row.get("answer", "")

            # Build options
            options_map: dict[str, str] = {}
            options_text = ""
            for j, opt in enumerate(choices):
                letter = letters[j] if j < len(letters) else f"Opt{j}"
                options_map[letter] = str(opt)
                options_text += f"{letter}. {opt}\n"

            # Ground truth — MMLU stores answer as integer index (0-3)
            ans_idx = self._answer_to_index(answer_key)
            ground_truth = letters[ans_idx] if 0 <= ans_idx < len(letters) else letters[0]

            # User content: few-shot prefix + test question
            user_content = (
                f"{few_shot_prefix}"
                f"Question: {question}\n"
                f"{options_text}"
                f"Answer:"
            )

            items.append({
                "item_id": f"mmlu_{subject}_{i:04d}",
                "subject": subject,
                "difficulty": "",
                "messages": [
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "ground_truth": ground_truth,
                "options": options_map,
            })

        return items

    # ------------------------------------------------------------------
    # Internal: few-shot prefix builder
    # ------------------------------------------------------------------

    def _build_few_shot_prefix(self, subject: str, dev_examples: list[dict]) -> str:
        """Build the 5-shot prompt prefix for a subject.

        Format::

            The following are multiple choice questions (with answers) about {subject}.

            Question: ...
            A. ... B. ... C. ... D. ...
            Answer: X

            ... (5 examples)
        """
        # Human-readable subject name
        subject_display = subject.replace("_", " ")

        parts = [
            f"The following are multiple choice questions (with answers) about {subject_display}.\n",
        ]

        letters = "ABCD"
        for ex in dev_examples:
            q = ex.get("question", "")
            choices = ex.get("choices", [])
            ans_key = ex.get("answer", "")

            options_text = ""
            for j, opt in enumerate(choices):
                letter = letters[j] if j < len(letters) else f"Opt{j}"
                options_text += f"{letter}. {opt}\n"

            parts.append(
                f"Question: {q}\n"
                f"{options_text}"
                f"Answer: \\boxed{{{ans_key}}}\n"
            )

        return "\n".join(parts) + "\n\n"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _answer_to_index(answer_key) -> int:
        """Convert answer key (A/B/C/D or 0/1/2/3) to index."""
        if isinstance(answer_key, int):
            return answer_key
        if isinstance(answer_key, str) and len(answer_key) == 1:
            upper = answer_key.strip().upper()
            if upper in "ABCD":
                return ord(upper) - ord("A")
            try:
                return int(answer_key)
            except ValueError:
                pass
        return 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_processed(self, items: list[dict]) -> None:
        with open(self.processed_path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)

    def _save_dev_examples(self, dev_examples: dict[str, list[dict]]) -> None:
        with open(self.dev_examples_path, "w", encoding="utf-8") as f:
            json.dump(dev_examples, f, ensure_ascii=False, indent=2)
