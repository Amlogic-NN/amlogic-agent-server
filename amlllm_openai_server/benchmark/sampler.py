# -*- coding: utf-8 -*-
"""Sample manager — creates and manages fixed random samples for summary tests."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from .datasets import get_dataset

logger = logging.getLogger("benchmark.sampler")


class SampleManager:
    """Manages fixed random samples of benchmark datasets for summary tests.

    Samples are stratified by category/subject/domain where possible and
    indices are persisted to JSON so they remain fixed across all future runs.
    """

    DEFAULT_N = 500
    DEFAULT_SEED = 42

    def __init__(self, cache_dir: str | Path = "dataset"):
        self.cache_dir = Path(cache_dir)

    def create_sample(
        self,
        dataset_name: str,
        n: int = DEFAULT_N,
        seed: int = DEFAULT_SEED,
    ) -> list[Any]:
        """Create (or load existing) a fixed sample for *dataset_name*.

        Returns:
            List of dataset items (dicts for MMLU-Pro/TAU2-Bench/BFCL).
        """
        ds = get_dataset(dataset_name)
        ds.cache_dir = self.cache_dir / ds.dataset_name
        ds.cache_dir.mkdir(parents=True, exist_ok=True)
        return ds.sample(n=n, seed=seed)

    def create_all_samples(
        self,
        n: int = DEFAULT_N,
        seed: int = DEFAULT_SEED,
    ) -> dict[str, list[Any]]:
        """Create samples for all supported datasets.

        Returns:
            Mapping from dataset name to list of sampled items.
        """
        results = {}
        for name in ("mmlu", "mmlu_pro", "tau2_bench", "bfcl", "mmstar", "ocrbench"):
            try:
                results[name] = self.create_sample(name, n=n, seed=seed)
            except Exception as exc:
                logger.warning("Failed to sample %s: %s", name, exc)
        return results

    def load_sample_indices(self, dataset_name: str, n: int = DEFAULT_N) -> list[int]:
        """Load the saved sample indices for a dataset."""
        ds = get_dataset(dataset_name)
        ds.cache_dir = self.cache_dir / ds.dataset_name
        sample_path = ds.get_sample_path(n)
        if not sample_path.exists():
            raise FileNotFoundError(f"No sample found at {sample_path}. Run 'sample' first.")
        with open(sample_path, "r", encoding="utf-8") as f:
            return json.load(f)
