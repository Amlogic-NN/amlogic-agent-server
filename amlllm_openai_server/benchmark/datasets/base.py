# -*- coding: utf-8 -*-
"""Abstract base class for benchmark dataset loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BenchmarkDataset(ABC):
    """Abstract base for loading and processing benchmark datasets.

    Each dataset subclass handles:
    - Downloading from Hugging Face
    - Converting to a unified internal representation
    - Caching processed data locally
    - Sampling items for summary tests
    """

    dataset_name: str = ""
    hf_repo: str = ""
    hf_config: str = "default"
    hf_split: str = "test"
    default_sample_size: int = 500

    def __init__(self, cache_dir: str | Path = "dataset"):
        self.cache_dir = Path(cache_dir) / self.dataset_name
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def processed_path(self) -> Path:
        """Path to the cached processed JSON file."""
        return self.cache_dir / "processed.json"

    @abstractmethod
    def download(self) -> None:
        """Download the raw dataset from Hugging Face."""
        ...

    @abstractmethod
    def load(self, force_reload: bool = False) -> list[Any]:
        """Load and return all items as a list of typed items.

        Args:
            force_reload: If True, re-download and re-process even if cached.

        Returns:
            List of typed items (MMLUProItem, TAU2BenchItem, BFCLItem).
        """
        ...

    @abstractmethod
    def sample(self, n: int = 500, seed: int = 42) -> list[Any]:
        """Return a stratified random sample of ``n`` items.

        The sample indices should be saved to a JSON file so they remain
        fixed across all future runs.
        """
        ...

    def get_sample_path(self, n: int) -> Path:
        return self.cache_dir / f"sample_{n}_indices.json"

    def name(self) -> str:
        return self.dataset_name
