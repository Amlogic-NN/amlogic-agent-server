# -*- coding: utf-8 -*-
"""Dataset loaders for benchmark datasets (MMLU-Pro, TAU2-Bench, BFCL)."""

from .base import BenchmarkDataset
from .mmlu_pro import MMLUProDataset
from .mmlu import MMLUDataset
from .tau2_bench import TAU2BenchDataset
from .bfcl import BFCLDataset
from .mmstar import MMStarDataset
from .ocrbench import OCRBenchDataset

__all__ = ["BenchmarkDataset", "MMLUProDataset", "MMLUDataset", "TAU2BenchDataset", "BFCLDataset", "MMStarDataset", "OCRBenchDataset"]


def get_dataset(name: str) -> BenchmarkDataset:
    """Factory: return the appropriate dataset loader by name."""
    name = name.lower().strip()
    if name in ("mmlu_pro", "mmlu-pro", "mmlupro"):
        return MMLUProDataset()
    if name in ("mmlu",):
        return MMLUDataset()
    if name in ("tau2_bench", "tau2-bench", "tau2bench"):
        return TAU2BenchDataset()
    if name in ("bfcl",):
        return BFCLDataset()
    if name in ("mmstar",):
        return MMStarDataset()
    if name in ("ocrbench", "ocr_bench"):
        return OCRBenchDataset()
    raise ValueError(f"Unknown dataset: {name}")
