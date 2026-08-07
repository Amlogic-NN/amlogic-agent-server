# -*- coding: utf-8 -*-
"""Pure statistics functions for the tool-calling evaluation framework.

Supports three levels of evaluation:

1. **Presence** — did the model produce a tool call when expected? (TP/FP/FN/TN)
2. **Name** — when a tool call was expected, was the correct tool name used?
3. **Verifier** — when a verifier was provided, did the arguments pass?
"""

from __future__ import annotations

from typing import Iterable

from .schema import RunRecord, StatsResult


def compute_confusion_matrix(
    records: Iterable[RunRecord],
) -> tuple[int, int, int, int]:
    """Compute TP, FP, FN, TN from run records at the *presence* level.

    - **TP**: ``expected_tool_name is not None`` **and** ``actual_has_tool_calls``.
    - **FP**: ``expected_tool_name is None`` **but** ``actual_has_tool_calls``.
    - **FN**: ``expected_tool_name is not None`` **but** no tool call.
    - **TN**: ``expected_tool_name is None`` **and** no tool call.

    Returns:
        ``(TP, FP, FN, TN)``.
    """
    tp = fp = fn = tn = 0
    for rec in records:
        expects_call = rec.expected_tool_name is not None
        if expects_call:
            if rec.actual_has_tool_calls:
                tp += 1
            else:
                fn += 1
        else:
            if rec.actual_has_tool_calls:
                fp += 1
            else:
                tn += 1
    return tp, fp, fn, tn


def compute_recall(tp: int, fn: int) -> float:
    """Recall = TP / (TP + FN)."""
    denom = tp + fn
    return tp / denom if denom > 0 else 0.0


def compute_precision(tp: int, fp: int) -> float:
    """Precision = TP / (TP + FP)."""
    denom = tp + fp
    return tp / denom if denom > 0 else 0.0


def compute_accuracy(tp: int, tn: int, fp: int, fn: int) -> float:
    """Accuracy = (TP + TN) / (TP + TN + FP + FN)."""
    denom = tp + tn + fp + fn
    return (tp + tn) / denom if denom > 0 else 0.0


def compute_f1(precision: float, recall: float) -> float:
    """F1 = 2 * P * R / (P + R)."""
    denom = precision + recall
    return 2 * precision * recall / denom if denom > 0 else 0.0


def compile_stats(
    records: Iterable[RunRecord],
    model_type: str,
) -> StatsResult:
    """Aggregate run records for a single model type into a ``StatsResult``.

    Counts TP/FP/FN/TN at the presence level, plus name-match and verifier
    statistics.
    """
    grouped = [r for r in records if r.model_type == model_type]
    tp, fp, fn, tn = compute_confusion_matrix(grouped)

    # Name-level: of the TP runs, how many name-matched?
    tp_records = [r for r in grouped if r.expected_tool_name is not None and r.actual_has_tool_calls]
    name_match = sum(1 for r in tp_records if r.name_matched)
    name_mismatch = len(tp_records) - name_match

    # Verifier-level: of the name-matched runs that had a verifier, how many passed?
    verifier_records = [r for r in tp_records if r.name_matched and r.has_verifier]
    verifier_total = len(verifier_records)
    verifier_pass = sum(1 for r in verifier_records if r.verifier_passed is True)

    error_count = sum(1 for r in grouped if r.error)
    latencies = [r.latency_ms for r in grouped if r.latency_ms > 0]

    return StatsResult(
        model_type=model_type,
        total_runs=len(grouped),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        name_match_count=name_match,
        name_mismatch_count=name_mismatch,
        verifier_total=verifier_total,
        verifier_pass_count=verifier_pass,
        error_count=error_count,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
    )


def compile_all_stats(
    records: Iterable[RunRecord],
) -> dict[str, StatsResult]:
    """Group run records by model_type and compute stats for each."""
    model_types: set[str] = {r.model_type for r in records}
    return {mt: compile_stats(records, mt) for mt in model_types}
