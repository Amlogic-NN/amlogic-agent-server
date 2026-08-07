# -*- coding: utf-8 -*-
"""Benchmark Scorer — offline scoring of collected benchmark JSONL results.

Runs separately from data collection. Reads JSONL produced by BenchmarkEvaluator,
scores each entry, and outputs scored JSONL + summary BenchmarkScoreReport.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from .schema import BenchmarkRunRecord, BenchmarkScoreReport, DatasetScore

logger = logging.getLogger("benchmark.scorer")


class BenchmarkScorer:
    """Scores benchmark JSONL results offline.

    Scoring logic per dataset:
    - MMLU-Pro: Extract answer letter from response_text, compare to ground_truth
    - TAU2-Bench: Compare tool call name + arguments to expected
    - BFCL: Compare function name + arguments (AST-level) to expected
    """

    def score_jsonl(self, jsonl_path: Path, output_path: Optional[Path] = None) -> BenchmarkScoreReport:
        """Read JSONL, score each record, write scored JSONL, return report.

        Args:
            jsonl_path: Path to collected (unscored) JSONL.
            output_path: Path for scored JSONL. If None, uses jsonl_path with '_scored' suffix.

        Returns:
            BenchmarkScoreReport with per-dataset/per-model breakdowns.
        """
        if output_path is None:
            output_path = jsonl_path.parent / f"{jsonl_path.stem}_scored.jsonl"

        records = self._load_records(jsonl_path)
        if not records:
            logger.warning("No records found in %s", jsonl_path)
            return BenchmarkScoreReport()

        # Score each record
        for rec in records:
            self._score_one(rec)

        # Write scored JSONL
        self._write_scored(records, output_path)
        logger.info("Scored JSONL written to %s (%d records)", output_path, len(records))

        # Build report
        report = self._build_report(records)
        return report

    # ------------------------------------------------------------------
    # Per-record scoring
    # ------------------------------------------------------------------

    def _score_one(self, rec: BenchmarkRunRecord) -> None:
        """Score a single BenchmarkRunRecord in-place, setting is_correct."""
        dataset = rec.dataset.lower().strip()

        if dataset in ("mmlu", "mmlu_pro", "mmlu-pro", "mmlupro", "mmstar"):
            self._score_mmlu_pro(rec)
        elif dataset in ("ocrbench", "ocr_bench"):
            self._score_ocrbench(rec)
        elif dataset in ("tau2_bench", "tau2-bench", "tau2bench"):
            self._score_tau2_bench(rec)
        elif dataset in ("bfcl",):
            self._score_bfcl(rec)
        else:
            logger.warning("Unknown dataset '%s' for item %s, skipping scoring", dataset, rec.item_id)
            rec.is_correct = None

    # ---- MMLU-Pro: text-based MC ----

    def _score_mmlu_pro(self, rec: BenchmarkRunRecord) -> None:
        """Extract answer letter from response_text, compare to ground_truth."""
        ground_truth = (rec.score_detail or {}).get("ground_truth")
        response = rec.response_text or ""

        if ground_truth is None:
            rec.is_correct = None
            return

        # Try multiple extraction strategies
        extracted = self._extract_mmlu_answer(response)

        rec.is_correct = (extracted == ground_truth.upper())
        if rec.score_detail is None:
            rec.score_detail = {}
        rec.score_detail["extracted_answer"] = extracted
        rec.score_detail["ground_truth"] = ground_truth

    def _extract_mmlu_answer(self, text: str) -> Optional[str]:
        """Extract the answer letter from an MMLU-Pro response.

        Strategies (in priority order):
        1. Explicit answer markers at the END of response:
           "Answer: X", "answer: X", "answer is X", "correct answer: **X**"
        2. LaTeX boxed: \\boxed{X}
        3. Bold letter at end: **X** (last occurrence)
        4. First leading letter on its own line near the end:
           "D. Customer participation..." (the original question format echoed back)
        5. Last standalone A-J letter as fallback
        """
        if not text:
            return None

        text = text.strip()

        # Strategy 1: Look for explicit answer markers — prefer matches near the end.
        # We search from end to beginning to find the LAST explicit answer marker.
        answer_patterns = [
            # "Answer: X" / "Answer: **X**" / "answer: X"
            r'(?im)(?:correct\s+)?answer\s*(?:is|:)?\s*\*{0,2}\s*([A-Ja-j])\b\s*\*{0,2}',
            # "✅ Answer: **X**" / "✅ Final answer: **X**"
            r'(?im)(?:final\s+)?answer\s*:\s*\*{0,2}\s*([A-Ja-j])\b\s*\*{0,2}',
            # "option **X**" / "select **X**" / "choose **X**"
            r'(?im)(?:option|select|choose|correct\s+option)\s*(?:is|:)?\s*\*{0,2}\s*([A-Ja-j])\b\s*\*{0,2}',
        ]
        for pat in answer_patterns:
            matches = list(re.finditer(pat, text))
            if matches:
                # Use the LAST match (closest to end of response)
                return matches[-1].group(1).upper()

        # Strategy 2: LaTeX boxed answer — \\boxed{X}
        boxed = re.findall(r'\\boxed\{([A-Ja-j])\}', text)
        if boxed:
            return boxed[-1].upper()

        # Strategy 3: Bold single letter at end of line: **X**
        bold_matches = list(re.finditer(r'\*\*([A-Ja-j])\*\*', text))
        if bold_matches:
            return bold_matches[-1].group(1).upper()

        # Strategy 4: Leading letter like "D. Customer participation..."
        # near the very end of the response
        last_200 = text[-200:] if len(text) > 200 else text
        leading = re.findall(r'(?:^|\n)\s*([A-Ja-j])\.\s', last_200, re.MULTILINE)
        if leading:
            return leading[-1].upper()

        # Strategy 5: Entire response is just a single letter or "X."
        m = re.match(r'^\s*([A-Ja-j])\s*[\.\)]?\s*$', text)
        if m:
            return m.group(1).upper()

        # Strategy 6: Last standalone A-J letter as last resort
        letters = re.findall(r'\b([A-Ja-j])\b', text)
        if letters:
            return letters[-1].upper()

        return None

    # ---- OCRBench: free-form OCR text matching ----

    # Question types that use HMER (no lowercase) normalization
    HMER_TYPES = {"Handwritten Mathematical Expression Recognition", "HMER"}

    def _score_ocrbench(self, rec: BenchmarkRunRecord) -> None:
        """Score OCRBench responses using the original evaluation logic.

        Original algorithm (from Yuliang-Liu/MultimodalOCR/OCRBench/example.py):
        - Non-HMER: answer.lower().strip().replace("\\n", " ") in predict.lower().strip().replace("\\n", " ")
        - HMER:      answer.strip().replace("\\n", " ").replace(" ", "") in predict.strip().replace("\\n", " ").replace(" ", "")

        Key difference from typical accuracy: uses substring match (answer *in* predict),
        not exact equality. HMER questions additionally remove all spaces and skip lowercasing.
        """
        ground_truth = (rec.score_detail or {}).get("ground_truth")
        response = rec.response_text or ""
        subject = (rec.score_detail or {}).get("subject", "")
        is_hmer = any(t in subject for t in self.HMER_TYPES)

        if ground_truth is None:
            rec.is_correct = None
            return

        if isinstance(ground_truth, list):
            gt_list = [str(a) for a in ground_truth]
        else:
            gt_list = [str(ground_truth)]

        if is_hmer:
            # HMER: no lowercase, strip + \n→space + remove all spaces, then substring check
            norm_predict = response.strip().replace("\n", " ").replace(" ", "")
            is_correct = any(
                a.strip().replace("\n", " ").replace(" ", "") in norm_predict
                for a in gt_list
            )
            extracted = norm_predict
        else:
            # Non-HMER: lowercase + strip + \n→space, then substring check
            norm_predict = response.lower().strip().replace("\n", " ")
            is_correct = any(
                a.lower().strip().replace("\n", " ") in norm_predict
                for a in gt_list
            )
            extracted = norm_predict

        rec.is_correct = is_correct
        if rec.score_detail is None:
            rec.score_detail = {}
        rec.score_detail["extracted_answer"] = extracted
        rec.score_detail["ground_truth"] = ground_truth
        rec.score_detail["is_hmer"] = is_hmer

    # ---- TAU2-Bench: multi-turn tool use ----

    def _score_tau2_bench(self, rec: BenchmarkRunRecord) -> None:
        """Score TAU2-Bench tool calls against expected actions.

        IMPORTANT: Only tasks with ACTION in reward_basis should be scored
        as correct/incorrect. In the original TAU2-Bench, the actions list
        is primarily a reference trajectory for DB state derivation. Action
        comparison is only meaningful when `reward_basis` includes ACTION.

        For tasks without ACTION in reward_basis, we still record tool call
        data but mark is_correct = None (not applicable).
        """
        detail = rec.score_detail or {}
        reward_basis = detail.get("reward_basis", [])
        all_expected = detail.get("all_expected_actions", [])
        expected_name = rec.expected_tool_name

        if rec.score_detail is None:
            rec.score_detail = {}
        rec.score_detail["actual_calls_detail"] = []

        # No actions expected in this turn
        if not expected_name and not all_expected:
            rec.is_correct = not rec.actual_has_tool_calls
            return

        # Check if ACTION is in reward_basis — only then is this a meaningful score
        action_is_required = "ACTION" in reward_basis
        if not action_is_required and not all_expected:
            rec.is_correct = None  # Not applicable for action evaluation
            return

        # Use all_expected_actions if available, otherwise fallback to single expected
        if all_expected:
            expected_list = all_expected
        elif expected_name:
            expected_args = detail.get("expected_tool_arguments")
            expected_list = [{"name": expected_name, "arguments": expected_args}]
        else:
            rec.is_correct = True
            return

        # Collect actual tool calls
        actual_calls = rec.response_tool_calls or []
        parsed_actual: list[dict] = []
        for tc in actual_calls:
            fn = tc.get("function") or {}
            act_name = fn.get("name", "")
            act_args = fn.get("arguments", {})
            if isinstance(act_args, str):
                try:
                    act_args = json.loads(act_args)
                except json.JSONDecodeError:
                    act_args = {}
            parsed_actual.append({"name": act_name, "arguments": act_args})

        # Match each expected action to an actual action (order-independent)
        matched = 0
        used_actual = set()
        for exp in expected_list:
            exp_name = exp.get("name", "")
            exp_args = exp.get("arguments")
            found = False
            for ai, act in enumerate(parsed_actual):
                if ai in used_actual:
                    continue
                if act["name"] != exp_name:
                    continue
                # Check arguments if expected
                if exp_args is not None:
                    if self._compare_args(exp_args, act["arguments"]):
                        found = True
                        used_actual.add(ai)
                        rec.score_detail["actual_calls_detail"].append({
                            "match": True, "expected": exp, "actual": act,
                        })
                        break
                else:
                    found = True
                    used_actual.add(ai)
                    rec.score_detail["actual_calls_detail"].append({
                        "match": True, "expected": exp, "actual": act,
                    })
                    break

            if not found:
                rec.score_detail["actual_calls_detail"].append({
                    "match": False, "expected": exp,
                    "actual_names": [a["name"] for a in parsed_actual],
                })

            if found:
                matched += 1

        rec.is_correct = (matched == len(expected_list))

    # ---- BFCL: function calling ----

    def _score_bfcl(self, rec: BenchmarkRunRecord) -> None:
        """Compare tool calls to expected for BFCL.

        Category-specific scoring (matching original BFCL v4 evaluation):
        - simple/parallel/multiple/parallel_multiple: exact function name +
          argument matching via ast_checker (order-independent for parallel).
        - irrelevance: model must NOT output any function call.
          Even an empty list or non-call output is acceptable.
        - relevance: model MUST output at least one function call.
          The specific function doesn't matter — only that a call was attempted.
        """
        detail = rec.score_detail or {}
        expected_calls = detail.get("expected_tool_calls", [])
        category = detail.get("category", "")

        # ---- irrelevance / relevance — check only whether a tool was called ----
        if category in ("irrelevance", "relevance"):
            if category == "irrelevance":
                # Original BFCL: success = NOT contain_func_call
                rec.is_correct = not rec.actual_has_tool_calls
            else:
                # Original BFCL: success = contain_func_call
                # (any function call counts, even if it fails to match expected)
                rec.is_correct = rec.actual_has_tool_calls
            return

        # ---- Standard scoring for simple/parallel/multiple/parallel_multiple ----
        if not expected_calls:
            # No expected calls (shouldn't happen for standard categories)
            rec.is_correct = False if rec.actual_has_tool_calls else True
            return

        actual_calls = rec.response_tool_calls or []

        # Must have same number of calls
        if len(actual_calls) != len(expected_calls):
            rec.is_correct = False
            return

        # Compare each call (order-independent for parallel calls)
        matched = 0
        used_actual = set()
        for expected in expected_calls:
            exp_name = expected.get("name", "")
            exp_args = expected.get("arguments", {})
            for ai, actual in enumerate(actual_calls):
                if ai in used_actual:
                    continue
                fn = actual.get("function") or {}
                act_name = fn.get("name", "")
                if act_name != exp_name:
                    continue
                act_args = fn.get("arguments", {})
                if isinstance(act_args, str):
                    try:
                        act_args = json.loads(act_args)
                    except json.JSONDecodeError:
                        act_args = {}
                if self._compare_args(exp_args, act_args):
                    matched += 1
                    used_actual.add(ai)
                    break

        rec.is_correct = (matched == len(expected_calls))

    # ------------------------------------------------------------------
    # Argument comparison
    # ------------------------------------------------------------------

    def _compare_args(self, expected: dict, actual: dict) -> bool:
        """Dict comparison for tool call arguments.

        Handles nested dicts and lists. Normalizes string/number types.
        List-valued expected args are treated as "any of these values is OK"
        (BFCL possible_answer format with multiple valid values).
        """
        if not isinstance(expected, dict) or not isinstance(actual, dict):
            return expected == actual

        exp_keys = set(expected.keys())
        act_keys = set(actual.keys())

        # Allow extra keys in actual (model added optional params)
        if not exp_keys.issubset(act_keys):
            return False

        for key in exp_keys:
            ev = expected[key]
            av = actual[key]

            # Any-of matching: expected is a list of valid values
            if isinstance(ev, list) and not (isinstance(av, list) and all(isinstance(x, dict) for x in ev)):
                # Check if actual value matches any of the expected values
                matched = False
                for candidate in ev:
                    if self._scalar_equal(candidate, av):
                        matched = True
                        break
                if not matched:
                    return False
                continue

            # Type normalization
            if isinstance(ev, str) and isinstance(av, (int, float)):
                av = str(av)
            elif isinstance(ev, (int, float)) and isinstance(av, str):
                try:
                    av = float(av) if isinstance(ev, float) else int(av)
                except (ValueError, TypeError):
                    pass

            if isinstance(ev, dict) and isinstance(av, dict):
                if not self._compare_args(ev, av):
                    return False
            elif isinstance(ev, list) and isinstance(av, list):
                # Both are lists — exact match (for nested arrays like item_ids)
                if len(ev) != len(av):
                    return False
                if not all(self._compare_args(e, a) if isinstance(e, dict) else self._scalar_equal(e, a) for e, a in zip(ev, av)):
                    return False
            elif not self._scalar_equal(ev, av):
                return False

        return True

    @staticmethod
    def _scalar_equal(expected, actual) -> bool:
        """Compare two scalar values (str/int/float/bool/None) with normalization."""
        if expected is None and actual is None:
            return True
        if expected is None or actual is None:
            return False
        # Normalize to string for comparison, but preserve numeric equality
        if isinstance(expected, bool) and isinstance(actual, bool):
            return expected == actual
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(expected - actual) < 1e-9
        if isinstance(expected, (int, float)) and isinstance(actual, str):
            try:
                return abs(expected - float(actual)) < 1e-9
            except (ValueError, TypeError):
                return str(expected).strip().lower() == actual.strip().lower()
        if isinstance(expected, str) and isinstance(actual, (int, float)):
            try:
                return abs(float(expected) - actual) < 1e-9
            except (ValueError, TypeError):
                return expected.strip().lower() == str(actual).strip().lower()
        return str(expected).strip().lower() == str(actual).strip().lower()

    # ------------------------------------------------------------------
    # Report building
    # ------------------------------------------------------------------

    def _build_report(self, records: list[BenchmarkRunRecord]) -> BenchmarkScoreReport:
        """Aggregate scored records into BenchmarkScoreReport.

        For multi-turn datasets (TAU2-Bench), records are aggregated by item_id:
        all turns must be correct for the task to be counted as correct.
        For single-turn datasets (MMLU-Pro, BFCL), each record is one item.
        """
        from collections import defaultdict
        from datetime import datetime, timezone

        # Group by (dataset, model_type)
        groups = defaultdict(list)
        for rec in records:
            groups[(rec.dataset, rec.model_type)].append(rec)

        scores: list[DatasetScore] = []
        for (dataset, model_type), recs in sorted(groups.items()):
            is_multi_turn = dataset.lower().strip() in ("tau2_bench", "tau2-bench", "tau2bench")

            if is_multi_turn:
                # TAU2-Bench: aggregate by item_id.
                # Only tasks with ACTION in reward_basis are scored for accuracy.
                # Other tasks are tracked but not counted in correct/incorrect.
                task_groups = defaultdict(list)
                for r in recs:
                    task_groups[r.item_id].append(r)

                correct = 0
                incorrect = 0
                errors = 0
                na_count = 0  # Not applicable (no ACTION in reward_basis)
                latencies: list[float] = []
                ttfts: list[float] = []
                tps_vals: list[float] = []

                breakdown = {}

                for tid, task_recs in task_groups.items():
                    domain = "unknown"
                    has_action_basis = False
                    for r in task_recs:
                        d = (r.score_detail or {}).get("domain")
                        if d:
                            domain = d
                        rb = (r.score_detail or {}).get("reward_basis", [])
                        if "ACTION" in rb:
                            has_action_basis = True

                    if not has_action_basis:
                        na_count += 1
                    else:
                        breakdown.setdefault(domain, {"total": 0, "correct": 0})
                        breakdown[domain]["total"] += 1

                        all_correct = all(r.is_correct is True for r in task_recs)
                        any_error = any(r.error for r in task_recs)

                        if any_error:
                            errors += 1
                        elif all_correct:
                            correct += 1
                            breakdown[domain]["correct"] += 1
                        else:
                            incorrect += 1

                    for r in task_recs:
                        if r.latency_ms > 0:
                            latencies.append(r.latency_ms)
                        if r.ttft_ms is not None and r.ttft_ms > 0:
                            ttfts.append(r.ttft_ms)
                        if r.tps is not None and r.tps > 0:
                            tps_vals.append(r.tps)

                total = len(task_groups)
            else:
                correct = sum(1 for r in recs if r.is_correct is True)
                incorrect = sum(1 for r in recs if r.is_correct is False)
                errors = sum(1 for r in recs if r.error)
                latencies = [r.latency_ms for r in recs if r.latency_ms > 0]
                ttfts = [r.ttft_ms for r in recs if r.ttft_ms is not None and r.ttft_ms > 0]
                tps_vals = [r.tps for r in recs if r.tps is not None and r.tps > 0]
                total = len(recs)

                # Per-category/subject/domain breakdown
                breakdown = {}
                for r in recs:
                    detail = r.score_detail or {}
                    group_key = (
                        detail.get("subject")
                        or detail.get("domain")
                        or detail.get("category")
                        or "unknown"
                    )
                    breakdown.setdefault(group_key, {"total": 0, "correct": 0})
                    breakdown[group_key]["total"] += 1
                    if r.is_correct is True:
                        breakdown[group_key]["correct"] += 1

            for key, bd in breakdown.items():
                bd["accuracy"] = bd["correct"] / bd["total"] if bd["total"] > 0 else 0.0

            scores.append(DatasetScore(
                dataset=dataset,
                model_type=model_type,
                total_items=total,
                correct_items=correct,
                accuracy=correct / total if total > 0 else 0.0,
                error_count=errors,
                avg_latency_ms=sum(latencies) / len(latencies) if latencies else 0.0,
                avg_ttft_ms=sum(ttfts) / len(ttfts) if ttfts else None,
                avg_tps=sum(tps_vals) / len(tps_vals) if tps_vals else None,
                breakdown=dict(breakdown),
            ))

        return BenchmarkScoreReport(
            scores=scores,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_records(self, jsonl_path: Path) -> list[BenchmarkRunRecord]:
        records = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(BenchmarkRunRecord(**json.loads(line)))
                except Exception as exc:
                    logger.warning("Failed to parse record: %s", exc)
        return records

    def _write_scored(self, records: list[BenchmarkRunRecord], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")
