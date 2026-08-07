# -*- coding: utf-8 -*-
"""Benchmark Evaluator — orchestrates data collection (inference) phase.

Runs benchmark items against local/cloud models, writes JSONL results.
Supports MMLU-Pro (text MC), TAU2-Bench (multi-turn tool use), BFCL (function calling).
Includes progress reporting, resume support, and direct API fast path for cloud models.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


from ..types import ChatMessage, ChatCompletionTool  # noqa: E402
from ..schema import CloudApiConfig  # noqa: E402
from .schema import BenchmarkRunRecord  # noqa: E402
from .runners import BenchmarkLocalRunner, BenchmarkCloudRunner, create_benchmark_cloud_runner  # noqa: E402
from .converters import convert_item  # noqa: E402

logger = logging.getLogger("benchmark.evaluator")


class BenchmarkEvaluator:
    """Runs benchmark items through models and writes JSONL results.

    Supports:
    - Text-based MC (MMLU-Pro): plain chat, no tools, records response_text
    - Multi-turn tool use (TAU2-Bench): records all-turn tool calls
    - Function calling (BFCL): records tool call accuracy
    - Resume: skip already-completed (dataset, item_id, repeat) tuples
    - Progress: periodic status output every N items
    """

    DEFAULT_SYSTEM = (
        "You are a helpful assistant with access to functions. "
        "Use the provided functions to answer the user's query."
    )

    def __init__(
        self,
        local_runner: Optional[BenchmarkLocalRunner] = None,
        cloud_runner: Optional[BenchmarkCloudRunner] = None,
        n_repeat: int = 1,
        save_responses: bool = True,
        progress_interval: int = 20,
        max_tokens: Optional[int] = None,
    ):
        self.local_runner = local_runner
        self.cloud_runner = cloud_runner
        self.n_repeat = n_repeat
        self.save_responses = save_responses
        self.progress_interval = progress_interval
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        dataset_name: str,
        items: list[dict],
        output_jsonl: Path,
        is_summary: bool = False,
    ) -> int:
        """Run evaluation on all items and write JSONL.

        Returns:
            Total number of runs written.
        """
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        completed = self._load_completed(output_jsonl)

        total_runs = 0
        errors = 0
        t0 = time.time()
        num_runners = max(1, int(self.local_runner is not None) + int(self.cloud_runner is not None))
        target = self._count_expected_runs(dataset_name, items) * num_runners * self.n_repeat

        with output_jsonl.open("a" if completed else "w", encoding="utf-8") as handle:
            for entry_idx, raw_item in enumerate(items):
                converted = convert_item(dataset_name, raw_item)

                for repeat in range(self.n_repeat):
                    if (dataset_name, converted["item_id"], repeat) in completed:
                        continue

                    if self.cloud_runner is not None:
                        recs = self._run_one_item(
                            dataset_name, converted, entry_idx, repeat,
                            self.cloud_runner.model_name,
                            runner=self.cloud_runner,
                            is_summary=is_summary,
                        )
                        for rec in recs:
                            handle.write(rec.model_dump_json() + "\n")
                            handle.flush()
                            total_runs += 1
                            if rec.error:
                                errors += 1

                    if self.local_runner is not None:
                        recs = self._run_one_item(
                            dataset_name, converted, entry_idx, repeat,
                            self.local_runner.model_name,
                            runner=self.local_runner,
                            is_summary=is_summary,
                        )
                        for rec in recs:
                            handle.write(rec.model_dump_json() + "\n")
                            handle.flush()
                            total_runs += 1
                            if rec.error:
                                errors += 1

                    if total_runs > 0 and total_runs % self.progress_interval == 0:
                        elapsed = time.time() - t0
                        rate = total_runs / elapsed if elapsed > 0 else 0
                        eta = (target - total_runs) / rate if rate > 0 else 0
                        print(f"  [{dataset_name}] {total_runs}/{target} ({rate:.1f}/s) ETA {eta:.0f}s errors={errors}")

        elapsed = time.time() - t0
        print(f"\nDONE [{dataset_name}]: {total_runs} runs, {errors} errors, {elapsed:.1f}s total")
        return total_runs

    # ------------------------------------------------------------------
    # Expected run counting
    # ------------------------------------------------------------------

    def _count_expected_runs(self, dataset_name: str, items: list[dict]) -> int:
        """Count the expected number of BenchmarkRunRecord per item.

        MMLU-Pro and BFCL produce 1 record per item.
        TAU2-Bench produces 1 record per turn (multi-turn trajectories).
        """
        dn = dataset_name.lower().strip()
        if dn in ("tau2_bench", "tau2-bench", "tau2bench"):
            total = 0
            for item in items:
                turns = item.get("turns", [])
                total += len(turns) if turns else 1
            return total
        # mmstar, mmlu, mmlu_pro, bfcl: 1 record per item
        return len(items)

    # ------------------------------------------------------------------
    # Per-item dispatch
    # ------------------------------------------------------------------

    def _run_one_item(self, dataset_name, converted, entry_idx, repeat,
                      model_type, runner, is_summary):
        item_id = converted["item_id"]

        if dataset_name in ("mmlu", "mmlu_pro", "mmlu-pro", "mmlupro", "mmstar", "ocrbench"):
            return self._run_mmlu_pro(dataset_name, converted, entry_idx, repeat, model_type, item_id, runner, is_summary)
        if dataset_name in ("tau2_bench", "tau2-bench", "tau2bench"):
            return self._run_tau2_bench(converted, entry_idx, repeat, model_type, item_id, runner, is_summary)
        if dataset_name in ("bfcl",):
            return self._run_bfcl(converted, entry_idx, repeat, model_type, item_id, runner, is_summary)

        logger.warning("Unknown dataset: %s", dataset_name)
        return []

    # ---- MMLU-Pro: text-based MC ----

    def _run_mmlu_pro(self, dataset_name, converted, entry_idx, repeat, model_type, item_id, runner, is_summary):
        messages = [ChatMessage(**m) for m in converted["messages"]]
        result = runner.run_single(messages=messages, max_tokens=self.max_tokens)
        rec = self._make_record(dataset_name, item_id, entry_idx, 0, repeat, model_type,
                                None, False, result, is_summary,
                                {"ground_truth": converted.get("ground_truth"),
                                 "subject": converted.get("subject"),
                                 "options": converted.get("options")})
        return [rec]

    # ---- TAU2-Bench: multi-turn action sequence ----

    def _run_tau2_bench(self, converted, entry_idx, repeat, model_type, item_id, runner, is_summary):
        turns = converted.get("turns", [])
        if not turns:
            return []

        records = []

        for ti, turn in enumerate(turns):
            turn_idx = turn.get("turn_index", ti)
            turn_msgs_raw = turn.get("messages", [])
            turn_tools_raw = turn.get("tools")

            # Build messages from scratch each turn using the pre-built messages
            # (which already include conversation history from dataset preparation)
            messages = [ChatMessage(**m) for m in turn_msgs_raw]

            # Convert tools
            tools = None
            if turn_tools_raw:
                tools = [ChatCompletionTool(**t) for t in turn_tools_raw]
                if not any(m.role == "system" for m in messages):
                    messages.insert(0, ChatMessage(role="system", content=self.DEFAULT_SYSTEM))

            # Run inference for this turn
            result = runner.run_single(messages=messages, tools=tools, max_tokens=self.max_tokens)

            # Build score_detail with all expected actions for this turn
            all_expected = turn.get("all_expected_actions", [])
            expected_name = turn.get("expected_tool_name", "")
            expected_args = turn.get("expected_tool_arguments")

            score_detail = {
                "domain": converted.get("domain"),
                "turn_index": turn_idx,
                "total_turns": len(turns),
                "expected_tool_name": expected_name,
                "expected_tool_arguments": expected_args,
                "all_expected_actions": all_expected,
                "reward_basis": converted.get("reward_basis", []),
            }

            rec = self._make_record("tau2_bench", item_id, entry_idx, turn_idx, repeat,
                                    model_type, expected_name, False, result, is_summary,
                                    score_detail)
            records.append(rec)

        return records

    # ---- BFCL: function calling ----

    def _run_bfcl(self, converted, entry_idx, repeat, model_type, item_id, runner, is_summary):
        msgs_raw = list(converted["messages"])
        tools_raw = converted.get("tools", [])
        tools = None

        if tools_raw:
            tools = [ChatCompletionTool(**t) for t in tools_raw]
            # System messages are already set during dataset conversion;
            # only add fallback if somehow missing.
            if not any(m.get("role") == "system" for m in msgs_raw):
                category = converted.get("category", "")
                if category == "irrelevance":
                    sys_content = (
                        "You are a helpful assistant. "
                        "You may be given access to functions, but only use them "
                        "if they are actually needed to answer the user's query."
                    )
                else:
                    sys_content = self.DEFAULT_SYSTEM
                msgs_raw.insert(0, {"role": "system", "content": sys_content})

        result = runner.run_single(messages=[ChatMessage(**m) for m in msgs_raw], tools=tools, max_tokens=self.max_tokens)

        expected_name = None
        ec = converted.get("expected_tool_calls", [])
        if ec:
            expected_name = ec[0].get("name")

        rec = self._make_record("bfcl", item_id, entry_idx, 0, repeat, model_type,
                                expected_name, False, result, is_summary,
                                {"category": converted.get("category"),
                                 "expected_tool_calls": ec})
        return [rec]

    # ------------------------------------------------------------------
    # Record factory
    # ------------------------------------------------------------------

    def _make_record(self, dataset, item_id, entry_idx, conv_idx, repeat,
                     model_type, expected_tool_name, has_verifier, result,
                     is_summary, score_detail) -> BenchmarkRunRecord:
        tool_calls = result.get("tool_calls") or []
        tool_names = [(tc.get("function") or {}).get("name", "unknown") for tc in tool_calls]
        actual_has_tool = bool(tool_calls)
        name_matched = bool(expected_tool_name and expected_tool_name in tool_names)
        resp_text = result.get("text") if self.save_responses else None
        resp_tc = tool_calls if tool_calls and self.save_responses else None

        return BenchmarkRunRecord(
            entry_index=entry_idx, conv_index=conv_idx, repeat=repeat,
            model_type=model_type, dataset=dataset, item_id=item_id,
            is_summary=is_summary,
            expected_tool_name=expected_tool_name, has_verifier=has_verifier,
            actual_has_tool_calls=actual_has_tool, tool_call_count=len(tool_calls),
            tool_names=tool_names, name_matched=name_matched,
            verifier_passed=None, verifier_reason=None,
            response_text=resp_text, response_tool_calls=resp_tc,
            finish_reason=result.get("finish_reason", "stop"),
            error=result.get("error"),
            latency_ms=result.get("latency_ms", 0.0),
            prefill_ms=result.get("prefill_ms"),
            decode_ms=result.get("decode_ms"),
            ttft_ms=result.get("ttft_ms"),
            tps=result.get("tps"),
            total_tokens=result.get("total_tokens", 0),
            completion_tokens=result.get("completion_tokens", 0),
            token_count=result.get("token_count", 0),
            score_detail=score_detail,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def _load_completed(self, jsonl_path: Path) -> set:
        completed: set = set()
        if not jsonl_path.exists():
            return completed
        try:
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        completed.add((rec.get("dataset", ""), rec.get("item_id", ""), rec.get("repeat", 0)))
                    except json.JSONDecodeError:
                        continue
        except Exception as exc:
            logger.warning("Could not read existing JSONL for resume: %s", exc)
        return completed


# ---------------------------------------------------------------------------
# Convenience API — creates runners from config and runs evaluation
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_name: str,
    items: list[dict],
    output_jsonl: Path,
    cloud_config_path: str = "test_data/test_config.yaml",
    skip_cloud: bool = False,
    is_summary: bool = False,
    save_responses: bool = True,
    max_tokens: Optional[int] = None,
) -> int:
    """One-shot: create cloud runner from config, run evaluation, return run count."""
    cloud_runner = None
    if not skip_cloud:
        cloud_cfg = CloudApiConfig.from_yaml(cloud_config_path)
        cloud_runner = create_benchmark_cloud_runner(cloud_cfg)
        logger.info("Cloud runner: %s @ %s", cloud_runner.model_name, cloud_cfg.base_url)

    evaluator = BenchmarkEvaluator(cloud_runner=cloud_runner, n_repeat=1, save_responses=save_responses, max_tokens=max_tokens)
    return evaluator.evaluate(dataset_name, items, output_jsonl, is_summary=is_summary)
