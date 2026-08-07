# -*- coding: utf-8 -*-
"""Evaluator orchestrator — runs test cases against local + cloud models,
writes JSONL results, and computes statistics.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .schema import (
    TestCaseEntry,
    ToolExpectEntry,
    RunRecord,
    StatsResult,
    CloudApiConfig,
    load_verifier,
)
from .metrics import compile_all_stats
from .runners import LocalModelRunner, CloudModelRunner
from .types import ChatMessage, ChatCompletionTool  # noqa: E402

logger = logging.getLogger("tools.evaluator")


class Evaluator:
    """Orchestrates running all test cases against local and cloud models."""

    def __init__(
        self,
        local_runner: Optional[LocalModelRunner] = None,
        cloud_runner: Optional[CloudModelRunner] = None,
        n_repeat: int = 3,
    ):
        self.local_runner = local_runner
        self.cloud_runner = cloud_runner
        self.n_repeat = n_repeat
        self._project_root = Path(__file__).resolve().parents[1]
        # Resolve model names once
        self._local_model_name = local_runner.model_name if local_runner else "local"
        self._cloud_model_name = cloud_runner.model_name if cloud_runner else "cloud"

    def evaluate(
        self,
        test_cases: list[TestCaseEntry],
        output_jsonl: Path,
    ) -> None:
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            total_runs = 0
            for entry_idx, entry in enumerate(test_cases):
                tools: Optional[list[ChatCompletionTool]] = None
                if entry.tools:
                    tools = [ChatCompletionTool(**t) for t in entry.tools]

                for conv_idx, (messages_raw, expect) in enumerate(
                    zip(entry.messages, entry.tool_expect)
                ):
                    messages = [ChatMessage(**m) for m in messages_raw]
                    verifier_fn = None
                    if expect.has_verifier and expect.verifier is not None:
                        verifier_fn = load_verifier(expect.verifier, self._project_root)

                    for repeat in range(self.n_repeat):
                        if self.local_runner is not None:
                            result = self.local_runner.run_single(
                                messages=messages, tools=tools,
                                tool_choice=entry.tool_choice,
                                temperature=entry.temperature,
                                top_p=entry.top_p,
                                max_tokens=entry.max_tokens,
                                stop=entry.stop,
                            )
                            rec = _make_record(entry_idx, conv_idx, repeat, self._local_model_name, expect, result, verifier_fn)
                            handle.write(rec.model_dump_json() + "\n")
                            handle.flush()
                            total_runs += 1

                        if self.cloud_runner is not None:
                            result = self.cloud_runner.run_single(
                                messages=messages, tools=tools,
                                tool_choice=entry.tool_choice,
                                temperature=entry.temperature,
                                top_p=entry.top_p,
                                max_tokens=entry.max_tokens,
                                stop=entry.stop,
                            )
                            rec = _make_record(entry_idx, conv_idx, repeat, self._cloud_model_name, expect, result, verifier_fn)
                            handle.write(rec.model_dump_json() + "\n")
                            handle.flush()
                            total_runs += 1

            logger.info("Total runs written: %d to %s", total_runs, output_jsonl)

    @staticmethod
    def compute_stats_from_jsonl(jsonl_path: Path) -> dict[str, StatsResult]:
        records: list[RunRecord] = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                records.append(RunRecord(**json.loads(line)))
        return compile_all_stats(records)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    entry_idx: int,
    conv_idx: int,
    repeat: int,
    model_type: str,
    expect: ToolExpectEntry,
    result: dict,
    verifier_fn,
) -> RunRecord:
    tool_calls = result.get("tool_calls") or []
    tool_names = [
        (tc.get("function") or {}).get("name", "unknown")
        for tc in tool_calls
    ]
    actual_has_tool_calls = bool(tool_calls)

    name_matched = False
    if expect.expects_tool_call and actual_has_tool_calls:
        name_matched = expect.name in tool_names

    verifier_passed: Optional[bool] = None
    verifier_reason: Optional[str] = None
    logger.debug('Input tool call: %s', json.dumps(tool_calls))
    if name_matched and expect.has_verifier and verifier_fn is not None:
        for tc in tool_calls:
            fn = tc.get("function") or {}
            if fn.get("name") == expect.name:
                args = fn.get("arguments", "{}")
                if isinstance(args, dict):
                    args = json.dumps(args, ensure_ascii=False)
                try:
                    v_result = verifier_fn(args)
                    if isinstance(v_result, tuple):
                        verifier_passed = bool(v_result[0])
                        verifier_reason = v_result[1] if len(v_result) > 1 and not v_result[0] else None
                    else:
                        # Backward-compat: single bool return
                        verifier_passed = bool(v_result)
                except Exception as exc:
                    logger.warning("Verifier raised exception: %s", exc)
                    verifier_passed = False
                    verifier_reason = f"{type(exc).__name__}: {exc}"
                break
    
    logger.info("Run record [%d/%d] details: tool=%s, actual=%s, name=%s, passed=%s, timecost=%.2fms",
                 entry_idx, conv_idx, expect.name, tool_names, name_matched, verifier_passed, result.get("latency_ms", 0.0))
    return RunRecord(
        entry_index=entry_idx,
        conv_index=conv_idx,
        repeat=repeat,
        model_type=model_type,
        expected_tool_name=expect.name,
        has_verifier=expect.has_verifier,
        actual_has_tool_calls=actual_has_tool_calls,
        tool_call_count=len(tool_calls),
        tool_names=tool_names,
        name_matched=name_matched,
        verifier_passed=verifier_passed,
        verifier_reason=verifier_reason,
        response_text=result.get("text"),
        response_tool_calls=tool_calls if tool_calls else None,
        finish_reason=result.get("finish_reason", "stop"),
        error=result.get("error"),
        latency_ms=result.get("latency_ms", 0.0),
        prefill_ms=result.get("prefill_ms"),
        decode_ms=result.get("decode_ms"),
        token_count=result.get("token_count", 0),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def create_local_runner_from_yaml(
    config_path: str,
    model_index: int = 0,
) -> LocalModelRunner:
    from .app import load_proxy_config  # noqa: E402
    _, model_configs = load_proxy_config(config_path)
    if not model_configs:
        raise ValueError(f"No models found in config: {config_path}")
    if model_index >= len(model_configs):
        raise IndexError(f"Model index {model_index} out of range (found {len(model_configs)} models)")
    chosen = model_configs[model_index]
    logger.info("Created local runner: model=%s backend=%s path=%s", chosen.name, chosen.backend, chosen.model_path)
    return LocalModelRunner(chosen)
