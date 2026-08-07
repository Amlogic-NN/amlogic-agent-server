# -*- coding: utf-8 -*-
"""Benchmark runners — extend existing model runners with benchmark-specific features.

Key additions over tools/runners.py:
- Compute TTFT (prefill time) and TPS (decode throughput) for ADLA models
- Multi-turn support for TAU2-Bench trajectories
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

from ..types import ChatMessage, ChatCompletionTool  # noqa: E402
from ..runners import LocalModelRunner, CloudModelRunner  # noqa: E402
from ..evaluator import create_local_runner_from_yaml  # noqa: E402
from ..schema import CloudApiConfig  # noqa: E402
from .schema import BenchmarkRunRecord  # noqa: E402

logger = logging.getLogger("benchmark.runners")


class BenchmarkLocalRunner:
    """Wraps LocalModelRunner with benchmark-specific result fields (TTFT, TPS)."""

    def __init__(self, local_runner: LocalModelRunner):
        self._runner = local_runner

    @property
    def model_name(self) -> str:
        return self._runner.model_name

    def run_single(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[ChatCompletionTool]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, list[str]]] = None,
    ) -> dict:
        """Run one inference and compute TTFT/TPS from runtime timestamps."""
        result = self._runner.run_single(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )

        # Compute TTFT and TPS from prefill/decode timestamps
        ttft_ms = None
        tps = None
        prefill_ms = result.get("prefill_ms")
        decode_ms = result.get("decode_ms")
        token_count = result.get("token_count", 0)

        if prefill_ms is not None:
            ttft_ms = prefill_ms
        if decode_ms is not None and decode_ms > 0 and token_count > 0:
            tps = token_count / (decode_ms / 1000.0)  # tokens per second

        result["ttft_ms"] = ttft_ms
        result["tps"] = tps
        return result

    def run_multi_turn(
        self,
        turns: list[dict],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> list[dict]:
        """Run a multi-turn conversation (for TAU2-Bench).

        Each turn: send messages + tools, get tool call, feed tool response
        back as next message, repeat.

        Args:
            turns: List of turn dicts with {messages, tools, expected_tool_name, tool_response}.

        Returns:
            List of per-turn results.
        """
        results = []
        conversation_msgs: list[ChatMessage] = []

        for turn_idx, turn in enumerate(turns):
            turn_msgs_raw = turn.get("messages", [])
            turn_tools_raw = turn.get("tools")
            tool_response = turn.get("tool_response")

            # Build cumulative conversation messages
            if turn_idx == 0:
                conversation_msgs = [ChatMessage(**m) for m in turn_msgs_raw]
            else:
                # Append new user message
                for m in turn_msgs_raw:
                    if ChatMessage(**m) not in conversation_msgs:
                        conversation_msgs.append(ChatMessage(**m))

            # Convert tools
            tools = None
            if turn_tools_raw:
                if isinstance(turn_tools_raw[0], ChatCompletionTool):
                    tools = turn_tools_raw
                else:
                    tools = [ChatCompletionTool(**t) for t in turn_tools_raw]

            # Run inference for this turn
            result = self.run_single(
                messages=list(conversation_msgs),
                tools=tools,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            results.append(result)

            # Feed tool response back for next turn
            if tool_response and result.get("tool_calls"):
                tc = result["tool_calls"][0]
                tc_id = tc.get("id", f"call_{turn_idx}")
                conversation_msgs.append(ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tc],
                ))
                conversation_msgs.append(ChatMessage(
                    role="tool",
                    tool_call_id=tc_id,
                    content=str(tool_response.get("content", "")),
                ))

        return results


class BenchmarkCloudRunner:
    """Wraps CloudModelRunner with benchmark-compatible interface."""

    def __init__(self, cloud_runner: CloudModelRunner):
        self._runner = cloud_runner

    @property
    def model_name(self) -> str:
        return self._runner.model_name

    def run_single(
        self,
        messages: list[ChatMessage],
        tools: Optional[list[ChatCompletionTool]] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[Union[str, list[str]]] = None,
    ) -> dict:
        """Run one cloud inference with API timing and token usage."""
        result = self._runner.run_single(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )
        result["ttft_ms"] = result.get("prefill_ms")  # TTFT = prefill time from API
        result["tps"] = None
        return result

    def run_multi_turn(
        self,
        turns: list[dict],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> list[dict]:
        """Run multi-turn conversation via cloud API."""
        results = []
        conversation_msgs: list[ChatMessage] = []

        for turn_idx, turn in enumerate(turns):
            turn_msgs_raw = turn.get("messages", [])
            turn_tools_raw = turn.get("tools")
            tool_response = turn.get("tool_response")

            if turn_idx == 0:
                conversation_msgs = [ChatMessage(**m) for m in turn_msgs_raw]
            else:
                for m in turn_msgs_raw:
                    if ChatMessage(**m) not in conversation_msgs:
                        conversation_msgs.append(ChatMessage(**m))

            tools = None
            if turn_tools_raw:
                if isinstance(turn_tools_raw[0], ChatCompletionTool):
                    tools = turn_tools_raw
                else:
                    tools = [ChatCompletionTool(**t) for t in turn_tools_raw]

            result = self.run_single(
                messages=list(conversation_msgs),
                tools=tools,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            results.append(result)

            if tool_response and result.get("tool_calls"):
                tc = result["tool_calls"][0]
                tc_id = tc.get("id", f"call_{turn_idx}")
                conversation_msgs.append(ChatMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tc],
                ))
                conversation_msgs.append(ChatMessage(
                    role="tool",
                    tool_call_id=tc_id,
                    content=str(tool_response.get("content", "")),
                ))

        return results


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def create_benchmark_local_runner(
    config_path: str,
    model_index: int = 0,
) -> BenchmarkLocalRunner:
    """Create a BenchmarkLocalRunner from a server YAML config."""
    local_runner = create_local_runner_from_yaml(config_path, model_index)
    return BenchmarkLocalRunner(local_runner)


def create_benchmark_cloud_runner(
    cloud_config: CloudApiConfig,
) -> BenchmarkCloudRunner:
    """Create a BenchmarkCloudRunner from a CloudApiConfig."""
    cloud_runner = CloudModelRunner(cloud_config)
    return BenchmarkCloudRunner(cloud_runner)
