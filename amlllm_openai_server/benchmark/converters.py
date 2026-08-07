# -*- coding: utf-8 -*-
"""Converters — transform raw dataset items into the format expected by Evaluator runners.

Handles the conversion from typed benchmark items to chat messages + tool definitions
that can be passed to LocalModelRunner.run_single() or CloudModelRunner.run_single().
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .schema import MMLUProItem, TAU2BenchItem, TAU2BenchTurn, BFCLItem

logger = logging.getLogger("benchmark.converters")


# ---------------------------------------------------------------------------
# MMLU-Pro → Chat messages (text-based MC, NO tools)
# ---------------------------------------------------------------------------

def mmlu_pro_to_messages(item: dict) -> dict:
    """Convert an MMLU-Pro item dict to {messages, ground_truth, item_id, subject, options}.

    Returns a dict with 'messages' ready for runner, plus metadata for scoring.
    """
    return {
        "item_id": item["item_id"],
        "messages": item["messages"],
        "tools": None,
        "ground_truth": item["ground_truth"],
        "subject": item.get("subject", "unknown"),
        "options": item.get("options", {}),
    }


# ---------------------------------------------------------------------------
# TAU2-Bench → Multi-turn chat
# ---------------------------------------------------------------------------

def tau2_bench_to_turns(item: dict) -> dict:
    """Convert a TAU2-Bench item dict to turn-by-turn data.

    Returns:
        {
            item_id, domain, user_task,
            turns: [{messages, tools, expected_tool_name, expected_tool_arguments,
                     all_expected_actions, tool_response, _mock_responses}, ...]
        }
    """
    return {
        "item_id": item["item_id"],
        "domain": item.get("domain", "unknown"),
        "user_task": item.get("user_task", ""),
        "reward_basis": item.get("reward_basis", []),
        "turns": [
            {
                "turn_index": t["turn_index"],
                "messages": t["messages"],
                "tools": t.get("tools"),
                "expected_tool_name": t.get("expected_tool_name", ""),
                "expected_tool_arguments": t.get("expected_tool_arguments"),
                "all_expected_actions": t.get("all_expected_actions", []),
                "tool_response": t.get("tool_response"),
                "_mock_responses": t.get("_mock_responses", []),
            }
            for t in item.get("turns", [])
        ],
    }


# ---------------------------------------------------------------------------
# BFCL → Chat messages + tools
# ---------------------------------------------------------------------------

def bfcl_to_messages(item: dict) -> dict:
    """Convert a BFCL item dict to {messages, tools, expected_tool_calls, item_id, category}.

    Returns a dict with 'messages' and 'tools' for runner, plus expected calls for scoring.
    Ensures a system message is present.
    """
    messages = list(item.get("messages", []))
    tools = item.get("tools", [])

    # Ensure system message for tool-calling scenarios
    if tools and not any(m.get("role") == "system" for m in messages):
        messages.insert(0, {
            "role": "system",
            "content": "You are a helpful assistant with access to functions. Use the provided functions to answer the user's query.",
        })

    return {
        "item_id": item["item_id"],
        "messages": messages,
        "tools": tools,
        "expected_tool_calls": item.get("expected_tool_calls", []),
        "category": item.get("category", "unknown"),
    }


# ---------------------------------------------------------------------------
# Unified converter — pick the right one based on dataset name
# ---------------------------------------------------------------------------

def convert_item(dataset_name: str, item: dict) -> dict:
    """Convert a raw dataset item to the format appropriate for its dataset type.

    Args:
        dataset_name: One of 'mmlu_pro', 'tau2_bench', 'bfcl'.
        item: Raw item dict from the dataset loader.

    Returns:
        Dict with at least {'item_id', 'messages'} plus dataset-specific metadata.
    """
    name = dataset_name.lower().strip()
    if name in ("mmlu", "mmlu_pro", "mmlu-pro", "mmlupro", "mmstar", "ocrbench"):
        return mmlu_pro_to_messages(item)
    elif name in ("tau2_bench", "tau2-bench", "tau2bench"):
        return tau2_bench_to_turns(item)
    elif name in ("bfcl",):
        return bfcl_to_messages(item)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
