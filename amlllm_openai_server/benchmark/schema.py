# -*- coding: utf-8 -*-
"""Pydantic schemas for the agent capability benchmark framework.

Defines:
- Internal dataset item representations (MMLU-Pro, TAU2-Bench, BFCL)
- BenchmarkRunRecord (extends RunRecord with benchmark-specific fields)
- BenchmarkScoreReport (aggregated scores per dataset/model)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field


from ..schema import CloudApiConfig  # noqa: E402


# ---------------------------------------------------------------------------
# Unified Dataset Item Types
# ---------------------------------------------------------------------------


class MMLUProItem(BaseModel):
    """A single MMLU-Pro multiple-choice question."""

    item_id: str
    """Unique identifier within MMLU-Pro."""

    subject: str
    """Subject category (e.g., 'biology', 'computer_science')."""

    difficulty: str = ""
    """Difficulty level if available."""

    messages: list[dict]
    """Chat messages to send to the model:
    [system, user] where user contains the question + options."""

    ground_truth: str
    """Correct answer letter (A-J)."""

    options: dict[str, str] = Field(default_factory=dict)
    """Mapping from answer letter to option text (for reporting)."""


class TAU2BenchTurn(BaseModel):
    """A single turn in a TAU2-Bench multi-turn trajectory."""

    turn_index: int
    """Zero-based turn number."""

    messages: list[dict]
    """Messages up to this turn (including conversation history)."""

    tools: Optional[list[dict]] = None
    """Tool definitions available at this turn."""

    expected_tool_name: str
    """Primary expected tool name the model should call."""

    expected_tool_arguments: Optional[dict] = None
    """Expected tool call arguments for the primary tool (for detailed comparison)."""

    all_expected_actions: list[dict] = []
    """All expected actions this turn: [{name, arguments}, ...].
    Supports parallel tool calls within a single turn."""

    tool_response: Optional[dict] = None
    """Ground-truth tool response to feed back for the next turn."""


class TAU2BenchItem(BaseModel):
    """A full TAU2-Bench task (multi-turn trajectory)."""

    item_id: str
    """Unique identifier within TAU2-Bench."""

    domain: str
    """Task domain (e.g., 'airline', 'retail', 'housing')."""

    turns: list[TAU2BenchTurn]
    """Ordered list of conversation turns."""

    user_task: str = ""
    """Description of the user's task/goal."""


class BFCLItem(BaseModel):
    """A single BFCL function-calling test item."""

    item_id: str
    """Unique identifier within BFCL."""

    category: str
    """BFCL category: 'simple', 'parallel', 'multiple', 'parallel_multiple',
    'irrelevance', 'relevance'."""

    language: str = "python"
    """Source language for the test entry: 'python', 'java', 'javascript'."""

    messages: list[dict]
    """Chat messages (system + user) for the conversation."""

    tools: list[dict]
    """Function definitions available to the model."""

    expected_tool_calls: list[dict]
    """Expected tool calls: [{name, arguments}, ...]"""


# ---------------------------------------------------------------------------
# Benchmark Run Record (extends tools.schema.RunRecord)
# ---------------------------------------------------------------------------


class BenchmarkRunRecord(BaseModel):
    """A single inference run record for benchmark evaluation, written as one JSONL line.

    Designed to be compatible with tools.schema.RunRecord but adds benchmark-specific fields.
    """

    # --- identity ---
    entry_index: int = 0
    conv_index: int = 0
    repeat: int = 0
    model_type: str = ""
    dataset: str = ""
    item_id: str = ""
    is_summary: bool = False

    # --- expectation ---
    expected_tool_name: Optional[str] = None
    has_verifier: bool = False

    # --- actual result ---
    actual_has_tool_calls: bool = False
    tool_call_count: int = 0
    tool_names: list[str] = Field(default_factory=list)

    # --- correctness ---
    name_matched: bool = False
    verifier_passed: Optional[bool] = None
    verifier_reason: Optional[str] = None
    is_correct: Optional[bool] = None
    """Set during scoring phase: whether this run produced the correct answer/tool calls."""

    score_detail: Optional[dict] = None
    """Dataset-specific scoring metadata (e.g., extracted answer, expected answer)."""

    # --- raw response ---
    response_text: Optional[str] = None
    response_tool_calls: Optional[list[dict]] = None

    # --- meta ---
    finish_reason: str = "stop"
    error: Optional[str] = None
    latency_ms: float = 0.0
    prefill_ms: Optional[float] = None
    """Prefill/prompt processing time in milliseconds. From API: timings.prompt_ms."""
    decode_ms: Optional[float] = None
    """Decode/generation time in milliseconds. From API: timings.predicted_ms."""
    ttft_ms: Optional[float] = None
    """Time to First Token: prefill phase duration in milliseconds (ADLA only)."""
    tps: Optional[float] = None
    """Tokens Per Second: decode throughput averaged over generated tokens (ADLA only)."""
    total_tokens: int = 0
    """Total tokens used (prompt + completion). From API: usage.total_tokens."""
    completion_tokens: int = 0
    """Completion/generated tokens. From API: usage.completion_tokens."""
    token_count: int = 0
    """Alias for completion_tokens (backward compat)."""
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Scoring Types
# ---------------------------------------------------------------------------


class DatasetScore(BaseModel):
    """Per-dataset score breakdown for a single model."""

    dataset: str
    model_type: str
    total_items: int = 0
    correct_items: int = 0
    accuracy: float = 0.0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    avg_ttft_ms: Optional[float] = None
    avg_tps: Optional[float] = None

    # Per-subject/category breakdowns
    breakdown: dict[str, dict] = Field(default_factory=dict)
    """Mapping from subject/category/domain -> {total, correct, accuracy}."""


class BenchmarkScoreReport(BaseModel):
    """Full benchmark scoring report across all datasets and models."""

    scores: list[DatasetScore] = Field(default_factory=list)
    """Per-dataset, per-model score breakdowns."""

    timestamp: str = ""

    @property
    def model_types(self) -> list[str]:
        return sorted(set(s.model_type for s in self.scores))

    @property
    def datasets(self) -> list[str]:
        return sorted(set(s.dataset for s in self.scores))

    def get_score(self, dataset: str, model_type: str) -> Optional[DatasetScore]:
        for s in self.scores:
            if s.dataset == dataset and s.model_type == model_type:
                return s
        return None
