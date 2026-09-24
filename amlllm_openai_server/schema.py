# -*- coding: utf-8 -*-
"""Pydantic schemas for the tool-call evaluation framework.

Defines the test data model and cloud API configuration model loaded from
``test_data/test_config.yaml``.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Callable, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger("tools.schema")


# ---------------------------------------------------------------------------
# Cloud API Configuration (from test_data/test_config.yaml)
# ---------------------------------------------------------------------------


class CloudApiConfig(BaseModel):
    """OpenAI-compatible cloud API endpoint configuration."""

    base_url: str
    api_key: str
    model: str
    thinking: bool = False
    """Whether to enable thinking/reasoning mode (passed via extra_body.chat_template_kwargs.enable_thinking)."""

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "CloudApiConfig":
        """Load cloud API config from a YAML file.

        Expected format::

            remote-api:
              base_url: http://...
              api_key: ...
              model: Qwen3.5-4B
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Cloud config file not found: {yaml_path}")
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        remote = raw.get("remote-api") or raw.get("remote_api") or {}
        if not remote:
            raise ValueError(
                f"Cloud config at {yaml_path} must have a 'remote-api' key"
            )
        return cls(
            base_url=str(remote.get("base_url", "")).rstrip("/"),
            api_key=str(remote.get("api_key", "")),
            model=str(remote.get("model", "")),
            thinking=bool(remote.get("thinking", False)),
        )


# ---------------------------------------------------------------------------
# Verifier — dynamically loaded callable for validating tool call arguments
# ---------------------------------------------------------------------------


class VerifierSpec(BaseModel):
    """Specification for a verifier function that validates tool call arguments.

    ``path`` must point to a ``.py`` file relative to the project root.
    ``function`` is the name of a callable within that file that accepts
    a single ``str`` (the function arguments JSON string) and returns ``bool``.
    """

    path: str
    function: str
    kwargs: Optional[dict] = Field(default_factory=dict)


def load_verifier(
    spec: VerifierSpec,
    project_root: Optional[Path] = None,
) -> Optional[Callable[[str], tuple[bool, Optional[str]]]]:
    """Dynamically import a verifier function from *spec*.

    The function must have the signature ``(arguments_json: str, **kwargs) -> (bool, str|None)``
    where the bool indicates pass/fail and the optional string is a failure reason.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    full_path = Path(spec.path)
    if not full_path.is_absolute():
        full_path = project_root / full_path

    if not full_path.exists():
        logger.warning("Verifier file not found: %s", full_path)
        return None

    module_name = f"_verifier_{full_path.stem}"
    try:
        spec_loader = importlib.util.spec_from_file_location(
            module_name, str(full_path)
        )
        if spec_loader is None or spec_loader.loader is None:
            logger.warning("Could not create module spec for: %s", full_path)
            return None
        module = importlib.util.module_from_spec(spec_loader)
        spec_loader.loader.exec_module(module)
    except Exception as exc:
        logger.warning("Failed to load verifier module %s: %s", full_path, exc)
        return None

    func = getattr(module, spec.function, None)
    if func is None:
        logger.warning(
            "Function '%s' not found in verifier module %s",
            spec.function,
            full_path,
        )
        return None
    if not callable(func):
        logger.warning(
            "Attribute '%s' in %s is not callable",
            spec.function,
            full_path,
        )
        return None
    return lambda args: func(args, **(spec.kwargs or {}))


# ---------------------------------------------------------------------------
# Tool Expectation Entry
# ---------------------------------------------------------------------------


class ToolExpectEntry(BaseModel):
    """Per-conversation tool call expectation.

    - ``name`` is ``None`` → the model must **not** produce a tool call.
    - ``name`` is a string → the model must produce a tool call with that exact name.
    - ``verifier``, when provided alongside a non-null ``name``, is loaded and
      called with the tool call's ``function.arguments`` JSON string.  The run
      is only fully correct when the verifier returns ``True``.
    """

    name: Optional[str] = None
    """Expected tool name, or ``None`` if no tool call is expected."""

    verifier: Optional[VerifierSpec] = None
    """Optional verifier for the tool call arguments."""

    # ---- helpers ----

    @property
    def expects_tool_call(self) -> bool:
        return self.name is not None

    @property
    def has_verifier(self) -> bool:
        return self.verifier is not None


# ---------------------------------------------------------------------------
# Test Case Entry
# ---------------------------------------------------------------------------


class TestCaseEntry(BaseModel):
    """A single test entry that may contain multiple independent conversations.

    ``messages`` is a list of conversation lists.  Each inner list is a complete,
    independent conversation consisting of :class:`ChatMessage` dicts.
    ``tool_expect`` is a parallel array: ``len(messages) == len(tool_expect)``.
    """

    __test__ = False  # prevent pytest from collecting this as a test class

    tools: Optional[list[dict]] = None
    """Shared tool definitions for all conversations in this entry."""

    max_tokens: int = 4096
    """Maximum generation tokens shared across conversations."""

    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stop: Optional[Union[str, list[str]]] = None
    tool_choice: Optional[Union[str, dict]] = None

    messages: list[list[dict]]
    """Each inner list is an independent complete conversation (list of ChatMessage dicts)."""

    tool_expect: list[ToolExpectEntry]
    """Per-conversation expectation: name=null means no tool; name=str means expect that tool."""

    @model_validator(mode="after")
    def _validate_lengths(self) -> "TestCaseEntry":
        if len(self.messages) != len(self.tool_expect):
            raise ValueError(
                f"len(messages)={len(self.messages)} must equal "
                f"len(tool_expect)={len(self.tool_expect)}"
            )
        return self


# ---------------------------------------------------------------------------
# Evaluation result record (one per run)
# ---------------------------------------------------------------------------


class RunRecord(BaseModel):
    """A single inference run record, written as one JSONL line."""

    entry_index: int
    conv_index: int
    repeat: int
    model_type: str  # "local" or "cloud"

    # --- expectation ---
    expected_tool_name: Optional[str] = None
    """Expected tool name from the test case (null = no tool expected)."""

    has_verifier: bool = False
    """Whether the test case included a verifier for this conversation."""

    # --- actual result ---
    actual_has_tool_calls: bool
    tool_call_count: int = 0
    tool_names: list[str] = Field(default_factory=list)

    # --- correctness ---
    name_matched: bool = False
    """True when expected_tool_name is not None and the model returned a tool call with that name."""

    verifier_passed: Optional[bool] = None
    """None = no verifier; True = verifier passed; False = verifier failed."""

    verifier_reason: Optional[str] = None
    """When verifier_passed is False, the reason string returned by the verifier."""

    # --- raw response (for debugging) ---
    response_text: Optional[str] = None
    """Raw text content produced by the model (None if only tool_calls were emitted)."""

    response_tool_calls: Optional[list[dict]] = None
    """Full tool_calls array from the model response, including id/type/function/arguments."""

    # --- meta ---
    finish_reason: str = "stop"
    error: Optional[str] = None
    latency_ms: float = 0.0
    prefill_ms: Optional[float] = None
    """Time spent in prefill (prompt processing) phase, in milliseconds."""
    decode_ms: Optional[float] = None
    """Time spent in decode (token generation) phase, in milliseconds."""
    token_count: int = 0
    timestamp: str = ""


# ---------------------------------------------------------------------------
# Statistics result
# ---------------------------------------------------------------------------


class StatsResult(BaseModel):
    """Aggregated statistics for a single model type."""

    model_type: str
    total_runs: int = 0

    # ---- presence-level (TP/FP/FN/TN) ----
    tp: int = 0   # expected a tool call & got one (any name)
    fp: int = 0   # expected NO tool call but got one
    fn: int = 0   # expected a tool call but got none
    tn: int = 0   # expected NO tool call & got none

    # ---- name-level (subset of TP) ----
    name_match_count: int = 0
    """Of the TP runs, how many had the correct tool name."""

    name_mismatch_count: int = 0
    """Of the TP runs, how many had a tool call but with the wrong name."""

    # ---- verifier-level (subset of name_match_count) ----
    verifier_total: int = 0
    """Number of correct-name runs that had a verifier."""

    verifier_pass_count: int = 0
    """Number of verifier runs that passed."""

    # ---- meta ----
    error_count: int = 0
    avg_latency_ms: float = 0.0

    # ---- computed properties ----

    @property
    def total_labeled(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def recall(self) -> float:
        """Recall at presence level: TP / (TP + FN)."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision at presence level: TP / (TP + FP)."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def accuracy(self) -> float:
        """Accuracy at presence level: (TP + TN) / total_labeled."""
        denom = self.total_labeled
        return (self.tp + self.tn) / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """F1 at presence level."""
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def name_accuracy(self) -> float:
        """Proportion of tool-call runs that used the correct tool name.

        = name_match_count / (name_match_count + name_mismatch_count)
        """
        denom = self.name_match_count + self.name_mismatch_count
        return self.name_match_count / denom if denom > 0 else 0.0

    @property
    def verifier_pass_rate(self) -> float:
        """Proportion of verifier-tested runs that passed."""
        return (
            self.verifier_pass_count / self.verifier_total
            if self.verifier_total > 0
            else 1.0  # no verifier → trivially "all passed"
        )

    # ---- serialization ----

    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "total_runs": self.total_runs,
            "total_labeled": self.total_labeled,
            # Presence
            "TP": self.tp,
            "FP": self.fp,
            "FN": self.fn,
            "TN": self.tn,
            "Recall": round(self.recall, 4),
            "Precision": round(self.precision, 4),
            "Accuracy": round(self.accuracy, 4),
            "F1": round(self.f1, 4),
            # Name
            "NameMatch": self.name_match_count,
            "NameMismatch": self.name_mismatch_count,
            "NameAccuracy": round(self.name_accuracy, 4),
            # Verifier
            "VerifierTotal": self.verifier_total,
            "VerifierPassed": self.verifier_pass_count,
            "VerifierPassRate": round(self.verifier_pass_rate, 4),
            # Meta

            "error_count": self.error_count,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
        }
