# -*- coding: utf-8 -*-
"""Tests for _apply_tool_hooks — tool return content post-processing.

These tests verify that tool message content is correctly transformed by
registered hooks from ``proxy_server.tools_hook`` before being fed into
the model.  All tests are standalone and do NOT depend on the AMLLLM
native backend.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Minimal fake ChatMessage so we can test without pydantic.
# ---------------------------------------------------------------------------


class _FakeChatMessage:
    def __init__(
        self,
        role: str,
        content: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: list[dict] | None = None,
    ):
        self.role = role
        self.content = content
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls


# ---------------------------------------------------------------------------
# Register a test hook before importing _apply_tool_hooks.
# ---------------------------------------------------------------------------

from proxy_server.tools_hook import tool_hook, get_tool_hook  # noqa: E402


@tool_hook(user_agent="TestAgent", tool="read_file", model="qwen")
def _test_read_file_hook(text: str) -> str:
    """Wrap file content in a JSON envelope for testing."""
    return json.dumps({"type": "file", "content": text}, ensure_ascii=False)


@tool_hook(user_agent="TestAgent", tool="search", model="qwen")
def _test_search_hook(text: str) -> str:
    """Prefix search results for testing."""
    return "[SEARCH_RESULT] " + text


# Now import the function under test.
from amlllm_openai_server.app import _apply_tool_hooks  # noqa: E402


# ============================================================================
# 1.  Basic no-op cases
# ============================================================================


def test_no_user_agent_is_noop():
    """When user_agent is empty, messages are untouched."""
    msgs = [
        _FakeChatMessage(role="tool", content="raw output", tool_call_id="call_1"),
    ]
    _apply_tool_hooks(msgs, user_agent="", model_type="qwen")
    assert msgs[0].content == "raw output"


def test_no_tool_messages_is_noop():
    """When there are no tool-role messages, nothing changes."""
    msgs = [
        _FakeChatMessage(role="user", content="hello"),
        _FakeChatMessage(role="assistant", content="hi there"),
    ]
    snapshot = [m.content for m in msgs]
    _apply_tool_hooks(msgs, user_agent="TestAgent", model_type="qwen")
    assert [m.content for m in msgs] == snapshot


# ============================================================================
# 2.  Hook matching
# ============================================================================


def test_matching_hook_transforms_content():
    """A tool message whose tool_call_id maps to a registered hook
    gets its content transformed."""
    msgs = [
        _FakeChatMessage(
            role="assistant",
            tool_calls=[{
                "id": "call_abc",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        ),
        _FakeChatMessage(
            role="tool",
            content="line1\nline2",
            tool_call_id="call_abc",
        ),
        _FakeChatMessage(role="user", content="thanks"),
    ]
    _apply_tool_hooks(msgs, user_agent="TestAgent", model_type="qwen")
    # The hook wraps content in JSON.
    parsed = json.loads(msgs[1].content)
    assert parsed["type"] == "file"
    assert parsed["content"] == "line1\nline2"
    # Non-tool messages are untouched.
    assert msgs[0].content is None
    assert msgs[2].content == "thanks"


def test_hook_respects_model():
    """Hook registered for model='qwen' should NOT match model='llama'."""
    msgs = [
        _FakeChatMessage(
            role="assistant",
            tool_calls=[{
                "id": "call_xyz",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        ),
        _FakeChatMessage(
            role="tool",
            content="original",
            tool_call_id="call_xyz",
        ),
    ]
    _apply_tool_hooks(msgs, user_agent="TestAgent", model_type="llama")
    # read_file hook is registered for model='qwen' only → identity fallback.
    assert msgs[1].content == "original"


def test_unmatched_tool_name_falls_back_to_identity():
    """When no hook is registered for the tool name, content stays unchanged."""
    msgs = [
        _FakeChatMessage(
            role="assistant",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "unknown_tool", "arguments": "{}"},
            }],
        ),
        _FakeChatMessage(
            role="tool",
            content="keep me",
            tool_call_id="call_1",
        ),
    ]
    _apply_tool_hooks(msgs, user_agent="TestAgent", model_type="qwen")
    assert msgs[1].content == "keep me"


def test_missing_tool_call_id_is_skipped():
    """A tool message without a tool_call_id is left alone."""
    msgs = [
        _FakeChatMessage(
            role="assistant",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        ),
        _FakeChatMessage(
            role="tool",
            content="no id here",
            tool_call_id=None,
        ),
    ]
    _apply_tool_hooks(msgs, user_agent="TestAgent", model_type="qwen")
    assert msgs[1].content == "no id here"


# ============================================================================
# 3.  Multiple tools
# ============================================================================


def test_multiple_tool_calls_with_hooks():
    """Different tool messages are processed by their respective hooks."""
    msgs = [
        _FakeChatMessage(
            role="assistant",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                },
            ],
        ),
        _FakeChatMessage(
            role="tool",
            content="file content",
            tool_call_id="call_1",
        ),
        _FakeChatMessage(
            role="tool",
            content="search results",
            tool_call_id="call_2",
        ),
    ]
    _apply_tool_hooks(msgs, user_agent="TestAgent", model_type="qwen")

    # read_file hook → JSON envelope
    parsed = json.loads(msgs[1].content)
    assert parsed["type"] == "file"
    assert parsed["content"] == "file content"

    # search hook → prefix
    assert msgs[2].content == "[SEARCH_RESULT] search results"


def test_empty_tool_content_skipped():
    """Tool messages with empty content are not processed."""
    msgs = [
        _FakeChatMessage(
            role="assistant",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        ),
        _FakeChatMessage(
            role="tool",
            content="",
            tool_call_id="call_1",
        ),
    ]
    _apply_tool_hooks(msgs, user_agent="TestAgent", model_type="qwen")
    assert msgs[1].content == ""


# ============================================================================
# 4.  User-agent matching (prefix match)
# ============================================================================


def test_user_agent_prefix_match():
    """Hooks match by User-Agent prefix (case-insensitive)."""
    msgs = [
        _FakeChatMessage(
            role="assistant",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        ),
        _FakeChatMessage(
            role="tool",
            content="data",
            tool_call_id="call_1",
        ),
    ]
    # "TestAgent/1.0" starts with "testagent" → should match
    _apply_tool_hooks(msgs, user_agent="TestAgent/1.0 (Linux)", model_type="qwen")
    parsed = json.loads(msgs[1].content)
    assert parsed["content"] == "data"


def test_user_agent_no_prefix_match():
    """A User-Agent that does NOT start with the registered prefix → no hook."""
    msgs = [
        _FakeChatMessage(
            role="assistant",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "read_file", "arguments": "{}"},
            }],
        ),
        _FakeChatMessage(
            role="tool",
            content="original",
            tool_call_id="call_1",
        ),
    ]
    _apply_tool_hooks(msgs, user_agent="OtherAgent/2.0", model_type="qwen")
    assert msgs[1].content == "original"


# ============================================================================
# 5.  Model=None in hook registration (wildcard model)
# ============================================================================


@tool_hook(user_agent="WildcardAgent", tool="calculate", model=None)
def _test_wildcard_hook(text: str) -> str:
    return "CALC:" + text


def test_wildcard_model_hook():
    """A hook registered with model=None matches only when
    the caller passes model_type='none' or '' (treated as None)."""
    msgs = [
        _FakeChatMessage(
            role="assistant",
            tool_calls=[{
                "id": "call_1",
                "type": "function",
                "function": {"name": "calculate", "arguments": "{}"},
            }],
        ),
        _FakeChatMessage(
            role="tool",
            content="42",
            tool_call_id="call_1",
        ),
    ]
    # Wildcard hook matches when model_type is "none" (treated as None).
    _apply_tool_hooks(msgs, user_agent="WildcardAgent", model_type="none")
    assert msgs[1].content == "CALC:42"

    # Reset and test with empty string (also treated as None).
    msgs[1].content = "42"
    _apply_tool_hooks(msgs, user_agent="WildcardAgent", model_type="")
    assert msgs[1].content == "CALC:42"

    # With a specific model, wildcard hook should NOT match.
    msgs[1].content = "42"
    _apply_tool_hooks(msgs, user_agent="WildcardAgent", model_type="qwen")
    assert msgs[1].content == "42"  # unchanged
