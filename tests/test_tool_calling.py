# -*- coding: utf-8 -*-
"""Standalone tests for AmlLlmModelRuntime function calling.

These tests verify the tool-calling logic without depending on the AMLLLM
platform-specific runtime.  All parsing, prompt-building, and result-formatting
helpers are exercised against representative inputs.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

# Ensure the proxy_server package is importable without pulling in heavy deps.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Minimal stand-ins so we can test without FastAPI / pydantic / AMLLLM.
# ---------------------------------------------------------------------------


class _FakeMessagePart:
    def __init__(self, type_: str = "text", text: str | None = None):
        self.type = type_
        self.text = text


class _FakeChatMessage:
    def __init__(
        self,
        role: str,
        content: str | list[_FakeMessagePart] | None = None,
        name: str | None = None,
        tool_call_id: str | None = None,
        tool_calls: list[dict] | None = None,
    ):
        self.role = role
        self.content = content
        self.name = name
        self.tool_call_id = tool_call_id
        self.tool_calls = tool_calls


class _FakeChatCompletionToolFunction:
    def __init__(self, name: str, description: str | None = None, parameters: dict | None = None):
        self.name = name
        self.description = description
        self.parameters = parameters

    def model_dump(self, **__):
        result: dict = {"name": self.name}
        if self.description is not None:
            result["description"] = self.description
        if self.parameters is not None:
            result["parameters"] = self.parameters
        return result


class _FakeChatCompletionTool:
    def __init__(self, type_: str, function: _FakeChatCompletionToolFunction):
        self.type = type_
        self.function = function

    def model_dump(self, **__):
        return {"type": self.type, "function": self.function.model_dump()}


# ---------------------------------------------------------------------------
# Import the real helpers from proxy_server.app
# ---------------------------------------------------------------------------

from amlllm_openai_server.app import (  # noqa: E402
    _build_tool_instruction,
    _clean_template_artifacts,
    _extract_tool_call_blocks,
    _fallback_tool_calls_from_text,
    _make_tool_call,
    _merge_system_prompt,
    _message_text,
    _messages_to_prompt,
    _normalize_tool_choice,
    _normalize_tools,
    _system_prompt_from_messages,
    _text_without_tool_markup,
    _tool_names,
)

# ---------------------------------------------------------------------------
# 1.  _build_tool_instruction  –  prompt injection
# ---------------------------------------------------------------------------


def test_build_tool_instruction_empty():
    assert _build_tool_instruction(None, None) == ""
    assert _build_tool_instruction([], "auto") == ""


def test_build_tool_instruction_single_tool():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    text = _build_tool_instruction(tools, "auto")
    assert "get_weather" in text
    assert "Get the current weather" in text
    assert "city" in text
    assert "<tool_call>" in text
    # "tool_choice" appears in the instruction text itself ("If tool_choice is auto..."),
    # but the explicit "tool_choice: auto" line is NOT injected when choice is "auto".
    # We verify the instruction body is present without the explicit directive.
    assert "tool_choice: auto" not in text.lower()
    assert "If tool_choice is auto" in text


def test_build_tool_instruction_explicit_tool_choice():
    tools = [{"type": "function", "function": {"name": "search", "description": "Search web"}}]
    text = _build_tool_instruction(tools, "required")
    assert "tool_choice" in text
    assert "required" in text


def test_build_tool_instruction_multiple_tools():
    tools = [
        {"type": "function", "function": {"name": "tool_a", "description": "A tool"}},
        {"type": "function", "function": {"name": "tool_b", "description": "B tool"}},
    ]
    text = _build_tool_instruction(tools, None)
    assert "tool_a" in text
    assert "tool_b" in text


def test_build_tool_instruction_missing_function():
    """Tools without a 'function' key are silently skipped, but the instruction
    preamble is still generated because tools list is non-empty."""
    text = _build_tool_instruction([{"type": "function"}], None)
    # preamble exists since a non-empty list was passed
    assert "You have access to tools" in text
    assert "<tool_call>" in text
    # no tool names listed
    assert "- name:" not in text


# ---------------------------------------------------------------------------
# 2.  _extract_tool_call_blocks  –  regex parsing
# ---------------------------------------------------------------------------


def test_extract_single_block():
    text = '<tool_call>{"name":"search","arguments":{"q":"hello"}}</tool_call>'
    blocks = _extract_tool_call_blocks(text)
    assert blocks == [{"name": "search", "arguments": {"q": "hello"}}]


def test_extract_multiple_blocks():
    text = (
        '<tool_call>{"name":"a","arguments":{}}</tool_call>'
        "\nsome text\n"
        '<tool_call>{"name":"b","arguments":{"x":1}}</tool_call>'
    )
    blocks = _extract_tool_call_blocks(text)
    assert len(blocks) == 2
    assert blocks[0]["name"] == "a"
    assert blocks[1]["name"] == "b"


def test_extract_list_inside_block():
    text = '<tool_call>[{"name":"a"},{"name":"b"}]</tool_call>'
    blocks = _extract_tool_call_blocks(text)
    assert len(blocks) == 2


def test_extract_invalid_json():
    text = '<tool_call>not json</tool_call>'
    assert _extract_tool_call_blocks(text) == []


def test_extract_no_blocks():
    assert _extract_tool_call_blocks("plain text") == []
    assert _extract_tool_call_blocks("") == []


def test_extract_whitespace_insensitive():
    text = '<tool_call>\n  {"name": "x", "arguments": {}}  \n</tool_call>'
    blocks = _extract_tool_call_blocks(text)
    assert blocks == [{"name": "x", "arguments": {}}]


# ---------------------------------------------------------------------------
# 3.  _fallback_tool_calls_from_text  –  full parse path
# ---------------------------------------------------------------------------


def test_fallback_basic():
    tools = [{"type": "function", "function": {"name": "search"}}]
    text = '<tool_call>{"name":"search","arguments":{"q":"hi"}}</tool_call>'
    calls = _fallback_tool_calls_from_text(text, tools)
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    func = calls[0]["function"]
    assert func["name"] == "search"
    assert json.loads(func["arguments"]) == {"q": "hi"}


def test_fallback_unknown_tool_filtered():
    """A tool_call for a name not in the tool list is filtered out."""
    tools = [{"type": "function", "function": {"name": "allowed"}}]
    text = '<tool_call>{"name":"disallowed","arguments":{}}</tool_call>'
    assert _fallback_tool_calls_from_text(text, tools) == []


def test_fallback_empty_tool_list_accepts_all():
    """When no tools are provided, all tool_calls in text are accepted."""
    text = '<tool_call>{"name":"anything","arguments":{}}</tool_call>'
    calls = _fallback_tool_calls_from_text(text, None)
    assert len(calls) == 1


def test_fallback_function_sub_key():
    """Block uses {"function": {"name":...}} format."""
    tools = [{"type": "function", "function": {"name": "run"}}]
    text = '<tool_call>{"function":{"name":"run","arguments":"{}"}}</tool_call>'
    calls = _fallback_tool_calls_from_text(text, tools)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "run"


def test_fallback_arguments_string_parsed():
    tools = [{"type": "function", "function": {"name": "calc"}}]
    text = '<tool_call>{"name":"calc","arguments":"{\\"expr\\":\\"1+1\\"}"}</tool_call>'
    calls = _fallback_tool_calls_from_text(text, tools)
    args = json.loads(calls[0]["function"]["arguments"])
    assert args == {"expr": "1+1"}


def test_fallback_python_exec_fallback():
    """Gemma-style tool_code blocks -> python_exec tool."""
    tools = [
        {"type": "function", "function": {"name": "python_exec"}},
        {"type": "function", "function": {"name": "search"}},
    ]
    text = "<start_of_turn>tool_code\nprint(1)\n</start_of_turn>"
    calls = _fallback_tool_calls_from_text(text, tools)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "python_exec"
    args = json.loads(calls[0]["function"]["arguments"])
    assert args["code"] == "print(1)"


def test_fallback_single_tool_code_fallback():
    """When only one tool exists and it's not python_exec, still match."""
    tools = [{"type": "function", "function": {"name": "runner"}}]
    text = "<start_of_turn>tool_code\necho hello\n</start_of_turn>"
    calls = _fallback_tool_calls_from_text(text, tools)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "runner"


def test_fallback_empty_text():
    assert _fallback_tool_calls_from_text("", None) == []
    assert _fallback_tool_calls_from_text("", []) == []


def test_fallback_dedup_tool_code():
    """Duplicate tool_code blocks are deduplicated."""
    tools = [{"type": "function", "function": {"name": "python_exec"}}]
    text = (
        "<start_of_turn>tool_code\nx=1\n</start_of_turn>"
        "<start_of_turn>tool_code\nx=1\n</start_of_turn>"
    )
    calls = _fallback_tool_calls_from_text(text, tools)
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# 4.  _make_tool_call  –  dict shape
# ---------------------------------------------------------------------------


def test_make_tool_call_shape():
    call = _make_tool_call("my_func", {"a": 1})
    assert call["type"] == "function"
    assert call["id"].startswith("call_")
    assert len(call["id"]) > 20
    func = call["function"]
    assert func["name"] == "my_func"
    assert json.loads(func["arguments"]) == {"a": 1}


# ---------------------------------------------------------------------------
# 5.  _merge_system_prompt  –  combining base + tool instructions
# ---------------------------------------------------------------------------


def test_merge_no_tools_returns_base():
    assert _merge_system_prompt("base", None, None) == "base"
    assert _merge_system_prompt("base", [], "auto") == "base"


def test_merge_no_base():
    tools = [{"type": "function", "function": {"name": "f1"}}]
    merged = _merge_system_prompt("", tools, None)
    assert merged == _build_tool_instruction(tools, None)


def test_merge_with_base():
    tools = [{"type": "function", "function": {"name": "f1"}}]
    merged = _merge_system_prompt("You are helpful.", tools, None)
    assert merged.startswith("You are helpful.")
    assert "f1" in merged
    assert "<tool_call>" in merged


# ---------------------------------------------------------------------------
# 6.  _text_without_tool_markup  –  cleaning
# ---------------------------------------------------------------------------


def test_text_without_tool_markup_removes_tool_call_blocks():
    text = "before <tool_call>{\"name\":\"x\"}</tool_call> after"
    cleaned = _text_without_tool_markup(text)
    assert "<tool_call>" not in cleaned
    assert "before" in cleaned
    assert "after" in cleaned


def test_text_without_tool_markup_removes_tool_code_blocks():
    text = "start <start_of_turn>tool_code\nx=1\n</start_of_turn> end"
    cleaned = _text_without_tool_markup(text)
    assert "tool_code" not in cleaned
    assert "start" in cleaned
    assert "end" in cleaned


# ---------------------------------------------------------------------------
# 7.  _clean_template_artifacts  –  removing think / turn tags
# ---------------------------------------------------------------------------


def test_clean_removes_think_tags():
    text = "<think>internal reasoning</think>actual answer"
    result = _clean_template_artifacts(text)
    assert "think" not in (result or "")
    assert "actual answer" in (result or "")


def test_clean_removes_turn_tags():
    text = "<start_of_turn>hello</start_of_turn>"
    result = _clean_template_artifacts(text)
    assert "<start_of_turn>" not in (result or "")
    assert "hello" in (result or "")


def test_clean_returns_none_for_empty():
    # Empty input is returned as-is (falsy), not converted to None.
    assert _clean_template_artifacts("") == ""
    assert _clean_template_artifacts(None) is None
    # Whitespace-only input becomes None.
    assert _clean_template_artifacts("   ") is None


# ---------------------------------------------------------------------------
# 8.  _messages_to_prompt  –  prompt building (used by AmlLlmModelRuntime)
# ---------------------------------------------------------------------------


def test_messages_to_prompt_single_user():
    msg = _FakeChatMessage(role="user", content="Hello")
    assert _messages_to_prompt([msg]) == "Hello"


def test_messages_to_prompt_conversation():
    msgs = [
        _FakeChatMessage(role="user", content="Hi"),
        _FakeChatMessage(role="assistant", content="Hello!"),
        _FakeChatMessage(role="user", content="How are you?"),
    ]
    prompt = _messages_to_prompt(msgs)
    assert "User: Hi" in prompt
    assert "Assistant: Hello!" in prompt
    assert "User: How are you?" in prompt


def test_messages_to_prompt_filters_system():
    msgs = [
        _FakeChatMessage(role="system", content="Be helpful"),
        _FakeChatMessage(role="user", content="Hi"),
    ]
    prompt = _messages_to_prompt(msgs)
    assert "Be helpful" not in prompt
    # Single user message -> raw content, no "User:" prefix
    assert prompt == "Hi"


# ---------------------------------------------------------------------------
# 9.  _system_prompt_from_messages  –  extracting system prompt
# ---------------------------------------------------------------------------


def test_system_prompt_extraction():
    msgs = [
        _FakeChatMessage(role="system", content="You are helpful."),
        _FakeChatMessage(role="user", content="Hi"),
        _FakeChatMessage(role="system", content="Also be concise."),
    ]
    sp = _system_prompt_from_messages(msgs)
    assert "You are helpful." in sp
    assert "Also be concise." in sp
    assert "Hi" not in sp


def test_system_prompt_empty():
    msgs = [_FakeChatMessage(role="user", content="Hi")]
    assert _system_prompt_from_messages(msgs) == ""


# ---------------------------------------------------------------------------
# 10.  Result-formatting integration  –  simulate AmlLlmModelRuntime output
# ---------------------------------------------------------------------------


def build_fake_result(
    raw_text: str,
    tools: list[dict] | None,
    finish_reason: str = "stop",
) -> dict:
    """Replicate the post-generation logic that AmlLlmModelRuntime now uses."""
    tool_calls = []
    if tools:
        tool_calls = _fallback_tool_calls_from_text(raw_text, tools)
    if tool_calls:
        finish_reason = "tool_calls"
        cleaned = _text_without_tool_markup(raw_text) or None
    else:
        cleaned = _clean_template_artifacts(raw_text)
    return {
        "text": cleaned,
        "tool_calls": tool_calls,
        "finish_reason": finish_reason,
        "token_count": len(raw_text),
    }


def _openai_response_message(result: dict) -> dict:
    """Mirrors _response_message from app.py."""
    message = {"role": "assistant", "content": result.get("text", "")}
    tool_calls = result.get("tool_calls") or []
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def test_result_no_tools():
    result = build_fake_result("Hello, world!", tools=None)
    assert result["text"] == "Hello, world!"
    assert result["tool_calls"] == []
    assert result["finish_reason"] == "stop"
    msg = _openai_response_message(result)
    assert msg["role"] == "assistant"
    assert msg["content"] == "Hello, world!"
    assert "tool_calls" not in msg


def test_result_with_tool_call():
    tools = [{"type": "function", "function": {"name": "search", "description": "..."}}]
    raw = '<tool_call>{"name":"search","arguments":{"q":"weather"}}</tool_call>'
    result = build_fake_result(raw, tools=tools)
    assert result["finish_reason"] == "tool_calls"
    assert len(result["tool_calls"]) == 1
    assert result["text"] is None  # no non-tool content
    msg = _openai_response_message(result)
    assert msg["tool_calls"] == result["tool_calls"]


def test_result_mixed_text_and_tool():
    tools = [{"type": "function", "function": {"name": "search", "description": "..."}}]
    raw = 'Let me search.\n<tool_call>{"name":"search","arguments":{"q":"x"}}</tool_call>\nDone.'
    result = build_fake_result(raw, tools=tools)
    assert result["finish_reason"] == "tool_calls"
    assert len(result["tool_calls"]) == 1
    # The non-tool text remains after _text_without_tool_markup
    assert "Let me search" in (result["text"] or "")
    assert "Done" in (result["text"] or "")
    msg = _openai_response_message(result)
    assert "content" in msg
    assert msg["tool_calls"] is not None


def test_result_streaming_shape():
    """Verify that the result dict can be consumed by _stream_tool_call_events."""
    from amlllm_openai_server.app import _stream_tool_call_events

    result = {
        "text": None,
        "tool_calls": [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"q":"hello"}',
                },
            }
        ],
        "finish_reason": "tool_calls",
    }
    events = list(_stream_tool_call_events("req-1", 123, "test-model", result))
    assert len(events) >= 3  # role delta + tool_call chunk + finish + [DONE]
    assert any("search" in e for e in events)
    assert events[-1] == "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# 11.  _message_text  –  part-list handling
# ---------------------------------------------------------------------------


def test_message_text_string():
    assert _message_text("  hello  ") == "hello"


def test_message_text_none():
    assert _message_text(None) == ""


def test_message_text_parts():
    parts = [
        _FakeMessagePart(type_="text", text="part1"),
        _FakeMessagePart(type_="image_url"),
        _FakeMessagePart(type_="text", text="part2"),
    ]
    assert _message_text(parts) == "part1\npart2"


# ---------------------------------------------------------------------------
# 12.  _tool_names
# ---------------------------------------------------------------------------


def test_tool_names():
    tools = [
        {"type": "function", "function": {"name": "a"}},
        {"type": "function", "function": {"name": "b"}},
        {"type": "function"},  # missing function
    ]
    assert _tool_names(tools) == ["a", "b"]
    assert _tool_names(None) == []
    assert _tool_names([]) == []


# ---------------------------------------------------------------------------
# 13.  _normalize_tools / _normalize_tool_choice
# ---------------------------------------------------------------------------


def test_normalize_tools_converts():
    tool = _FakeChatCompletionTool(
        type_="function",
        function=_FakeChatCompletionToolFunction(name="f", description="d"),
    )
    result = _normalize_tools([tool])
    assert result == [{"type": "function", "function": {"name": "f", "description": "d"}}]


def test_normalize_tools_none():
    assert _normalize_tools(None) is None
    assert _normalize_tools([]) is None


def test_normalize_tool_choice():
    assert _normalize_tool_choice(None) is None
    assert _normalize_tool_choice("auto") == "auto"
    assert _normalize_tool_choice({"type": "function"}) == {"type": "function"}


# ---------------------------------------------------------------------------
# 14.  Integration: full end-to-end AmlLlmModelRuntime simulation
# ---------------------------------------------------------------------------


class FakeClient:
    """Mimics the AMLLLMLite API surface used by AmlLlmModelRuntime."""

    def __init__(self, fake_response: str):
        self.fake_response = fake_response
        self.on_token = None
        self.break_called = False
        self.retain_history = False

    def config(self, **kwargs):
        self.on_token = kwargs.get("on_token")

    def init(self):
        pass

    def uninit(self):
        pass

    def set_chat_template(self, system_prompt, prompt_prefix, prompt_postfix):
        self.system_prompt = system_prompt
        self.prompt_prefix = prompt_prefix
        self.prompt_postfix = prompt_postfix

    def run(self, prompt, retain_history=False, user_data=None):
        self.retain_history = retain_history
        # Simulate token-by-token callback
        if self.on_token:
            for char in self.fake_response:
                self.on_token({"text": char, "status": type("S", (), {"name": "NORMAL"})()})
        return {"text": self.fake_response, "token_count": len(self.fake_response)}

    def break_generation(self):
        self.break_called = True

    def reset_session(self):
        pass


def test_end_to_end_no_tools():
    """Simulate a plain chat completion through AmlLlmModelRuntime flow."""
    from amlllm_openai_server.app import AmlLlmModelRuntime, ModelConfig

    config = ModelConfig(
        name="test-model",
        model_path="/fake/weights.bin",
        model_type="none",
    )
    runtime = AmlLlmModelRuntime.__new__(AmlLlmModelRuntime)
    runtime.config = config
    runtime.lock = __import__("threading").Lock()
    runtime.client = FakeClient("Hello, user!")
    runtime.initialized = True
    runtime.current_signature = None  # force re-config
    runtime.queue = None
    runtime._chat_template_str = ""
    runtime._tokenizer_config = {}

    messages = [_FakeChatMessage(role="user", content="Hi")]
    result = runtime.run(messages=messages, user_data=None)

    assert result["text"] == "Hello, user!"
    assert result["tool_calls"] == []
    assert result["finish_reason"] == "stop"


def test_end_to_end_with_tools():
    """Simulate a chat completion with tool calls through AmlLlmModelRuntime."""
    from amlllm_openai_server.app import AmlLlmModelRuntime, ModelConfig

    config = ModelConfig(
        name="test-tool-model",
        model_path="/fake/weights.bin",
        model_type="none",
    )
    runtime = AmlLlmModelRuntime.__new__(AmlLlmModelRuntime)
    runtime.config = config
    runtime.lock = __import__("threading").Lock()
    runtime.client = FakeClient(
        '<tool_call>{"name":"get_weather","arguments":{"city":"Beijing"}}</tool_call>'
    )
    runtime.initialized = True
    runtime.current_signature = None
    runtime.queue = None
    runtime._chat_template_str = ""
    runtime._tokenizer_config = {}

    tool = _FakeChatCompletionTool(
        type_="function",
        function=_FakeChatCompletionToolFunction(
            name="get_weather",
            description="Get weather",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        ),
    )
    messages = [_FakeChatMessage(role="user", content="What's the weather in Beijing?")]
    result = runtime.run(messages=messages, user_data=None, tools=[tool])

    assert result["finish_reason"] == "tool_calls"
    assert len(result["tool_calls"]) == 1
    func = result["tool_calls"][0]["function"]
    assert func["name"] == "get_weather"
    assert json.loads(func["arguments"]) == {"city": "Beijing"}
    # Non-tool text should be cleaned
    assert result["text"] is None


def test_end_to_end_system_prompt_includes_tools():
    """Verify that tools are merged into the system prompt."""
    from amlllm_openai_server.app import AmlLlmModelRuntime, ModelConfig

    config = ModelConfig(
        name="test-model",
        model_path="/fake/weights.bin",
        model_type="none",
    )
    runtime = AmlLlmModelRuntime.__new__(AmlLlmModelRuntime)
    runtime.config = config
    runtime.lock = __import__("threading").Lock()
    runtime.client = FakeClient("OK")
    runtime.initialized = True
    runtime.current_signature = None
    runtime.queue = None
    runtime._chat_template_str = ""
    runtime._tokenizer_config = {}

    tool = _FakeChatCompletionTool(
        type_="function",
        function=_FakeChatCompletionToolFunction(name="search", description="Search the web"),
    )
    messages = [
        _FakeChatMessage(role="system", content="You are helpful."),
        _FakeChatMessage(role="user", content="search for cats"),
    ]
    runtime.run(messages=messages, user_data=None, tools=[tool])

    sp = runtime.client.system_prompt
    assert "You are helpful." in sp
    assert "search" in sp
    assert "<tool_call>" in sp


def test_end_to_end_no_tools_preserves_original_system_prompt():
    """When no tools are passed, the system prompt is unchanged."""
    from amlllm_openai_server.app import AmlLlmModelRuntime, ModelConfig

    config = ModelConfig(
        name="test-model",
        model_path="/fake/weights.bin",
        model_type="none",
        system_prompt="default system",
    )
    runtime = AmlLlmModelRuntime.__new__(AmlLlmModelRuntime)
    runtime.config = config
    runtime.lock = __import__("threading").Lock()
    runtime.client = FakeClient("Hi")
    runtime.initialized = True
    runtime.current_signature = None
    runtime.queue = None
    runtime._chat_template_str = ""
    runtime._tokenizer_config = {}

    messages = [_FakeChatMessage(role="user", content="hello")]
    runtime.run(messages=messages, user_data=None)

    sp = runtime.client.system_prompt
    assert "default system" in sp
    assert "<tool_call>" not in sp  # No tool instruction injected
    assert sp == "default system"


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import traceback

    tests = [
        # 1 build_tool_instruction
        test_build_tool_instruction_empty,
        test_build_tool_instruction_single_tool,
        test_build_tool_instruction_explicit_tool_choice,
        test_build_tool_instruction_multiple_tools,
        test_build_tool_instruction_missing_function,
        # 2 extract_tool_call_blocks
        test_extract_single_block,
        test_extract_multiple_blocks,
        test_extract_list_inside_block,
        test_extract_invalid_json,
        test_extract_no_blocks,
        test_extract_whitespace_insensitive,
        # 3 fallback_tool_calls_from_text
        test_fallback_basic,
        test_fallback_unknown_tool_filtered,
        test_fallback_empty_tool_list_accepts_all,
        test_fallback_function_sub_key,
        test_fallback_arguments_string_parsed,
        test_fallback_python_exec_fallback,
        test_fallback_single_tool_code_fallback,
        test_fallback_empty_text,
        test_fallback_dedup_tool_code,
        # 4 make_tool_call
        test_make_tool_call_shape,
        # 5 merge_system_prompt
        test_merge_no_tools_returns_base,
        test_merge_no_base,
        test_merge_with_base,
        # 6 text_without_tool_markup
        test_text_without_tool_markup_removes_tool_call_blocks,
        test_text_without_tool_markup_removes_tool_code_blocks,
        # 7 clean_template_artifacts
        test_clean_removes_think_tags,
        test_clean_removes_turn_tags,
        test_clean_returns_none_for_empty,
        # 8 messages_to_prompt
        test_messages_to_prompt_single_user,
        test_messages_to_prompt_conversation,
        test_messages_to_prompt_filters_system,
        # 9 system_prompt_from_messages
        test_system_prompt_extraction,
        test_system_prompt_empty,
        # 10 result formatting
        test_result_no_tools,
        test_result_with_tool_call,
        test_result_mixed_text_and_tool,
        test_result_streaming_shape,
        # 11 message_text
        test_message_text_string,
        test_message_text_none,
        test_message_text_parts,
        # 12 tool_names
        test_tool_names,
        # 13 normalize
        test_normalize_tools_converts,
        test_normalize_tools_none,
        test_normalize_tool_choice,
        # 14 end-to-end
        test_end_to_end_no_tools,
        test_end_to_end_with_tools,
        test_end_to_end_system_prompt_includes_tools,
        test_end_to_end_no_tools_preserves_original_system_prompt,
    ]

    passed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  ✓ {test.__name__}")
        except Exception:
            print(f"  ✗ {test.__name__}")
            traceback.print_exc()

    print(f"\n{passed}/{len(tests)} tests passed")
    sys.exit(0 if passed == len(tests) else 1)
