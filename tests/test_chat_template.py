# -*- coding: utf-8 -*-
"""Tests for chat_template rendering with Jinja2 templates.

These tests verify the chat_template parsing and rendering logic
introduced for multi-template support in AmlLlmModelRuntime.

All tests are standalone and do NOT depend on the AMLLLM native backend.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from amlllm_openai_server.app import (
    _parse_tokenizer_config,
    _render_chat_template,
)


# ============================================================================
# 1.  _parse_tokenizer_config  —  parsing config.chat_format
# ============================================================================


def test_parse_empty():
    assert _parse_tokenizer_config("") == {}
    assert _parse_tokenizer_config(None) == {}


def test_parse_json_string_with_chat_template():
    cfg = json.dumps({
        "chat_template": "{% for m in messages %}{{ m['content'] }}{% endfor %}",
        "bos_token": "<s>",
        "eos_token": "</s>",
    })
    result = _parse_tokenizer_config(cfg)
    assert result["chat_template"] == "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    assert result["bos_token"] == "<s>"
    assert result["eos_token"] == "</s>"


def test_parse_file_path():
    cfg = {
        "chat_template": "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}",
        "bos_token": "<|begin_of_text|>",
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as f:
        json.dump(cfg, f)
        tmp_path = f.name

    try:
        result = _parse_tokenizer_config(tmp_path)
        assert result["chat_template"] == cfg["chat_template"]
        assert result["bos_token"] == "<|begin_of_text|>"
    finally:
        Path(tmp_path).unlink()


def test_parse_raw_jinja2_template():
    template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    result = _parse_tokenizer_config(template)
    assert result["chat_template"] == template


def test_parse_non_json_non_template():
    result = _parse_tokenizer_config("chatml")
    assert result == {}


def test_parse_invalid_json():
    result = _parse_tokenizer_config("{invalid json}")
    assert result == {}


# ============================================================================
# 2.  _render_chat_template  —  Jinja2 rendering
# ============================================================================


def test_render_simple_chatml_template():
    """Render a ChatML-style template with single user message."""
    template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    messages = [{"role": "user", "content": "Hello, who are you?"}]
    result = _render_chat_template(template, messages, add_generation_prompt=True)
    assert "<|im_start|>user\nHello, who are you?<|im_end|>\n" in result
    assert result.endswith("<|im_start|>assistant\n")


def test_render_chatml_multi_turn():
    """Render a multi-turn conversation with ChatML template."""
    template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "How are you?"},
    ]
    result = _render_chat_template(template, messages, add_generation_prompt=True)
    assert "<|im_start|>system\nYou are helpful.<|im_end|>\n" in result
    assert "<|im_start|>user\nHi<|im_end|>\n" in result
    assert "<|im_start|>assistant\nHello!<|im_end|>\n" in result
    assert "<|im_start|>user\nHow are you?<|im_end|>\n" in result
    assert result.endswith("<|im_start|>assistant\n")


def test_render_without_generation_prompt():
    """Render without the final assistant generation prompt."""
    template = (
        "{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}ASSISTANT:{% endif %}"
    )
    messages = [{"role": "user", "content": "Hi"}]
    result = _render_chat_template(template, messages, add_generation_prompt=False)
    assert result == "user: Hi\n"
    assert "ASSISTANT:" not in result


def test_render_with_bos_eos_tokens():
    """Template that uses bos_token and eos_token variables."""
    template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + eos_token + '\n' }}"
        "{% endfor %}"
    )
    messages = [{"role": "user", "content": "Hi"}]
    result = _render_chat_template(
        template, messages,
        bos_token="<s>", eos_token="</s>",
    )
    assert result == "<s>user: Hi</s>\n"


def test_render_llama3_style_template():
    """Render with a Llama 3 style template."""
    template = (
        "{{ bos_token }}"
        "{% set loop_messages = messages %}"
        "{% for message in loop_messages %}"
        "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
        "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}"
        "{{ content }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    result = _render_chat_template(
        template, messages,
        bos_token="<|begin_of_text|>",
        add_generation_prompt=True,
    )
    assert "<|start_header_id|>system<|end_header_id|>" in result
    assert "<|start_header_id|>user<|end_header_id|>" in result
    assert "You are a helpful assistant." in result
    assert "Hello!" in result
    assert result.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")


def test_render_with_tools_variable():
    """Template that has access to the tools variable."""
    template = (
        "{% if tools %}"
        "Available tools:\n"
        "{% for tool in tools %}"
        "- {{ tool.function.name }}\n"
        "{% endfor %}"
        "{% endif %}"
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
    )
    messages = [{"role": "user", "content": "Search for cats"}]
    tools = [
        {"type": "function", "function": {"name": "search", "description": "Search the web"}},
        {"type": "function", "function": {"name": "calculate", "description": "Do math"}},
    ]
    result = _render_chat_template(template, messages, tools=tools)
    assert "Available tools:" in result
    assert "- search" in result
    assert "- calculate" in result
    assert "user: Search for cats" in result


def test_render_gemma_style_template():
    """Render with a Gemma-style template."""
    template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
        "<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
        "{% elif message['role'] == 'assistant' %}"
        "<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    )
    messages = [
        {"role": "user", "content": "Tell me a joke"},
    ]
    result = _render_chat_template(
        template, messages,
        bos_token="<bos>",
        add_generation_prompt=True,
    )
    assert result.startswith("<bos>")
    assert "<start_of_turn>user\nTell me a joke<end_of_turn>\n" in result
    assert result.endswith("<start_of_turn>model\n")


def test_render_empty_messages():
    """Rendering with no messages should still handle generation prompt."""
    template = (
        "{% for message in messages %}"
        "{{ message['content'] }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}READY{% endif %}"
    )
    result = _render_chat_template(template, [], add_generation_prompt=True)
    assert result == "READY"


def test_render_tool_call_messages():
    """Render messages that include tool call and tool result."""
    template = (
        "{% for message in messages %}"
        "{{ '<|im_start|>' + message['role'] + '\n' }}"
        "{% if message['role'] == 'tool' %}"
        "{{ 'tool_call_id: ' + message.get('tool_call_id', '') + '\n' }}"
        "{% endif %}"
        "{{ message['content'] + '<|im_end|>' + '\n' }}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": '{"city":"Beijing"}'}}
        ]},
        {"role": "tool", "content": "Sunny, 25°C", "tool_call_id": "call_123"},
    ]
    result = _render_chat_template(template, messages, add_generation_prompt=True)
    assert "<|im_start|>user\nWhat's the weather?<|im_end|>\n" in result
    assert "<|im_start|>assistant\n<|im_end|>\n" in result
    assert "<|im_start|>tool\ntool_call_id: call_123\nSunny, 25°C<|im_end|>\n" in result
    assert result.endswith("<|im_start|>assistant\n")


# ============================================================================
# 3.  Integration-style tests  —  full pipeline
# ============================================================================


def test_full_pipeline_chatml():
    """Full pipeline: parse config, render multi-turn conversation with tools."""
    # Simulate what AmlLlmModelRuntime would do
    tokenizer_config = {
        "chat_template": (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        ),
        "bos_token": "",
        "eos_token": "<|im_end|>",
    }
    chat_format_str = json.dumps(tokenizer_config)

    # Step 1: Parse
    parsed = _parse_tokenizer_config(chat_format_str)
    assert parsed["chat_template"] == tokenizer_config["chat_template"]

    # Step 2: Render with tools
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": "Search for Python tutorials"},
    ]
    tools = [
        {"type": "function", "function": {"name": "web_search", "description": "Search the internet"}},
    ]
    prompt = _render_chat_template(
        parsed["chat_template"],
        messages=messages,
        tools=tools,
        add_generation_prompt=True,
        bos_token=parsed.get("bos_token", ""),
        eos_token=parsed.get("eos_token", ""),
    )
    assert "<|im_start|>system\nYou are a helpful assistant with tools.<|im_end|>\n" in prompt
    assert "<|im_start|>user\nSearch for Python tutorials<|im_end|>\n" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_full_pipeline_multi_turn_with_tool_results():
    """Full pipeline: multi-turn conversation including tool call results."""
    tokenizer_config = {
        "chat_template": (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|system|>{{ message['content'] }}"
            "{% elif message['role'] == 'user' %}"
            "<|user|>{{ message['content'] }}"
            "{% elif message['role'] == 'assistant' %}"
            "<|assistant|>"
            "{% if message.get('tool_calls') %}"
            "{% for tc in message['tool_calls'] %}"
            "<tool_call>{{ tc.function.name }}({{ tc.function.arguments }})</tool_call>"
            "{% endfor %}"
            "{% else %}"
            "{{ message['content'] }}"
            "{% endif %}"
            "{% elif message['role'] == 'tool' %}"
            "<|tool|>{{ message['content'] }}"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|assistant|>{% endif %}"
        ),
    }
    chat_format_str = json.dumps(tokenizer_config)

    # Parse
    parsed = _parse_tokenizer_config(chat_format_str)

    # Multi-turn with tool calls
    messages = [
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "calculate", "arguments": '{"expr":"2+2"}'}}
        ]},
        {"role": "tool", "content": "4", "tool_call_id": "call_1"},
    ]
    prompt = _render_chat_template(
        parsed["chat_template"],
        messages=messages,
        add_generation_prompt=True,
    )
    assert "<|user|>What's 2+2?" in prompt
    assert "<tool_call>calculate({\"expr\":\"2+2\"})</tool_call>" in prompt
    assert "<|tool|>4" in prompt
    assert prompt.endswith("<|assistant|>")


# ============================================================================
# 4.  Edge cases
# ============================================================================


def test_template_with_special_chars_in_content():
    """Messages containing quotes, angle brackets, etc."""
    template = (
        "{% for message in messages %}"
        "{{ message['role'] }}: {{ message['content'] }}\n"
        "{% endfor %}"
    )
    messages = [
        {"role": "user", "content": 'Say "<hello>" & \'world\''},
    ]
    result = _render_chat_template(template, messages)
    assert 'Say "<hello>" & \'world\'' in result


def test_template_with_unicode():
    """Messages containing Unicode characters."""
    template = (
        "{% for message in messages %}"
        "{{ message['content'] }}"
        "{% endfor %}"
    )
    messages = [
        {"role": "user", "content": "你好世界 🌍"},
    ]
    result = _render_chat_template(template, messages)
    assert "你好世界 🌍" in result


def test_template_no_messages_key():
    """Template that doesn't use messages at all (degenerate case)."""
    template = "Fixed prompt"
    result = _render_chat_template(template, [])
    assert result == "Fixed prompt"


def test_render_with_message_name_and_tool_call_id():
    """Messages with name and tool_call_id fields."""
    template = (
        "{% for message in messages %}"
        "{{ message['role'] }}"
        "{% if message.get('name') %}({{ message['name'] }}){% endif %}"
        "{% if message.get('tool_call_id') %}[{{ message['tool_call_id'] }}]{% endif %}"
        ": {{ message['content'] }}\n"
        "{% endfor %}"
    )
    messages = [
        {"role": "tool", "content": "result", "tool_call_id": "call_abc", "name": "my_tool"},
    ]
    result = _render_chat_template(template, messages)
    assert "tool(my_tool)[call_abc]: result" in result
