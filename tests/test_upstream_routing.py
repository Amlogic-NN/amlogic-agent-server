# -*- coding: utf-8 -*-
"""Tests for upstream routing: estimate_token_count and inject_skills_for_forwarding."""

from __future__ import annotations

import json
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from proxy_server.types import ChatMessage
from proxy_server.llm_utils import estimate_token_count, inject_skills_for_forwarding


# ============================================================================
# estimate_token_count
# ============================================================================


def test_estimate_empty_messages():
    assert estimate_token_count([]) == 1  # max(1, 0 // 4)


def test_estimate_single_text_message():
    msgs = [ChatMessage(role="user", content="Hello, world!")]
    # "Hello, world!" = 13 chars + "user" = 4 chars = 17 // 4 = 4
    assert estimate_token_count(msgs) == 4


def test_estimate_multiple_messages():
    msgs = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="What is the capital of France?"),
    ]
    # "You are a helpful assistant." = 28, "What is the capital of France?" = 30
    # "system" = 6, "user" = 4
    # total = 68 // 4 = 17
    assert estimate_token_count(msgs) == 17


def test_estimate_with_tool_calls():
    msgs = [
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query":"test"}'},
                }
            ],
        ),
    ]
    estimated = estimate_token_count(msgs)
    assert estimated > 0


def test_estimate_null_content():
    msgs = [ChatMessage(role="user", content=None)]
    # "user" = 4 chars // 4 = 1
    assert estimate_token_count(msgs) == 1


# ============================================================================
# inject_skills_for_forwarding
# ============================================================================


def test_inject_no_skills_returns_unchanged():
    msgs = [
        ChatMessage(role="system", content="You are a helpful assistant."),
        ChatMessage(role="user", content="Hello"),
    ]
    result = inject_skills_for_forwarding(msgs)
    assert len(result) == 2
    assert result[0].content == "You are a helpful assistant."
    assert result[1].content == "Hello"


def test_inject_no_system_message_returns_unchanged():
    msgs = [ChatMessage(role="user", content="Hello")]
    result = inject_skills_for_forwarding(msgs)
    assert len(result) == 1
    assert result[0].content == "Hello"


def test_inject_empty_messages():
    result = inject_skills_for_forwarding([])
    assert result == []


def test_inject_skills_with_real_files():
    """Create temporary skill files and verify injection works end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create skill files
        skill1_dir = os.path.join(tmpdir, "skill1")
        os.makedirs(skill1_dir)
        skill1_path = os.path.join(skill1_dir, "SKILL.md")
        with open(skill1_path, "w") as f:
            f.write("---\nname: skill1\ndescription: First skill\n---\n\n# Skill 1 Content\n\nThis is skill one content from {{SKILL_DIR}}.")

        skill2_dir = os.path.join(tmpdir, "skill2")
        os.makedirs(skill2_dir)
        skill2_path = os.path.join(skill2_dir, "SKILL.md")
        with open(skill2_path, "w") as f:
            f.write("---\nname: skill2\n---\n\n# Skill 2 Content\n\nThis is skill two.")

        # Build messages with skill references
        skill_template = os.path.join(tmpdir, "{skill_name}", "SKILL.md")
        system_prompt = (
            "You are a helpful assistant.\n\n"
            "# Skills\n"
            "<skills>\n"
            "<skill>\n"
            "<name>skill1</name>\n"
            "<description>First skill description</description>\n"
            "</skill>\n"
            "<skill>\n"
            "<name>skill2</name>\n"
            "<description>Second skill description</description>\n"
            "</skill>\n"
            "</skills>\n\n"
            "## Workspace\n"
            f"- Skills: {skill_template}\n"
        )

        msgs = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content="Use the skills"),
        ]

        result = inject_skills_for_forwarding(msgs)

        # Should have 2 messages: system + user
        assert len(result) == 2
        assert result[0].role == "system"
        assert result[1].role == "user"
        assert result[1].content == "Use the skills"

        # The injected system prompt should contain the actual skill content
        injected_system = result[0].content
        assert "Skill 1 Content" in injected_system
        assert "this is skill one content from" in injected_system.lower()
        # {{SKILL_DIR}} should be replaced with the actual skill directory
        assert skill1_dir in injected_system
        assert "Skill 2 Content" in injected_system
        # Should NOT contain the XML skill blocks
        assert "<skill>" not in injected_system
        assert "<name>" not in injected_system


def test_inject_skills_without_path_template():
    """When system prompt has skills but no workspace path template, return unchanged."""
    system_prompt = (
        "You are a helpful assistant.\n\n"
        "# Skills\n"
        "<skills>\n"
        "<skill>\n"
        "<name>nonexistent</name>\n"
        "<description>Some skill</description>\n"
        "</skill>\n"
        "</skills>\n"
    )
    msgs = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content="Hello"),
    ]
    result = inject_skills_for_forwarding(msgs)
    # Should be unchanged since there's no path template
    assert len(result) == 2
    assert result[0].content == system_prompt


def test_inject_skills_missing_file_graceful():
    """When a skill file doesn't exist, it should be silently skipped."""
    with tempfile.TemporaryDirectory() as tmpdir:
        skill_template = os.path.join(tmpdir, "{skill_name}", "SKILL.md")
        system_prompt = (
            "# Skills\n"
            "<skills>\n"
            "<skill>\n"
            "<name>missing_skill</name>\n"
            "<description>A skill that doesn't exist</description>\n"
            "</skill>\n"
            "</skills>\n\n"
            "## Workspace\n"
            f"- Skills: {skill_template}\n"
        )
        msgs = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content="Hello"),
        ]
        result = inject_skills_for_forwarding(msgs)
        # Should still return valid messages (skill skipped gracefully)
        assert len(result) == 2
        assert result[1].content == "Hello"
