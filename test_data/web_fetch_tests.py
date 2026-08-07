# -*- coding: utf-8 -*-
"""Verifier functions for tool-call argument validation.

Each verifier function must have the signature:
    (json_content: str, **kwargs) -> tuple[bool, str | None]

- The bool indicates pass (True) or fail (False).
- The optional string describes the failure reason (only meaningful on fail).
"""

from __future__ import annotations

import json
import os
from typing import Optional
from urllib.parse import urlparse


def verify_web_fetch_json_ex1(json_content: str, **kwargs) -> tuple[bool, Optional[str]]:
    """Verify that the web_fetch call targets https://api.example.com/users."""
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        return False, "Arguments are not valid JSON"

    if not isinstance(data, dict):
        return False, "Arguments must be a JSON object"

    if "url" not in data:
        return False, "Missing required key: 'url'"

    if not isinstance(data["url"], str):
        return False, "Key 'url' must be a string"

    if data["url"] != "https://api.example.com/users":
        return False, f"Expected url='https://api.example.com/users', got '{data['url']}'"

    return True, None


def verify_write_file_text_ex1(json_content: str, **kwargs) -> tuple[bool, Optional[str]]:
    """Verify that a write_file call writes the expected content to the expected file."""
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        return False, "Arguments are not valid JSON"

    if not isinstance(data, dict):
        return False, "Arguments must be a JSON object"

    if "path" not in data:
        return False, "Missing required key: 'path'"
    if "content" not in data:
        return False, "Missing required key: 'content'"

    if not isinstance(data["path"], str):
        return False, "Key 'path' must be a string"
    if not isinstance(data["content"], str):
        return False, "Key 'content' must be a string"

    filename = kwargs.get("filename", "greeting.txt")
    if os.path.basename(data["path"]) != filename:
        return False, f"Expected filename '{filename}', got '{os.path.basename(data['path'])}'"

    expected_content = kwargs.get("content", "Hello from agent")
    if data["content"] != expected_content:
        return False, f"Expected content '{expected_content}', got '{data['content']}'"

    return True, None


def verify_web_fetch_weather_ex1(json_content: str, **kwargs) -> tuple[bool, Optional[str]]:
    """Verify that a web_fetch call targets wttr.in with the expected location and query."""
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        return False, "Arguments are not valid JSON"

    if not isinstance(data, dict):
        return False, "Arguments must be a JSON object"

    if "url" not in data:
        return False, "Missing required key: 'url'"

    if not isinstance(data["url"], str):
        return False, "Key 'url' must be a string"

    parsed_url = urlparse(data["url"])

    if parsed_url.netloc != "wttr.in":
        return False, f"Expected host 'wttr.in', got '{parsed_url.netloc}'"

    expected_location = kwargs.get("location", "Tokyo")
    expected_path = f"/{expected_location}"
    if parsed_url.path != expected_path:
        return False, f"Expected path '{expected_path}', got '{parsed_url.path}'"

    expected_query = kwargs.get("query")
    if expected_query is not None and parsed_url.query != expected_query:
        return False, f"Expected query '{expected_query}', got '{parsed_url.query}'"

    return True, None


def verify_script_exec(json_content: str, **kwargs) -> tuple[bool, Optional[str]]:
    """Verify that a web_fetch call targets wttr.in with the expected location and query."""
    try:
        data = json.loads(json_content)
    except json.JSONDecodeError:
        return False, "Arguments are not valid JSON"

    if not isinstance(data, dict):
        return False, "Arguments must be a JSON object"

    if "action" not in data or 'command' not in data:
        return False, "Missing required key: 'action' or 'command'"

    if not isinstance(data["action"], str) or not isinstance(data["command"], str):
        return False, "Key 'action' and 'command' must be a string"

    # Pre-filter: extract the actual command from bash -c wrappers.
    # Supported forms:
    #   /bin/bash -c 'program args ...'
    #   bash -c "program args ..."
    #   program args ...
    import re
    command_str = data["command"].strip()
    match = re.match(r"^(?:/bin/)?bash\s+-c\s+(['\''\"])(.+?)\1\s*$", command_str)
    if match:
        command_str = match.group(2).strip()
        data["command"] = command_str

    action = data["action"]
    target_action = kwargs.get("action", "run")
    if action != target_action:
        return False, f"Expected action '{target_action}', got '{action}'"

    command = data["command"].split(' ')
    miss_program = kwargs.get("program", False)
    expected_program = kwargs.get("program", "")
    expected_args = kwargs.get("args", [])
    if not miss_program:
        program = command[0]
        args = [a.strip().strip('"').strip("'") for a in  command[1:]]
        if program != expected_program:
            return False, f"Expected program '{expected_program}', got '{program}'"
    else:
        program = command[0]
        if program != expected_program:
            args = [a.strip().strip('"').strip("'") for a in  command]
        else:
            args = [a.strip().strip('"').strip("'") for a in  command[1:]]

    # Element-wise matching: expected_args[i] can be:
    #   str        -> exact match required
    #   None       -> wildcard (any value allowed)
    #   list[str|None] -> any value in the list is accepted
    if not kwargs.get('ignore_extra', False):
        def _arg_matches(actual: str, expected) -> bool:
            if expected is None:
                return True
            if isinstance(expected, list):
                return any(_arg_matches(actual, e) for e in expected)
            return actual == expected

        if len(args) != len(expected_args):
            return False, f"Expected {len(expected_args)} args, got {len(args)}: args={args}, expected_args={expected_args}"
        for i, (actual, expected) in enumerate(zip(args, expected_args)):
            if not _arg_matches(actual, expected):
                return False, f"Arg[{i}]: expected one of {expected!r}, got '{actual}'"

    return True, None
