# -*- coding: utf-8 -*-
"""Unit tests for tools/evaluator.py — integration logic (mocked runtime)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.schema import TestCaseEntry
from tools.evaluator import Evaluator, create_local_runner_from_yaml
from tools.runners import LocalModelRunner


def _mock_result(tool_calls=False, name="", finish_reason="stop", error=None, arguments="{}"):
    return {
        "text": "" if tool_calls else "Hello.",
        "tool_calls": [{"id": "c1", "function": {"name": name, "arguments": arguments}}] if tool_calls else [],
        "finish_reason": "tool_calls" if tool_calls else finish_reason,
        "token_count": 10,
        "error": error,
        "latency_ms": 100.0,
    }


def _mock_runner(**kwargs):
    """Create a MagicMock LocalModelRunner with model_name set."""
    runner = MagicMock(spec=LocalModelRunner)
    runner.model_name = kwargs.pop("model_name", "local_mock")
    for k, v in kwargs.items():
        setattr(runner, k, v)
    return runner


class TestEvaluator:
    def test_writes_jsonl(self):
        tc = [TestCaseEntry.model_validate({
            "messages": [[{"role":"user","content":"weather"}],[{"role":"user","content":"hello"}]],
            "tool_expect": [{"name":"get_weather"},{"name":None}],
            "max_tokens": 128,
        })]
        runner = _mock_runner(
            run_single=MagicMock(side_effect=[
                _mock_result(tool_calls=True, name="get_weather"),
                _mock_result(tool_calls=False),
            ]),
        )
        with tempfile.TemporaryDirectory() as d:
            jp = Path(d) / "r.jsonl"
            Evaluator(local_runner=runner, cloud_runner=None, n_repeat=1).evaluate(tc, jp)
            lines = jp.read_text("utf-8").strip().split("\n")
            assert len(lines) == 2
            r0 = json.loads(lines[0])
            assert r0["model_type"] == "local_mock"
            assert r0["name_matched"] is True
            r1 = json.loads(lines[1])
            assert r1["expected_tool_name"] is None

    def test_name_mismatch(self):
        tc = [TestCaseEntry.model_validate({
            "messages": [[{"role":"user","content":"fetch"}]],
            "tool_expect": [{"name":"web_fetch"}],
        })]
        runner = _mock_runner(
            run_single=MagicMock(return_value=_mock_result(tool_calls=True, name="get_weather")),
        )
        with tempfile.TemporaryDirectory() as d:
            jp = Path(d) / "r.jsonl"
            Evaluator(local_runner=runner, cloud_runner=None, n_repeat=1).evaluate(tc, jp)
            r = json.loads(jp.read_text("utf-8").strip())
            assert r["actual_has_tool_calls"] is True
            assert r["name_matched"] is False

    def test_verifier(self):
        tc = [TestCaseEntry.model_validate({
            "messages": [[{"role":"user","content":"fetch"}]],
            "tool_expect": [{
                "name":"web_fetch",
                "verifier":{"path":"test_data/web_fetch_tests.py","function":"verify_web_fetch_json_ex1"},
            }],
        })]
        runner = _mock_runner(
            run_single=MagicMock(return_value=_mock_result(
                tool_calls=True, name="web_fetch",
                arguments='{"url":"https://other.com"}',
            )),
        )
        with tempfile.TemporaryDirectory() as d:
            jp = Path(d) / "r.jsonl"
            Evaluator(local_runner=runner, cloud_runner=None, n_repeat=1).evaluate(tc, jp)
            r = json.loads(jp.read_text("utf-8").strip())
            assert r["name_matched"] is True
            assert r["verifier_passed"] is False
            assert r["verifier_reason"] is not None

    def test_handles_errors(self):
        tc = [TestCaseEntry.model_validate({
            "messages": [[{"role":"user","content":"test"}]],
            "tool_expect": [{"name":"get_weather"}],
        })]
        runner = _mock_runner(
            run_single=MagicMock(return_value=_mock_result(error="RuntimeError: crash")),
        )
        with tempfile.TemporaryDirectory() as d:
            jp = Path(d) / "r.jsonl"
            Evaluator(local_runner=runner, cloud_runner=None, n_repeat=1).evaluate(tc, jp)
            r = json.loads(jp.read_text("utf-8").strip())
            assert "RuntimeError" in (r["error"] or "")

    def test_compute_stats(self):
        tc = [TestCaseEntry.model_validate({
            "messages": [[{"role":"user","content":"call"}],[{"role":"user","content":"no"}]],
            "tool_expect": [{"name":"get_weather"},{"name":None}],
        })]
        runner = _mock_runner(
            run_single=MagicMock(side_effect=[
                _mock_result(tool_calls=True, name="get_weather"),
                _mock_result(tool_calls=False),
            ]),
        )
        with tempfile.TemporaryDirectory() as d:
            jp = Path(d) / "r.jsonl"
            ev = Evaluator(local_runner=runner, cloud_runner=None, n_repeat=1)
            ev.evaluate(tc, jp)
            s = ev.compute_stats_from_jsonl(jp)["local_mock"]
            assert s.tp == 1
            assert s.tn == 1
            assert s.name_match_count == 1


def test_create_local_runner_skips_without_config():
    project_dir = Path(__file__).resolve().parents[1]
    config_path = project_dir / "config" / "server.yaml"
    if not config_path.exists():
        pytest.skip(f"No config found at {config_path}")
    runner = create_local_runner_from_yaml(str(config_path), model_index=0)
    assert runner is not None
    assert runner.model_config.name
