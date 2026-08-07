# -*- coding: utf-8 -*-
"""Unit tests for tools/schema.py."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.schema import (  # noqa: E402
    TestCaseEntry, ToolExpectEntry, VerifierSpec, CloudApiConfig,
    RunRecord, StatsResult, load_verifier,
)


class TestToolExpectEntry:
    def test_with_name(self):
        e = ToolExpectEntry.model_validate({"name": "web_fetch"})
        assert e.expects_tool_call is True
        assert e.has_verifier is False

    def test_null_name(self):
        e = ToolExpectEntry.model_validate({"name": None})
        assert e.expects_tool_call is False

    def test_with_verifier(self):
        e = ToolExpectEntry.model_validate({
            "name": "web_fetch",
            "verifier": {"path": "test_data/web_fetch_tests.py", "function": "verify_web_fetch_json_ex1"},
        })
        assert e.has_verifier is True
        assert e.verifier.path == "test_data/web_fetch_tests.py"


class TestTestCaseEntry:
    def test_valid(self):
        entry = TestCaseEntry.model_validate({
            "messages": [[{"role":"user","content":"hello"}],[{"role":"user","content":"tool"}]],
            "tool_expect": [{"name": None}, {"name": "get_weather"}],
        })
        assert len(entry.messages) == 2

    def test_mismatched_raises(self):
        with pytest.raises(ValueError, match="len\\(messages\\)"):
            TestCaseEntry.model_validate({
                "messages": [[{"role":"user","content":"hello"}]],
                "tool_expect": [{"name":"a"},{"name":None}],
            })


class TestVerifierSpec:
    def test_load_real(self):
        root = Path(__file__).resolve().parents[1]
        spec = VerifierSpec(path="test_data/web_fetch_tests.py", function="verify_web_fetch_json_ex1")
        fn = load_verifier(spec, root)
        assert fn is not None
        # Now returns (bool, str|None)
        result = fn('{"url":"https://api.example.com/users"}')
        assert result == (True, None)
        result = fn('{"url":"https://other.com"}')
        assert result[0] is False
        assert result[1] is not None  # failure reason string

    def test_missing_file(self):
        assert load_verifier(VerifierSpec(path="/nonexistent.py", function="foo")) is None

    def test_missing_function(self):
        root = Path(__file__).resolve().parents[1]
        assert load_verifier(VerifierSpec(path="test_data/web_fetch_tests.py", function="no_such_fn"), root) is None


class TestCloudApiConfig:
    def test_from_yaml(self):
        yaml = "remote-api:\n  base_url: http://x:8001/\n  api_key: k\n  model: m\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml); f.flush()
            cfg = CloudApiConfig.from_yaml(f.name)
        Path(f.name).unlink(missing_ok=True)
        assert cfg.base_url == "http://x:8001"
        assert cfg.api_key == "k"
        assert cfg.model == "m"


class TestRunRecord:
    def test_basic(self):
        r = RunRecord(entry_index=0, conv_index=1, repeat=2, model_type="local",
                      expected_tool_name="f", has_verifier=False,
                      actual_has_tool_calls=True, tool_call_count=1, tool_names=["f"],
                      name_matched=True, verifier_passed=None,
                      finish_reason="tool_calls", latency_ms=123.4, token_count=50,
                      timestamp="ts")
        d = r.model_dump()
        assert d["expected_tool_name"] == "f"
        assert d["name_matched"] is True
        assert d["verifier_passed"] is None


class TestStatsResult:
    def test_perfect(self):
        s = StatsResult(model_type="local", tp=10, tn=10, total_runs=20,
                        name_match_count=10, name_mismatch_count=0)
        assert s.recall == 1.0
        assert s.name_accuracy == 1.0

    def test_verifier(self):
        s = StatsResult(model_type="local", tp=5, tn=5, total_runs=10,
                        name_match_count=5, name_mismatch_count=0,
                        verifier_total=5, verifier_pass_count=4)
        assert s.verifier_pass_rate == 0.8

    def test_to_dict(self):
        s = StatsResult(model_type="local", tp=5, fp=2, fn=3, tn=10, total_runs=20,
                        name_match_count=4, name_mismatch_count=1,
                        verifier_total=3, verifier_pass_count=2,
                        error_count=1, avg_latency_ms=42.5)
        d = s.to_dict()
        assert d["TP"] == 5
        assert d["NameMatch"] == 4
        assert d["VerifierPassRate"] == round(2/3, 4)
