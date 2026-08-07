# -*- coding: utf-8 -*-
"""Unit tests for tools/metrics.py."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.schema import RunRecord  # noqa: E402
from tools.metrics import (  # noqa: E402
    compute_confusion_matrix, compute_recall, compute_precision,
    compute_accuracy, compute_f1, compile_stats, compile_all_stats,
)


def _r(mt="local", etn=None, ah=False, nm=False, hv=False, vp=None, err=None, lat=10.0):
    return RunRecord(
        entry_index=0, conv_index=0, repeat=0, model_type=mt,
        expected_tool_name=etn, has_verifier=hv,
        actual_has_tool_calls=ah,
        tool_names=[etn] if etn and ah else [],
        name_matched=nm, verifier_passed=vp,
        error=err, latency_ms=lat,
    )


class TestConfusionMatrix:
    def test_all_tp(self):
        tp, fp, fn, tn = compute_confusion_matrix([_r(etn="f", ah=True) for _ in range(5)])
        assert (tp, fp, fn, tn) == (5, 0, 0, 0)

    def test_all_tn(self):
        tp, fp, fn, tn = compute_confusion_matrix([_r(etn=None, ah=False) for _ in range(5)])
        assert (tp, fp, fn, tn) == (0, 0, 0, 5)

    def test_mixed(self):
        recs = [
            _r(etn="f", ah=True), _r(etn="f", ah=True),
            _r(etn="f", ah=False),
            _r(etn=None, ah=True),
            _r(etn=None, ah=False), _r(etn=None, ah=False),
        ]
        assert compute_confusion_matrix(recs) == (2, 1, 1, 2)


class TestFormulas:
    def test_recall(self):
        assert compute_recall(10, 0) == 1.0
        assert compute_recall(0, 10) == 0.0

    def test_precision(self):
        assert compute_precision(10, 0) == 1.0
        assert compute_precision(0, 10) == 0.0

    def test_accuracy(self):
        assert compute_accuracy(10, 10, 0, 0) == 1.0
        assert compute_accuracy(0, 0, 10, 10) == 0.0

    def test_f1(self):
        assert compute_f1(1.0, 1.0) == 1.0
        assert compute_f1(0.5, 0.5) == 0.5


class TestCompileStats:
    def test_name_match(self):
        recs = [
            _r("l", "f", True, nm=True),
            _r("l", "f", True, nm=True),
            _r("l", "f", True, nm=False),
            _r("l", None, False),
            _r("c", "f", True, nm=True),
        ]
        s = compile_stats(recs, "l")
        assert s.tp == 3
        assert s.name_match_count == 2
        assert s.name_mismatch_count == 1

    def test_verifier(self):
        recs = [
            _r("l", "f", True, nm=True, hv=True, vp=True),
            _r("l", "f", True, nm=True, hv=True, vp=True),
            _r("l", "f", True, nm=True, hv=True, vp=False),
            _r("l", "f", True, nm=True, hv=False, vp=None),
        ]
        s = compile_stats(recs, "l")
        assert s.verifier_total == 3
        assert s.verifier_pass_count == 2

    def test_all_stats(self):
        recs = [
            _r("l", "f", True, nm=True),
            _r("l", None, False),
            _r("c", "f", False),
            _r("c", None, False),
        ]
        all_s = compile_all_stats(recs)
        assert set(all_s.keys()) == {"l", "c"}
        assert all_s["l"].accuracy == 1.0
