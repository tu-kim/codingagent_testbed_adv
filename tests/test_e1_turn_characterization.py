"""Tests for scripts/arm/e1_turn_characterization.py — the E0 views
adapted to concurrent (interleaved-session) runs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "arm" / "e1_turn_characterization.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("e1_turn_characterization", _SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["e1_turn_characterization"] = m
    spec.loader.exec_module(m)
    return m


class _T:
    def __init__(self, sid, step, start, end):
        self.session_id = sid
        self.step = step
        self.llm_start_ts = start
        self.llm_end_ts = end


def test_session_first_ordinals_no_overlap_filter(mod):
    # A and B OVERLAP in time (concurrent run); both must still yield a
    # boundary — E0's window filter would have dropped B as "nested".
    ordered = [_T("A", 1, 0.0, 1.0), _T("B", 1, 0.5, 1.5),
               _T("A", 2, 3.0, 4.0), _T("B", 2, 3.5, 4.5)]
    assert mod.session_first_ordinals(ordered) == [0, 1]
    assert mod.session_first_times_abs(ordered) == [0.0, 0.5]


def _turn(step, a, b, inp, out, cache, tool):
    evs = [{"ev": "llm.start", "ts": a, "step": step},
           {"ev": "llm.end", "ts": b, "step": step, "duration_s": b - a,
            "tokens": {"input": inp, "output": out, "cache": {"read": cache}},
            "dynamo": {}}]
    if tool:
        evs.append({"ev": "tool.end", "ts": b + 0.05, "step": step,
                    "name": tool, "callID": f"c{step}", "duration_s": 0.5,
                    "ok": True})
    return evs


def test_main_interleaved_sessions_no_figures(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    (prof / "A.jsonl").write_text("".join(
        json.dumps(e) + "\n" for e in
        _turn(1, 0, 1, 500, 50, 0, "read") + _turn(2, 3, 4, 600, 60, 500, None)))
    (prof / "B.jsonl").write_text("".join(
        json.dumps(e) + "\n" for e in
        _turn(1, 0.5, 1.5, 400, 40, 0, "bash") + _turn(2, 3.5, 4.5, 500, 50, 400, None)))
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"session_id": "A"}) + "\n"
                     + json.dumps({"session_id": "B"}) + "\n")
    out = tmp_path / "o"
    rc = mod.main(["--profiles", str(prof), "--trace", str(trace),
                   "--out", str(out), "--no-figures"])
    assert rc == 0
    assert not list(out.glob("*.pdf"))
    assert not list(out.glob("*.csv"))


class _TE:
    def __init__(self, sid, step, inp, cache, out, away=None, disp=None):
        self.session_id = sid
        self.step = step
        self.input_tokens = inp
        self.cache_read = cache
        self.output_tokens = out
        self.away_s = away
        self.away_displaced_tokens = disp
        self.llm_start_ts = float(step)
        self.llm_end_ts = float(step) + 0.5

    @property
    def effective_input(self):
        return None if self.input_tokens is None else self.input_tokens + self.cache_read


def test_eviction_events_discriminates_eviction_vs_compaction(mod):
    # step1: prev_cached = 1000 + 100 = 1100
    # step2: prompt GREW (eff 1600 > 1000) but cache_read 200 -> eviction
    # step3: prompt SHRANK (eff 300 < 0.6*1600) -> compaction
    turns = [_TE("A", 1, 1000, 0, 100),
             _TE("A", 2, 1400, 200, 80, away=5.0, disp=4000),
             _TE("A", 3, 300, 0, 50, away=2.0, disp=100)]
    ev = {e["step"]: e for e in mod.eviction_events(turns)}
    assert ev[2]["label"] == "eviction"
    assert ev[2]["shortfall"] == 1100 - 200
    assert ev[3]["label"] == "compaction"


def test_eviction_events_min_shortfall_noise_is_ok(mod):
    # a tiny shortfall (block granularity) is neither eviction nor compaction
    turns = [_TE("A", 1, 1000, 0, 0),
             _TE("A", 2, 1050, 950, 0)]     # prev_cached 1000, cr 950 -> short 50
    ev = mod.eviction_events(turns, min_shortfall=128)
    assert ev[0]["label"] == "ok"


def test_main_missing_trace_returns_2(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    rc = mod.main(["--profiles", str(prof), "--trace", str(tmp_path / "no")])
    assert rc == 2
