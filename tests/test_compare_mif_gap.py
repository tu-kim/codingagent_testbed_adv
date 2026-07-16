"""Tests for scripts/arm/compare_mif_gap.py — cross-mif turn-gap
decomposition (tool vs others). No network/GPU; synthetic profiles."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "arm" / "compare_mif_gap.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("compare_mif_gap", _SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["compare_mif_gap"] = m
    spec.loader.exec_module(m)
    return m


def _mkrun(d, tool_dur, gap):
    prof = d / "profiles"; prof.mkdir()
    evs = [
        {"ev": "llm.start", "ts": 0, "step": 1},
        {"ev": "llm.end", "ts": 1, "step": 1, "duration_s": 1,
         "tokens": {"input": 100, "output": 10, "cache": {"read": 0}},
         "dynamo": {}},
        {"ev": "tool.end", "ts": 1.1, "step": 1, "name": "bash",
         "callID": "c1", "duration_s": tool_dur, "ok": True},
        {"ev": "turn.end", "ts": 1.2, "step": 1, "duration_s": 2,
         "llm_wall_s": 1, "tool_wall_s": tool_dur, "llm_wall_true_s": 1},
        {"ev": "llm.start", "ts": 1 + gap, "step": 2},
        {"ev": "llm.end", "ts": 2 + gap, "step": 2, "duration_s": 1,
         "tokens": {"input": 50, "output": 10, "cache": {"read": 100}},
         "dynamo": {}},
    ]
    (prof / "A.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs))
    (d / "trace.jsonl").write_text(json.dumps({"session_id": "A"}) + "\n")
    return prof, d / "trace.jsonl"


def test_gap_full_and_tool_without_logs(mod, tmp_path):
    prof, trace = _mkrun(tmp_path, tool_dur=0.5, gap=2.0)
    ats = mod._load("analyze_turn_scheduling", mod._ATS_PATH)
    e0 = mod._load("e0_turn_characterization", mod._E0_PATH)
    rows = mod.gap_rows(prof, trace, e0, ats)
    assert len(rows) == 1
    r = rows[0]
    assert r["gap_full"] == pytest.approx(2.0)
    assert r["tool"] == pytest.approx(0.5)       # prev step's tool wall
    # no --logs / --frontend -> server components unresolved
    assert r["queue_wait"] is None
    assert r["scaffold"] is None
    assert r["turn_gap"] is None


def test_tool_durations_per_name(mod, tmp_path):
    prof, trace = _mkrun(tmp_path, tool_dur=0.7, gap=2.0)
    e0 = mod._load("e0_turn_characterization", mod._E0_PATH)
    td = mod.tool_durations(prof, e0.trace_session_ids(trace))
    assert td == {"bash": [0.7]}


def test_main_writes_summary(mod, tmp_path):
    d1 = tmp_path / "r1"; d1.mkdir()
    p1, t1 = _mkrun(d1, tool_dur=0.5, gap=2.0)
    d4 = tmp_path / "r4"; d4.mkdir()
    p4, t4 = _mkrun(d4, tool_dur=1.5, gap=5.0)
    out = tmp_path / "o"
    rc = mod.main(["--run", "mif1", str(p1), str(t1),
                   "--run", "mif4", str(p4), str(t4),
                   "--out", str(out), "--no-figures"])
    assert rc == 0
    rows = list(csv.DictReader((out / "mif_gap_summary.csv").open()))
    got = {(r["mif"], r["component"]): float(r["p50"]) for r in rows}
    assert got[("mif1", "tool")] == pytest.approx(0.5)
    assert got[("mif4", "tool")] == pytest.approx(1.5)      # tool inflated
    assert got[("mif4", "gap_full")] == pytest.approx(5.0)


def test_canonical_turn_gap_excludes_prefill(mod, tmp_path):
    # gap 62: tool 0.5; SCHED queue 49; ttft 60 -> prefill 11; scaffold 1.5;
    # turn_gap = tool+scaffold+queue = 0.5+1.5+49 = 51 = gap_full - prefill
    prof = tmp_path / "profiles"; prof.mkdir()
    base = 1782350000.0
    evs = [
        {"ev": "llm.start", "ts": base, "step": 1},
        {"ev": "llm.end", "ts": base + 1, "step": 1, "duration_s": 1,
         "tokens": {"input": 100, "output": 10, "cache": {"read": 0}},
         "dynamo": {}},
        {"ev": "tool.end", "ts": base + 1.1, "step": 1, "name": "bash",
         "callID": "c1", "duration_s": 0.5, "ok": True},
        {"ev": "turn.end", "ts": base + 1.2, "step": 1, "duration_s": 2,
         "llm_wall_s": 1, "tool_wall_s": 0.5, "llm_wall_true_s": 1},
        {"ev": "llm.start", "ts": base + 63.0, "step": 2},
        {"ev": "llm.end", "ts": base + 64.0, "step": 2, "duration_s": 1,
         "request_id": "rid-2",
         "tokens": {"input": 50, "output": 10, "cache": {"read": 100}},
         "dynamo": {}},
    ]
    (prof / "A.jsonl").write_text("".join(json.dumps(e) + "\n" for e in evs))
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"session_id": "A"}) + "\n")
    logs = tmp_path / "logs"; logs.mkdir()
    (logs / "vllm-a0.log").write_text(
        "SCHED_DELAY request_id=rid-2 role=prefill queue_ms=49000.0 "
        "queued_ts=150.0 scheduled_ts=199.0\n")
    fe = tmp_path / "frontend.log"
    fe.write_text("request completed request_id=rid-2 ttft_ms=60000\n")
    ats = mod._load("analyze_turn_scheduling", mod._ATS_PATH)
    e0 = mod._load("e0_turn_characterization", mod._E0_PATH)
    r = mod.gap_rows(prof, trace, e0, ats, logs=logs, frontend=fe)[0]
    assert r["queue_wait"] == pytest.approx(49.0)
    assert r["prefill"] == pytest.approx(11.0)
    assert r["scaffold"] == pytest.approx(1.5)      # 62 - 0.5 - 60
    assert r["turn_gap"] == pytest.approx(51.0)     # tool+scaffold+queue
    assert r["turn_gap"] == pytest.approx(
        r["tool"] + r["scaffold"] + r["queue_wait"])


def test_main_missing_profiles_returns_2(mod, tmp_path):
    trace = tmp_path / "t.jsonl"; trace.write_text("{}\n")
    rc = mod.main(["--run", "x", str(tmp_path / "nope"), str(trace),
                   "--no-figures"])
    assert rc == 2
