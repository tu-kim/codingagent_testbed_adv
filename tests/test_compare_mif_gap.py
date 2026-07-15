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


def test_gap_decomposes_into_tool_and_others(mod, tmp_path):
    prof, trace = _mkrun(tmp_path, tool_dur=0.5, gap=2.0)
    ats = mod._load("analyze_turn_scheduling", mod._ATS_PATH)
    e0 = mod._load("e0_turn_characterization", mod._E0_PATH)
    rows = mod.gap_rows(prof, trace, e0, ats)
    assert len(rows) == 1
    r = rows[0]
    assert r["gap"] == pytest.approx(2.0)
    assert r["tool"] == pytest.approx(0.5)       # prev step's tool wall
    assert r["others"] == pytest.approx(1.5)     # gap - tool


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
    assert got[("mif4", "tool")] == pytest.approx(1.5)   # tool inflated
    assert got[("mif4", "others")] == pytest.approx(3.5)


def test_main_missing_profiles_returns_2(mod, tmp_path):
    trace = tmp_path / "t.jsonl"; trace.write_text("{}\n")
    rc = mod.main(["--run", "x", str(tmp_path / "nope"), str(trace),
                   "--no-figures"])
    assert rc == 2
