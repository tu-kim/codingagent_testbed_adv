"""Tests for scripts/arm/analyze_session_nesting.py — parent/task-subagent
structure extraction (no network/GPU; synthetic profile NDJSON)."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "arm" / "analyze_session_nesting.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_session_nesting", _SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["analyze_session_nesting"] = m
    spec.loader.exec_module(m)
    return m


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


def _write(dirpath, sid, evs):
    (dirpath / f"{sid}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")


def _setup(tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    # M (main): turn1 spawns `task`, then S runs, then M resumes turn2
    _write(prof, "M",
           _turn(1, 0, 2, 1000, 100, 0, "task")
           + _turn(2, 20, 22, 500, 80, 3000, None))
    # S (task sub-agent, NOT in trace): 3 turns nested inside M's window
    _write(prof, "S",
           _turn(1, 3, 4, 200, 50, 0, "bash")
           + _turn(2, 6, 7, 400, 40, 150, "bash")
           + _turn(3, 9, 10, 500, 45, 300, None))
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"session_id": "M"}) + "\n")
    return prof, trace


def test_recovers_subagent_not_in_trace_and_classifies(mod, tmp_path):
    prof, trace = _setup(tmp_path)
    out = tmp_path / "o"
    rc = mod.main(["--profiles", str(prof), "--trace", str(trace),
                   "--out", str(out)])
    assert rc == 0
    nest = list(csv.DictReader((out / "session_nesting.csv").open()))
    assert len(nest) == 1
    r = nest[0]
    assert (r["sample"], r["child"]) == ("M", "S")
    assert r["spawn_step"] == "1"
    assert r["spawn_had_task"] == "True"           # M turn1 carried `task`
    assert r["child_turns"] == "3"
    assert r["child_first_hit_ratio"] == "0.0"     # fresh prefix, no KV shared
    assert r["resume_step"] == "2"                 # M resumes at step 2
    assert float(r["resume_hit_ratio"]) > 0.8      # parent KV reused on resume


def test_ordinal_map_marks_interleave(mod, tmp_path):
    prof, trace = _setup(tmp_path)
    out = tmp_path / "o"
    mod.main(["--profiles", str(prof), "--trace", str(trace), "--out", str(out)])
    rows = list(csv.DictReader((out / "ordinal_map.csv").open()))
    roles = [(r["session_id"], r["role"]) for r in rows]
    # M turn, then 3 S turns (subagent), then M resume — the interleave
    assert roles == [("M", "sample"), ("S", "task_subagent"),
                     ("S", "task_subagent"), ("S", "task_subagent"),
                     ("M", "sample")]
    assert all(r["parent_sample"] == "M" for r in rows)


def test_missing_trace_returns_2(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    _write(prof, "M", _turn(1, 0, 2, 1000, 100, 0, None))
    rc = mod.main(["--profiles", str(prof), "--trace", str(tmp_path / "nope")])
    assert rc == 2
