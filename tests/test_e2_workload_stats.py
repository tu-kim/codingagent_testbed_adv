"""Tests for scripts/arm/e2_workload_stats.py — ISL/OSL, tool counts,
per-tool token effects, and tool transition extraction."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "arm" / "e2_workload_stats.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("e2_workload_stats", _SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["e2_workload_stats"] = m
    spec.loader.exec_module(m)
    return m


class _T:
    """Turn double: session_id/step + effective_input/output_tokens/
    reasoning_tokens/tool_names, matching agent_trace_stats.TurnRec's
    surface as e2 consumes it."""

    def __init__(self, sid, step, eff_input=None, output=None,
                 reasoning=0, tools=None):
        self.session_id = sid
        self.step = step
        self.effective_input = eff_input
        self.output_tokens = output
        self.reasoning_tokens = reasoning
        self.tool_names = tools or []


# ---------- isl_osl ----------


def test_isl_osl_skips_none_per_series(mod):
    turns = [
        _T("A", 1, eff_input=500, output=50),
        _T("A", 2, eff_input=None, output=60),   # ISL dropped, OSL kept
        _T("A", 3, eff_input=700, output=None),  # OSL dropped, ISL kept
    ]
    isl, osl = mod.isl_osl(turns)
    assert isl == [500.0, 700.0]
    assert osl == [50.0, 60.0]


# ---------- tool_counts ----------


def test_tool_counts_multi_tool_turns(mod):
    turns = [
        _T("A", 1, tools=["read", "bash"]),
        _T("A", 2, tools=["read"]),
        _T("B", 1, tools=[]),
    ]
    counts = mod.tool_counts(turns)
    assert counts["read"] == 2
    assert counts["bash"] == 1
    assert "(none)" not in counts   # tool_counts only counts invocations


# ---------- tool_token_effects ----------


def test_tool_token_effects_hand_computed(mod):
    # turn1 -> turn2: eff(2)-eff(1)-kept(1) = 1000-500-(80-20) = 440
    # turn2 -> turn3: eff(3)-eff(2)-kept(2) = 900-1000-40 = -140 (compaction, kept)
    turns = [
        _T("A", 1, eff_input=500, output=80, reasoning=20, tools=["read"]),
        _T("A", 2, eff_input=1000, output=40, reasoning=0, tools=["bash"]),
        _T("A", 3, eff_input=900, output=10, tools=[]),
    ]
    effects = mod.tool_token_effects(turns)
    assert effects["read"]["out"] == [80.0]
    assert effects["read"]["added"] == [440.0]
    assert effects["bash"]["out"] == [40.0]
    assert effects["bash"]["added"] == [-140.0]
    # turn3 has no next turn: out recorded, no "added" entry
    assert effects["(none)"]["out"] == [10.0]
    assert effects["(none)"]["added"] == []


def test_tool_token_effects_missing_next_turn_or_none_fields_skipped(mod):
    turns = [
        # no step+1 turn in this session at all -> "added" skipped
        _T("A", 1, eff_input=500, output=80, tools=["read"]),
        # next turn exists but its effective_input is None -> skipped
        _T("B", 1, eff_input=500, output=80, tools=["bash"]),
        _T("B", 2, eff_input=None, output=40, tools=[]),
        # this turn's own output_tokens is None -> "added" skipped
        _T("C", 1, eff_input=500, output=None, tools=["webfetch"]),
        _T("C", 2, eff_input=900, output=30, tools=[]),
    ]
    effects = mod.tool_token_effects(turns)
    assert effects["read"]["added"] == []
    assert effects["bash"]["added"] == []
    assert effects["webfetch"]["added"] == []
    # out is still recorded when present
    assert effects["read"]["out"] == [80.0]
    assert effects["webfetch"]["out"] == []   # output_tokens None -> not appended


# ---------- tool_transitions ----------


def test_tool_transitions_cross_session_isolation_and_ordering(mod):
    # A: step1 read -> step2 bash -> step3 (none)
    # B: step1 bash -> step2 read
    # Fed out of step order to confirm sort-by-step within session.
    turns = [
        _T("A", 2, tools=["bash"]),
        _T("A", 1, tools=["read"]),
        _T("A", 3, tools=[]),
        _T("B", 1, tools=["bash"]),
        _T("B", 2, tools=["read"]),
    ]
    trans = mod.tool_transitions(turns)
    assert trans[("read", "bash")] == 1
    assert trans[("bash", "(none)")] == 1
    assert trans[("bash", "read")] == 1
    # no cross-session pair (A's last tool -> B's first tool) must appear
    assert ("bash", "bash") not in trans or trans[("bash", "bash")] == 0
    assert len(trans) == 3


def test_tool_transitions_none_as_next_state(mod):
    turns = [
        _T("A", 1, tools=["read"]),
        _T("A", 2, tools=[]),
    ]
    trans = mod.tool_transitions(turns)
    assert trans == {("read", "(none)"): 1}


# ---------- main() integration ----------


def _turn_events(step, a, b, inp, out, cache, tool, reasoning=0):
    evs = [{"ev": "llm.start", "ts": a, "step": step},
           {"ev": "llm.end", "ts": b, "step": step, "duration_s": b - a,
            "tokens": {"input": inp, "output": out, "reasoning": reasoning,
                       "cache": {"read": cache}},
            "dynamo": {}}]
    if tool:
        evs.append({"ev": "tool.end", "ts": b + 0.05, "step": step,
                    "name": tool, "callID": f"c{step}", "duration_s": 0.5,
                    "ok": True})
    return evs


def test_main_interleaved_sessions_no_figures(mod, tmp_path):
    prof = tmp_path / "profiles"
    prof.mkdir()
    (prof / "A.jsonl").write_text("".join(
        json.dumps(e) + "\n" for e in
        _turn_events(1, 0, 1, 500, 50, 0, "read")
        + _turn_events(2, 3, 4, 600, 60, 500, None)))
    (prof / "B.jsonl").write_text("".join(
        json.dumps(e) + "\n" for e in
        _turn_events(1, 0.5, 1.5, 400, 40, 0, "bash")
        + _turn_events(2, 3.5, 4.5, 500, 50, 400, None)))
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"session_id": "A"}) + "\n"
                      + json.dumps({"session_id": "B"}) + "\n")
    out = tmp_path / "o"
    rc = mod.main(["--profiles", str(prof), "--trace", str(trace),
                   "--out", str(out), "--no-figures"])
    assert rc == 0
    assert not list(out.glob("*.pdf"))
