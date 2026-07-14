"""Tests for the --exclude-calls integration in scripts/analyze_profiles.py.

Scope is deliberately narrow: the callID-based exclusion added so a
hung server/bash call (flagged by filter_hanging_tools.py) can be
dropped. Two effects are asserted:
  1. per-tool stats (fig2/4/5 family) drop the excluded CALL, and
  2. any TURN containing an excluded call is skipped wholesale from the
     turn-decomposition / ratio views (its pre-aggregated tool_wall_s
     still carries the hang) -- same treatment task turns get.

analyze_profiles imports matplotlib at module load, so the whole file is
skipped when matplotlib is unavailable.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_profiles.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_profiles", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_profiles"] = module  # dataclass needs the module registered
    spec.loader.exec_module(module)
    return module


def _tool_events(sid: str, step: int, call_id: str, name: str,
                 dur: float, ts: float, *, ok: bool = True,
                 out_chars: int = 0) -> list[dict]:
    return [
        {"ev": "tool.start", "sessionID": sid, "step": step,
         "callID": call_id, "args_head": name},
        {"ev": "tool.end", "sessionID": sid, "step": step, "callID": call_id,
         "name": name, "ok": ok, "duration_s": dur, "ts": ts,
         "output_chars": out_chars},
    ]


def _turn_end(sid: str, step: int, *, dur: float, llm: float, tool: float,
              ts: float) -> dict:
    return {"ev": "turn.end", "sessionID": sid, "step": step,
            "duration_s": dur, "llm_wall_s": llm, "tool_wall_s": tool, "ts": ts}


def _write(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _base_profile(tmp_path: Path) -> Path:
    """One session, two turns: turn0 has a 300s hung bash + turn.end
    tool_wall reflecting it; turn1 a normal 2s bash."""
    f = tmp_path / "ses1.jsonl"
    _write(f, [
        {"ev": "query.start", "sessionID": "ses1", "ts": 0.0},
        *_tool_events("ses1", 0, "c_hang", "bash", 300.0, 300.0),
        _turn_end("ses1", 0, dur=305.0, llm=5.0, tool=300.0, ts=305.0),
        *_tool_events("ses1", 1, "c_ok", "bash", 2.0, 320.0, out_chars=10),
        _turn_end("ses1", 1, dur=10.0, llm=8.0, tool=2.0, ts=330.0),
    ])
    return f


# ---------- load-time marking ----------


def test_load_marks_excluded_calls(mod, tmp_path):
    prof = _base_profile(tmp_path)
    sessions = mod.load_sessions(prof, exclude_calls={"c_hang"})
    tools = {tc.call_id: tc for s in sessions.values()
             for t in s.turns.values() for tc in t.tools}
    assert tools["c_hang"].excluded is True
    assert tools["c_ok"].excluded is False


def test_load_no_exclude_leaves_all_included(mod, tmp_path):
    prof = _base_profile(tmp_path)
    sessions = mod.load_sessions(prof)
    assert all(not tc.excluded for s in sessions.values()
               for t in s.turns.values() for tc in t.tools)


def test_turn_has_excluded_call_helper(mod, tmp_path):
    prof = _base_profile(tmp_path)
    sessions = mod.load_sessions(prof, exclude_calls={"c_hang"})
    turns = {step: t for s in sessions.values() for step, t in s.turns.items()}
    assert mod._turn_has_excluded_call(turns[0]) is True
    assert mod._turn_has_excluded_call(turns[1]) is False


# ---------- per-tool stats drop the excluded call ----------


def test_per_tool_duration_drops_hang(mod, tmp_path):
    prof = _base_profile(tmp_path)
    sessions = mod.load_sessions(prof, exclude_calls={"c_hang"})
    stats = mod.per_tool_duration_stats(sessions)
    mean, _std, n = stats["bash"]
    assert n == 1                      # only the 2s call survives
    assert mean == pytest.approx(2.0)


def test_per_tool_duration_without_exclude_includes_hang(mod, tmp_path):
    prof = _base_profile(tmp_path)
    sessions = mod.load_sessions(prof)   # no exclusion
    _mean, _std, n = mod.per_tool_duration_stats(sessions)["bash"]
    assert n == 2                        # both calls counted


# ---------- turn-decomposition / ratio views skip the hang turn ----------


def test_turn_ratio_rows_skip_hang_turn(mod, tmp_path):
    prof = _base_profile(tmp_path)
    sessions = mod.load_sessions(prof, exclude_calls={"c_hang"})
    rows = mod.turn_ratio_rows(sessions)
    steps = {step for _sid, step, *_ in rows}
    assert steps == {1}                  # hang turn (step 0) dropped
    # without exclusion both turns present
    rows_all = mod.turn_ratio_rows(mod.load_sessions(prof))
    assert {step for _s, step, *_ in rows_all} == {0, 1}


def test_collect_turn_decomposition_skips_hang_turn(mod, tmp_path):
    prof = _base_profile(tmp_path)
    sessions = mod.load_sessions(prof, exclude_calls={"c_hang"})
    rows = mod._collect_turn_decomposition(sessions)
    steps = {r[1] for r in rows}         # (session_id, step, ...)
    assert 0 not in steps                # hang turn excluded
    assert steps == {1}


# ---------- main() end-to-end ----------


def test_main_exclude_calls_end_to_end(mod, tmp_path, capsys):
    prof_dir = tmp_path / "prof"
    prof_dir.mkdir()
    _base_profile(prof_dir).rename(prof_dir / "ses1.jsonl")
    excl = tmp_path / "exclude.txt"
    excl.write_text("c_hang\n\n")        # blank line tolerated
    out = tmp_path / "figs"
    rc = mod.main(["--input", str(prof_dir), "--output", str(out),
                   "--exclude-calls", str(excl)])
    assert rc == 0
    assert "matched 1 tool call(s)" in capsys.readouterr().out


def test_main_missing_exclude_calls_file_returns_2(mod, tmp_path, capsys):
    prof_dir = tmp_path / "prof"
    prof_dir.mkdir()
    _base_profile(prof_dir).rename(prof_dir / "ses1.jsonl")
    rc = mod.main(["--input", str(prof_dir), "--output", str(tmp_path / "o"),
                   "--exclude-calls", str(tmp_path / "nope.txt")])
    assert rc == 2
    assert "exclude-calls file not found" in capsys.readouterr().err


def test_main_exclude_calls_no_match_warns(mod, tmp_path, capsys):
    prof_dir = tmp_path / "prof"
    prof_dir.mkdir()
    _base_profile(prof_dir).rename(prof_dir / "ses1.jsonl")
    excl = tmp_path / "exclude.txt"
    excl.write_text("c_does_not_exist\n")
    out = tmp_path / "figs"
    rc = mod.main(["--input", str(prof_dir), "--output", str(out),
                   "--exclude-calls", str(excl)])
    assert rc == 0
    assert "matched any tool.end" in capsys.readouterr().err
