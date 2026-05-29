"""Tests for scripts/analyze_subagent_time.py.

Pure NDJSON parsing + interval math. No network / GPU. The interesting
risk is the parallel-task UNION (vs naive sum) and the reconstruction
of a missing interval endpoint from duration_s.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_subagent_time.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_subagent_time", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_subagent_time"] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


# ---------- union_length ----------


def test_union_length_empty(mod):
    assert mod.union_length([]) == 0.0


def test_union_length_disjoint_sums(mod):
    # [0,2] and [5,8] don't overlap → 2 + 3 = 5
    assert mod.union_length([(0.0, 2.0), (5.0, 8.0)]) == pytest.approx(5.0)


def test_union_length_overlap_counts_once(mod):
    # [0,5] and [3,8] overlap → union is [0,8] = 8
    assert mod.union_length([(0.0, 5.0), (3.0, 8.0)]) == pytest.approx(8.0)


def test_union_length_nested_interval(mod):
    # [0,10] fully contains [2,4] → 10
    assert mod.union_length([(0.0, 10.0), (2.0, 4.0)]) == pytest.approx(10.0)


def test_union_length_unsorted_input(mod):
    # Order-independent.
    assert mod.union_length([(5.0, 8.0), (0.0, 2.0)]) == pytest.approx(5.0)


# ---------- interval reconstruction ----------


def test_interval_from_start_and_end(mod):
    ti = mod._ToolInterval(call_id="c", name="task", start_ts=10.0, end_ts=15.0)
    assert ti.interval() == (10.0, 15.0)


def test_interval_swaps_inverted_timestamps(mod):
    """Defensive: if end < start (clock weirdness), return ordered pair
    so union math never produces a negative length."""
    ti = mod._ToolInterval(call_id="c", name="task", start_ts=15.0, end_ts=10.0)
    assert ti.interval() == (10.0, 15.0)


def test_interval_reconstructs_end_from_duration(mod):
    """tool.end ts missing but duration_s present → end = start + dur."""
    ti = mod._ToolInterval(call_id="c", name="task", start_ts=10.0, duration_s=4.0)
    assert ti.interval() == (10.0, 14.0)


def test_interval_reconstructs_start_from_duration(mod):
    """tool.start missing (truncated) but end ts + duration present."""
    ti = mod._ToolInterval(call_id="c", name="task", end_ts=20.0, duration_s=4.0)
    assert ti.interval() == (16.0, 20.0)


def test_interval_none_when_unrecoverable(mod):
    ti = mod._ToolInterval(call_id="c", name="task")
    assert ti.interval() is None


# ---------- end-to-end turn summary ----------


def _turn_events(sid: str, step: int, *, turn_dur: float,
                 tasks: list[tuple[str, float, float]],
                 other_tools: list[tuple[str, float, float]] | None = None) -> list[dict]:
    """Build a turn's events. tasks/other_tools are (callID, start_ts, end_ts)."""
    evs: list[dict] = [{"ev": "turn.start", "sessionID": sid, "step": step, "ts": 0.0}]
    for cid, s, e in tasks:
        evs.append({"ev": "tool.start", "sessionID": sid, "step": step,
                    "callID": cid, "name": "task", "ts": s})
        evs.append({"ev": "tool.end", "sessionID": sid, "step": step,
                    "callID": cid, "name": "task", "ok": True,
                    "duration_s": e - s, "ts": e})
    for cid, s, e in (other_tools or []):
        evs.append({"ev": "tool.start", "sessionID": sid, "step": step,
                    "callID": cid, "name": "read", "ts": s})
        evs.append({"ev": "tool.end", "sessionID": sid, "step": step,
                    "callID": cid, "name": "read", "ok": True,
                    "duration_s": e - s, "ts": e})
    evs.append({"ev": "turn.end", "sessionID": sid, "step": step,
                "duration_s": turn_dur, "ts": turn_dur})
    return evs


def test_single_task_turn_ratio(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    # turn 10s, one task taking 6s → ratio 0.6
    _write(p, _turn_events("ses_a", 0, turn_dur=10.0,
                           tasks=[("c1", 2.0, 8.0)]))
    rows = mod.summarize(mod.load_turns(p))
    assert len(rows) == 1
    r = rows[0]
    assert r.n_task == 1
    assert r.subagent_wall_s == pytest.approx(6.0)
    assert r.subagent_sum_s == pytest.approx(6.0)
    assert r.ratio == pytest.approx(0.6)


def test_parallel_tasks_wall_less_than_sum(mod, tmp_path):
    """Two task calls overlapping in time: sum=10 but wall (union)=7.
    The ratio MUST use wall, not sum, else it could exceed 1.0."""
    p = tmp_path / "ses.jsonl"
    # task1 [1,6] = 5s, task2 [4,8] = 4s. overlap [4,6]. union [1,8] = 7. sum = 9.
    _write(p, _turn_events("ses_b", 0, turn_dur=10.0,
                           tasks=[("c1", 1.0, 6.0), ("c2", 4.0, 8.0)]))
    rows = mod.summarize(mod.load_turns(p))
    r = rows[0]
    assert r.n_task == 2
    assert r.subagent_sum_s == pytest.approx(9.0)
    assert r.subagent_wall_s == pytest.approx(7.0)   # union, not 9
    assert r.ratio == pytest.approx(0.7)             # 7/10, never > 1


def test_non_task_tools_excluded_from_subagent_time(mod, tmp_path):
    """A `read` tool in the same turn must NOT count toward sub-agent
    time -- only `task` does."""
    p = tmp_path / "ses.jsonl"
    _write(p, _turn_events("ses_c", 0, turn_dur=10.0,
                           tasks=[("t1", 1.0, 3.0)],
                           other_tools=[("r1", 4.0, 9.0)]))
    rows = mod.summarize(mod.load_turns(p))
    r = rows[0]
    assert r.n_task == 1
    assert r.subagent_wall_s == pytest.approx(2.0)   # only the task call
    assert r.ratio == pytest.approx(0.2)


def test_turn_without_task_has_zero_and_none_ratio_when_zero_duration(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    _write(p, _turn_events("ses_d", 0, turn_dur=5.0, tasks=[]))
    rows = mod.summarize(mod.load_turns(p))
    r = rows[0]
    assert r.n_task == 0
    assert r.subagent_wall_s == 0.0
    assert r.ratio == pytest.approx(0.0)


def test_ratio_none_when_turn_duration_missing(mod, tmp_path):
    """No turn.end duration AND no usable start/end ts → total None →
    ratio None (avoid div-by-zero / bogus number)."""
    p = tmp_path / "ses.jsonl"
    # Only a tool pair, no turn.start/turn.end with duration.
    _write(p, [
        {"ev": "tool.start", "sessionID": "ses_e", "step": 0,
         "callID": "c1", "name": "task", "ts": 1.0},
        {"ev": "tool.end", "sessionID": "ses_e", "step": 0,
         "callID": "c1", "name": "task", "ok": True, "duration_s": 2.0, "ts": 3.0},
    ])
    rows = mod.summarize(mod.load_turns(p))
    r = rows[0]
    assert r.turn_duration_s is None
    assert r.subagent_wall_s == pytest.approx(2.0)
    assert r.ratio is None


def test_turn_with_no_turn_end_has_none_duration(mod, tmp_path):
    """A truncated turn (turn.start + tools but NO turn.end) carries no
    duration_s -- the profile patch only emits duration_s on turn.end,
    and there is deliberately no ts-based fallback. total stays None →
    ratio None (we don't fabricate a turn time)."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        {"ev": "turn.start", "sessionID": "ses_f", "step": 0, "ts": 100.0},
        {"ev": "tool.start", "sessionID": "ses_f", "step": 0,
         "callID": "c1", "name": "task", "ts": 101.0},
        {"ev": "tool.end", "sessionID": "ses_f", "step": 0,
         "callID": "c1", "name": "task", "ok": True, "duration_s": 4.0, "ts": 105.0},
        # no turn.end
    ])
    rows = mod.summarize(mod.load_turns(p))
    r = rows[0]
    assert r.turn_duration_s is None
    assert r.subagent_wall_s == pytest.approx(4.0)
    assert r.ratio is None


def test_failed_task_still_counted_in_wall(mod, tmp_path):
    """A sub-agent that errored still consumed wall time; it must count
    toward subagent_wall_s AND bump n_task_failed."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        {"ev": "turn.start", "sessionID": "ses_g", "step": 0, "ts": 0.0},
        {"ev": "tool.start", "sessionID": "ses_g", "step": 0,
         "callID": "c1", "name": "task", "ts": 1.0},
        {"ev": "tool.end", "sessionID": "ses_g", "step": 0,
         "callID": "c1", "name": "task", "ok": False, "duration_s": 3.0, "ts": 4.0},
        {"ev": "turn.end", "sessionID": "ses_g", "step": 0, "duration_s": 10.0, "ts": 10.0},
    ])
    rows = mod.summarize(mod.load_turns(p))
    r = rows[0]
    assert r.n_task == 1
    assert r.n_task_failed == 1
    assert r.subagent_wall_s == pytest.approx(3.0)


def test_multiple_sessions_and_steps_partitioned(mod, tmp_path):
    """Turns keyed by (sessionID, step). Two sessions × two steps stay
    separate so a parent session and its nested child don't merge."""
    p = tmp_path / "all.jsonl"
    events = (
        _turn_events("ses_parent", 0, turn_dur=10.0, tasks=[("c1", 1.0, 9.0)])
        + _turn_events("ses_parent", 1, turn_dur=4.0, tasks=[])
        + _turn_events("ses_child", 0, turn_dur=8.0, tasks=[])  # child has no task
    )
    _write(p, events)
    rows = mod.summarize(mod.load_turns(p))
    keyed = {(r.session_id, r.step): r for r in rows}
    assert keyed[("ses_parent", 0)].ratio == pytest.approx(0.8)
    assert keyed[("ses_parent", 1)].n_task == 0
    assert keyed[("ses_child", 0)].n_task == 0
    assert len(rows) == 3


def test_custom_task_tool_name(mod, tmp_path):
    """--task-tool-name lets the user treat a differently-named tool as
    the sub-agent spawn."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        {"ev": "turn.start", "sessionID": "s", "step": 0, "ts": 0.0},
        {"ev": "tool.start", "sessionID": "s", "step": 0,
         "callID": "c1", "name": "delegate", "ts": 1.0},
        {"ev": "tool.end", "sessionID": "s", "step": 0,
         "callID": "c1", "name": "delegate", "ok": True, "duration_s": 5.0, "ts": 6.0},
        {"ev": "turn.end", "sessionID": "s", "step": 0, "duration_s": 10.0, "ts": 10.0},
    ])
    rows = mod.summarize(mod.load_turns(p), task_name="delegate")
    assert rows[0].n_task == 1
    assert rows[0].subagent_wall_s == pytest.approx(5.0)
    # Under the default name it would be 0.
    rows_default = mod.summarize(mod.load_turns(p))
    assert rows_default[0].n_task == 0


def test_malformed_lines_skipped(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    p.write_text(
        json.dumps({"ev": "turn.start", "sessionID": "s", "step": 0, "ts": 0.0}) + "\n"
        + "GARBAGE not json\n"
        + json.dumps({"ev": "tool.start", "sessionID": "s", "step": 0,
                      "callID": "c1", "name": "task", "ts": 1.0}) + "\n"
        + json.dumps({"ev": "tool.end", "sessionID": "s", "step": 0,
                      "callID": "c1", "name": "task", "ok": True,
                      "duration_s": 2.0, "ts": 3.0}) + "\n"
        + json.dumps({"ev": "turn.end", "sessionID": "s", "step": 0,
                      "duration_s": 8.0, "ts": 8.0}) + "\n"
    )
    rows = mod.summarize(mod.load_turns(p))
    assert len(rows) == 1
    assert rows[0].subagent_wall_s == pytest.approx(2.0)


# ---------- CSV / main ----------


def test_main_writes_csv_and_returns_zero(mod, tmp_path, capsys):
    prof = tmp_path / "profiles"
    prof.mkdir()
    _write(prof / "ses_a.jsonl",
           _turn_events("ses_a", 0, turn_dur=10.0, tasks=[("c1", 2.0, 8.0)]))
    out = tmp_path / "out"
    rc = mod.main(["--profile", str(prof), "--output", str(out)])
    assert rc == 0
    csv_path = out / "subagent_time.csv"
    rows = list(csv.DictReader(csv_path.open()))
    assert len(rows) == 1
    assert rows[0]["session_id"] == "ses_a"
    assert rows[0]["n_task"] == "1"
    assert float(rows[0]["ratio"]) == pytest.approx(0.6)
    assert float(rows[0]["subagent_wall_s"]) == pytest.approx(6.0)
    captured = capsys.readouterr().out
    assert "Aggregate sub-agent ratio" in captured


def test_main_accepts_single_file(mod, tmp_path):
    f = tmp_path / "agg.jsonl"
    _write(f, _turn_events("ses_a", 0, turn_dur=4.0, tasks=[("c1", 0.0, 1.0)]))
    out = tmp_path / "out"
    rc = mod.main(["--profile", str(f), "--output", str(out)])
    assert rc == 0
    assert (out / "subagent_time.csv").exists()


def test_main_missing_profile_returns_2(mod, tmp_path, capsys):
    rc = mod.main(["--profile", str(tmp_path / "nope"), "--output", str(tmp_path / "o")])
    assert rc == 2
    assert "profile path not found" in capsys.readouterr().err


def test_main_empty_profile_returns_1(mod, tmp_path, capsys):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    rc = mod.main(["--profile", str(f), "--output", str(tmp_path / "o")])
    assert rc == 1
    assert "no turns found" in capsys.readouterr().err
