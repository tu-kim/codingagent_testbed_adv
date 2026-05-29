"""Tests for scripts/analyze_tool_time.py.

The complement of analyze_subagent_time: per-turn wall time of NON-task
tools. Key risks: the exclude-set filtering, the union (vs sum) for
parallel tools, and the by-tool-name breakdown.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_tool_time.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_tool_time", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_tool_time"] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _tool(sid: str, step: int, call_id: str, name: str, start: float, end: float,
          ok: bool = True) -> list[dict]:
    return [
        {"ev": "tool.start", "sessionID": sid, "step": step,
         "callID": call_id, "name": name, "ts": start},
        {"ev": "tool.end", "sessionID": sid, "step": step,
         "callID": call_id, "name": name, "ok": ok,
         "duration_s": end - start, "ts": end},
    ]


def _turn_end(sid: str, step: int, dur: float) -> dict:
    return {"ev": "turn.end", "sessionID": sid, "step": step,
            "duration_s": dur, "ts": dur}


# ---------- union_length (shared math) ----------


def test_union_length_overlap_counts_once(mod):
    assert mod.union_length([(0.0, 5.0), (3.0, 8.0)]) == pytest.approx(8.0)


def test_union_length_empty(mod):
    assert mod.union_length([]) == 0.0


# ---------- exclude filtering ----------


def test_task_excluded_by_default(mod, tmp_path):
    """Default exclude={task}: a turn with only a task call reports
    zero non-task tool time."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        {"ev": "turn.start", "sessionID": "s", "step": 0, "ts": 0.0},
        *_tool("s", 0, "c1", "task", 1.0, 9.0),
        _turn_end("s", 0, 10.0),
    ])
    rows = mod.summarize(mod.load_turns(p), frozenset({"task"}))
    assert rows[0].n_calls == 0
    assert rows[0].tool_wall_s == 0.0
    assert rows[0].ratio == pytest.approx(0.0)


def test_non_task_tools_counted(mod, tmp_path):
    """read + grep counted; task in the same turn ignored."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        *_tool("s", 0, "c1", "read", 1.0, 3.0),     # 2s
        *_tool("s", 0, "c2", "grep", 3.0, 4.0),     # 1s, disjoint from read
        *_tool("s", 0, "c3", "task", 4.0, 9.0),     # excluded
        _turn_end("s", 0, 10.0),
    ])
    rows = mod.summarize(mod.load_turns(p), frozenset({"task"}))
    r = rows[0]
    assert r.n_calls == 2                       # read + grep, NOT task
    assert r.tool_wall_s == pytest.approx(3.0)  # 2 + 1 disjoint
    assert r.ratio == pytest.approx(0.3)


def test_empty_exclude_includes_task(mod, tmp_path):
    """exclude=frozenset() (empty) → every tool counts, including task."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        *_tool("s", 0, "c1", "task", 1.0, 5.0),     # 4s
        *_tool("s", 0, "c2", "read", 5.0, 7.0),     # 2s
        _turn_end("s", 0, 10.0),
    ])
    rows = mod.summarize(mod.load_turns(p), frozenset())
    r = rows[0]
    assert r.n_calls == 2
    assert r.tool_wall_s == pytest.approx(6.0)   # task + read both counted
    assert "task" in r.per_tool_wall


def test_custom_multi_exclude(mod, tmp_path):
    """exclude={task,bash} drops both."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        *_tool("s", 0, "c1", "task", 0.0, 2.0),
        *_tool("s", 0, "c2", "bash", 2.0, 4.0),
        *_tool("s", 0, "c3", "read", 4.0, 7.0),     # 3s, only this survives
        _turn_end("s", 0, 10.0),
    ])
    rows = mod.summarize(mod.load_turns(p), frozenset({"task", "bash"}))
    r = rows[0]
    assert r.n_calls == 1
    assert r.tool_wall_s == pytest.approx(3.0)
    assert set(r.per_tool_wall) == {"read"}


# ---------- parallel union vs sum ----------


def test_parallel_non_task_tools_wall_less_than_sum(mod, tmp_path):
    """Two read calls overlapping: sum double counts, wall (union) does
    not -- ratio must stay <= 1."""
    p = tmp_path / "ses.jsonl"
    # read1 [1,6]=5, read2 [4,8]=4. union [1,8]=7. sum=9.
    _write(p, [
        *_tool("s", 0, "c1", "read", 1.0, 6.0),
        *_tool("s", 0, "c2", "read", 4.0, 8.0),
        _turn_end("s", 0, 10.0),
    ])
    rows = mod.summarize(mod.load_turns(p), frozenset({"task"}))
    r = rows[0]
    assert r.tool_sum_s == pytest.approx(9.0)
    assert r.tool_wall_s == pytest.approx(7.0)
    assert r.ratio == pytest.approx(0.7)


# ---------- per-tool breakdown ----------


def test_per_tool_wall_independent_unions(mod, tmp_path):
    """per_tool_wall computes a SEPARATE union per tool name. Two
    different tools overlapping → each per-tool wall counts its own
    full span (they can sum to more than the combined tool_wall_s)."""
    p = tmp_path / "ses.jsonl"
    # read [0,5]=5, grep [3,8]=5. combined union [0,8]=8. per-tool: read 5, grep 5.
    _write(p, [
        *_tool("s", 0, "c1", "read", 0.0, 5.0),
        *_tool("s", 0, "c2", "grep", 3.0, 8.0),
        _turn_end("s", 0, 10.0),
    ])
    rows = mod.summarize(mod.load_turns(p), frozenset({"task"}))
    r = rows[0]
    assert r.tool_wall_s == pytest.approx(8.0)        # combined union
    assert r.per_tool_wall["read"] == pytest.approx(5.0)
    assert r.per_tool_wall["grep"] == pytest.approx(5.0)
    # per-tool sums to 10 > 8 -- intentional (overlap counted in both).


def test_by_tool_name_aggregates_across_turns(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    _write(p, [
        *_tool("s", 0, "c1", "read", 0.0, 2.0),
        _turn_end("s", 0, 5.0),
        *_tool("s", 1, "c2", "read", 0.0, 3.0),
        *_tool("s", 1, "c3", "grep", 0.0, 1.0),
        _turn_end("s", 1, 6.0),
    ])
    rows = mod.summarize(mod.load_turns(p), frozenset({"task"}))
    agg = mod.by_tool_name(rows)
    # read: 2 + 3 = 5 across 2 turns; grep: 1 across 1 turn.
    assert agg["read"] == (pytest.approx(5.0), 2)
    assert agg["grep"] == (pytest.approx(1.0), 1)


# ---------- duration / ratio edge cases ----------


def test_ratio_none_when_turn_duration_missing(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    _write(p, _tool("s", 0, "c1", "read", 1.0, 3.0))   # no turn.end
    rows = mod.summarize(mod.load_turns(p), frozenset({"task"}))
    r = rows[0]
    assert r.turn_duration_s is None
    assert r.tool_wall_s == pytest.approx(2.0)
    assert r.ratio is None


def test_failed_tool_still_counted(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    _write(p, [
        *_tool("s", 0, "c1", "bash", 1.0, 4.0, ok=False),
        _turn_end("s", 0, 10.0),
    ])
    rows = mod.summarize(mod.load_turns(p), frozenset({"task"}))
    assert rows[0].tool_wall_s == pytest.approx(3.0)


def test_malformed_lines_skipped(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    p.write_text(
        "not json\n"
        + json.dumps(_tool("s", 0, "c1", "read", 1.0, 3.0)[0]) + "\n"
        + json.dumps(_tool("s", 0, "c1", "read", 1.0, 3.0)[1]) + "\n"
        + json.dumps(_turn_end("s", 0, 8.0)) + "\n"
    )
    rows = mod.summarize(mod.load_turns(p), frozenset({"task"}))
    assert len(rows) == 1
    assert rows[0].tool_wall_s == pytest.approx(2.0)


# ---------- CSV / main ----------


def test_main_writes_both_csvs(mod, tmp_path, capsys):
    prof = tmp_path / "profiles"
    prof.mkdir()
    _write(prof / "ses_a.jsonl", [
        *_tool("ses_a", 0, "c1", "read", 2.0, 8.0),    # 6s
        *_tool("ses_a", 0, "c2", "task", 0.0, 1.0),    # excluded
        _turn_end("ses_a", 0, 10.0),
    ])
    out = tmp_path / "out"
    rc = mod.main(["--profile", str(prof), "--output", str(out)])
    assert rc == 0

    per_turn = list(csv.DictReader((out / "tool_time_per_turn.csv").open()))
    assert len(per_turn) == 1
    assert per_turn[0]["n_calls"] == "1"
    assert float(per_turn[0]["tool_wall_s"]) == pytest.approx(6.0)
    assert float(per_turn[0]["ratio"]) == pytest.approx(0.6)

    by_name = list(csv.DictReader((out / "tool_time_by_name.csv").open()))
    names = {r["tool_name"] for r in by_name}
    assert names == {"read"}        # task excluded from breakdown too
    assert float(by_name[0]["total_wall_s"]) == pytest.approx(6.0)
    assert float(by_name[0]["pct_of_tool_wall"]) == pytest.approx(100.0)

    captured = capsys.readouterr().out
    assert "Aggregate non-task ratio" in captured
    assert "Wall time by tool name" in captured


def test_main_exclude_flag_parsed(mod, tmp_path):
    prof = tmp_path / "p.jsonl"
    _write(prof, [
        *_tool("s", 0, "c1", "read", 0.0, 2.0),
        *_tool("s", 0, "c2", "bash", 2.0, 5.0),
        _turn_end("s", 0, 10.0),
    ])
    out = tmp_path / "out"
    rc = mod.main(["--profile", str(prof), "--output", str(out),
                   "--exclude", "task,bash"])
    assert rc == 0
    by_name = list(csv.DictReader((out / "tool_time_by_name.csv").open()))
    assert {r["tool_name"] for r in by_name} == {"read"}   # bash excluded


def test_main_empty_exclude_string_includes_all(mod, tmp_path):
    prof = tmp_path / "p.jsonl"
    _write(prof, [
        *_tool("s", 0, "c1", "task", 0.0, 4.0),
        _turn_end("s", 0, 10.0),
    ])
    out = tmp_path / "out"
    rc = mod.main(["--profile", str(prof), "--output", str(out), "--exclude", ""])
    assert rc == 0
    by_name = list(csv.DictReader((out / "tool_time_by_name.csv").open()))
    assert {r["tool_name"] for r in by_name} == {"task"}   # nothing excluded


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
