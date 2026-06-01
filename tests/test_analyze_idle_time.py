"""Tests for scripts/analyze_idle_time.py.

Inter-turn idle decomposition from profile NDJSON. All four event types
(query.start, turn.start, turn.end, query.end) carry `ts`, so gaps are
exact timestamp differences. Covers gap computation, the bootstrap /
inter-turn / teardown split, missing-timestamp handling, and CSVs.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_idle_time.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_idle_time", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_idle_time"] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, events: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _q_start(sid: str, ts: float) -> dict:
    return {"ev": "query.start", "sessionID": sid, "ts": ts}


def _q_end(sid: str, ts: float, dur: float) -> dict:
    return {"ev": "query.end", "sessionID": sid, "ts": ts, "duration_s": dur}


def _turn(sid: str, step: int, start: float, end: float) -> list[dict]:
    return [
        {"ev": "turn.start", "sessionID": sid, "step": step, "ts": start},
        {"ev": "turn.end", "sessionID": sid, "step": step, "ts": end,
         "duration_s": end - start},
    ]


# ---------- gap computation ----------


def test_single_inter_turn_gap(mod, tmp_path):
    """Two turns: turn0 [10,15], turn1 [18,22]. Gap = 18-15 = 3."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        *_turn("s", 0, 10.0, 15.0),
        *_turn("s", 1, 18.0, 22.0),
    ])
    sessions = mod.load_sessions(p)
    gaps = mod.session_gaps(sessions[0])
    assert len(gaps) == 1
    assert gaps[0].prev_step == 0
    assert gaps[0].next_step == 1
    assert gaps[0].idle_s == pytest.approx(3.0)


def test_multiple_gaps_ordered_by_step(mod, tmp_path):
    """Three turns produce two consecutive gaps, ordered by step even if
    events arrive out of order in the file."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        *_turn("s", 2, 30.0, 35.0),   # written first, but step 2
        *_turn("s", 0, 10.0, 14.0),
        *_turn("s", 1, 16.0, 25.0),
    ])
    gaps = mod.session_gaps(mod.load_sessions(p)[0])
    assert [(g.prev_step, g.next_step) for g in gaps] == [(0, 1), (1, 2)]
    assert gaps[0].idle_s == pytest.approx(2.0)    # 16 - 14
    assert gaps[1].idle_s == pytest.approx(5.0)    # 30 - 25


def test_gap_skipped_when_endpoint_missing(mod, tmp_path):
    """If a turn has no turn.end (truncated), the gap that would touch
    its end_ts is skipped rather than computed from None."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        {"ev": "turn.start", "sessionID": "s", "step": 0, "ts": 10.0},
        # no turn.end for step 0
        *_turn("s", 1, 20.0, 25.0),
    ])
    gaps = mod.session_gaps(mod.load_sessions(p)[0])
    assert gaps == []   # step0 has no end_ts → gap(0→1) not computable


# ---------- session decomposition ----------


def test_full_decomposition(mod, tmp_path):
    """query [100,140] (dur 40). turn0 [105,115], turn1 [120,135].
      bootstrap = 105-100 = 5
      inter     = 120-115 = 5
      teardown  = 140-135 = 5
      busy      = 10 + 15 = 25
      total_idle= 15  → idle_pct = 15/40 = 37.5%
    busy + total_idle = 25 + 15 = 40 = query wall (accounts cleanly)."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        _q_start("s", 100.0),
        *_turn("s", 0, 105.0, 115.0),
        *_turn("s", 1, 120.0, 135.0),
        _q_end("s", 140.0, 40.0),
    ])
    r = mod.session_idle(mod.load_sessions(p)[0])
    assert r.n_turns == 2
    assert r.busy_turn_s == pytest.approx(25.0)
    assert r.bootstrap_s == pytest.approx(5.0)
    assert r.inter_turn_idle_s == pytest.approx(5.0)
    assert r.teardown_s == pytest.approx(5.0)
    assert r.total_idle_s == pytest.approx(15.0)
    assert r.idle_pct == pytest.approx(37.5)
    # accounting identity
    assert r.busy_turn_s + r.total_idle_s == pytest.approx(r.query_duration_s)


def test_single_turn_has_no_inter_turn_idle(mod, tmp_path):
    """One turn: only bootstrap + teardown, zero inter-turn idle."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        _q_start("s", 0.0),
        *_turn("s", 0, 2.0, 8.0),
        _q_end("s", 10.0, 10.0),
    ])
    r = mod.session_idle(mod.load_sessions(p)[0])
    assert r.n_turns == 1
    assert r.inter_turn_idle_s == pytest.approx(0.0)
    assert r.bootstrap_s == pytest.approx(2.0)
    assert r.teardown_s == pytest.approx(2.0)
    assert r.total_idle_s == pytest.approx(4.0)


def test_bootstrap_none_when_query_start_missing(mod, tmp_path):
    """No query.start event → bootstrap can't be computed → None, and
    it's excluded from total_idle (which then = inter + teardown)."""
    p = tmp_path / "ses.jsonl"
    _write(p, [
        *_turn("s", 0, 5.0, 10.0),
        *_turn("s", 1, 12.0, 18.0),
        _q_end("s", 20.0, 20.0),
    ])
    r = mod.session_idle(mod.load_sessions(p)[0])
    assert r.bootstrap_s is None
    assert r.inter_turn_idle_s == pytest.approx(2.0)   # 12 - 10
    assert r.teardown_s == pytest.approx(2.0)          # 20 - 18
    assert r.total_idle_s == pytest.approx(4.0)        # bootstrap excluded


def test_teardown_none_when_query_end_missing(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    _write(p, [
        _q_start("s", 0.0),
        *_turn("s", 0, 2.0, 8.0),
    ])
    r = mod.session_idle(mod.load_sessions(p)[0])
    assert r.teardown_s is None
    assert r.bootstrap_s == pytest.approx(2.0)
    assert r.total_idle_s == pytest.approx(2.0)


def test_idle_pct_none_when_no_query_duration(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    _write(p, [*_turn("s", 0, 2.0, 8.0), *_turn("s", 1, 10.0, 12.0)])
    r = mod.session_idle(mod.load_sessions(p)[0])
    assert r.query_duration_s is None
    assert r.idle_pct is None
    # inter-turn idle still computed from turn timestamps
    assert r.inter_turn_idle_s == pytest.approx(2.0)


def test_sessions_partitioned(mod, tmp_path):
    """Two sessions in one file stay separate (a parent and its nested
    sub-agent child each get their own decomposition)."""
    p = tmp_path / "all.jsonl"
    _write(p, [
        _q_start("parent", 0.0),
        *_turn("parent", 0, 1.0, 5.0),
        *_turn("parent", 1, 9.0, 14.0),
        _q_end("parent", 15.0, 15.0),
        _q_start("child", 2.0),
        *_turn("child", 0, 2.5, 4.5),
        _q_end("child", 5.0, 3.0),
    ])
    rows = {r.session_id: r for r in
            (mod.session_idle(s) for s in mod.load_sessions(p))}
    assert rows["parent"].inter_turn_idle_s == pytest.approx(4.0)   # 9 - 5
    assert rows["child"].n_turns == 1
    assert rows["child"].inter_turn_idle_s == pytest.approx(0.0)


def test_malformed_lines_skipped(mod, tmp_path):
    p = tmp_path / "ses.jsonl"
    p.write_text(
        "garbage\n"
        + json.dumps(_q_start("s", 0.0)) + "\n"
        + json.dumps(_turn("s", 0, 2.0, 8.0)[0]) + "\n"
        + json.dumps(_turn("s", 0, 2.0, 8.0)[1]) + "\n"
        + json.dumps(_turn("s", 1, 10.0, 12.0)[0]) + "\n"
        + json.dumps(_turn("s", 1, 10.0, 12.0)[1]) + "\n"
        + json.dumps(_q_end("s", 14.0, 14.0)) + "\n"
    )
    r = mod.session_idle(mod.load_sessions(p)[0])
    assert r.inter_turn_idle_s == pytest.approx(2.0)


# ---------- CSV / main ----------


def test_main_writes_both_csvs(mod, tmp_path, capsys):
    prof = tmp_path / "profiles"
    prof.mkdir()
    _write(prof / "ses_a.jsonl", [
        _q_start("ses_a", 100.0),
        *_turn("ses_a", 0, 105.0, 115.0),
        *_turn("ses_a", 1, 120.0, 135.0),
        _q_end("ses_a", 140.0, 40.0),
    ])
    out = tmp_path / "out"
    rc = mod.main(["--profile", str(prof), "--output", str(out)])
    assert rc == 0

    gaps = list(csv.DictReader((out / "idle_gaps.csv").open()))
    assert len(gaps) == 1
    assert gaps[0]["prev_step"] == "0"
    assert gaps[0]["next_step"] == "1"
    assert float(gaps[0]["idle_s"]) == pytest.approx(5.0)

    per_sess = list(csv.DictReader((out / "idle_per_session.csv").open()))
    assert len(per_sess) == 1
    assert per_sess[0]["n_turns"] == "2"
    assert float(per_sess[0]["total_idle_s"]) == pytest.approx(15.0)
    assert float(per_sess[0]["idle_pct"]) == pytest.approx(37.5)

    captured = capsys.readouterr().out
    assert "Aggregate total idle" in captured
    assert "Inter-turn gap distribution" in captured


def test_main_accepts_single_file(mod, tmp_path):
    f = tmp_path / "agg.jsonl"
    _write(f, [
        _q_start("s", 0.0),
        *_turn("s", 0, 1.0, 3.0),
        _q_end("s", 4.0, 4.0),
    ])
    out = tmp_path / "out"
    assert mod.main(["--profile", str(f), "--output", str(out)]) == 0
    assert (out / "idle_per_session.csv").exists()


def test_main_missing_profile_returns_2(mod, tmp_path, capsys):
    rc = mod.main(["--profile", str(tmp_path / "nope"), "--output", str(tmp_path / "o")])
    assert rc == 2
    assert "profile path not found" in capsys.readouterr().err


def test_main_empty_profile_returns_1(mod, tmp_path, capsys):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    rc = mod.main(["--profile", str(f), "--output", str(tmp_path / "o")])
    assert rc == 1
    assert "no sessions found" in capsys.readouterr().err
