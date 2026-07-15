"""Tests for scripts/analyze_turn_scheduling.py.

Joins opencode profile turns with worker SCHED_DELAY records. Key risks:
the exact-vs-timestamp mode auto-detection, greedy timestamp matching
(tolerance window, 1:1 assignment, ambiguity accounting), ISL band
rejection, preceding-tool adjacency keying, and the CSV/session-map
output contracts.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_turn_scheduling.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_turn_scheduling", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_turn_scheduling"] = module
    spec.loader.exec_module(module)
    return module


# ---------- fixtures ----------


def _llm_end(step, *, request_id=None, recv_ts=None, elapsed_s=None,
             inp=100, out=50, cache_read=0, duration_s=1.0):
    ev = {
        "ev": "llm.end", "ts": recv_ts or 0.0, "step": step,
        "duration_s": duration_s,
        "tokens": {"input": inp, "output": out, "cache": {"read": cache_read}},
        "dynamo": {},
    }
    if request_id is not None:
        ev["request_id"] = request_id
    if recv_ts is not None:
        ev["dynamo"]["request_received_unix_s"] = recv_ts
    if elapsed_s is not None:
        ev["dynamo"]["elapsed_s"] = elapsed_s
    return ev


def _tool_end(step, name, duration_s=0.5):
    return {"ev": "tool.end", "ts": 0.0, "step": step, "name": name,
            "callID": f"c{step}", "duration_s": duration_s, "ok": True}


def _write_profile(dirpath: Path, session_id: str, events: list[dict]) -> None:
    (dirpath / f"{session_id}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events), encoding="utf-8")


def _sched_line(rid, role, queue_ms, queued_ts=None):
    line = f"SCHED_DELAY request_id={rid} role={role} queue_ms={queue_ms}"
    if queued_ts is not None:
        line += f" queued_ts={queued_ts} scheduled_ts={queued_ts + queue_ms / 1000.0}"
    return line + "\n"


def _write_worker_log(dirpath: Path, lines: list[str], name="vllm-p0.log") -> None:
    (dirpath / name).write_text("".join(lines), encoding="utf-8")


# ---------- load_turns ----------


def test_load_turns_adjacency(mod, tmp_path):
    _write_profile(tmp_path, "ses1", [
        _llm_end(1, recv_ts=100.0),
        _tool_end(1, "bash"),
        _tool_end(1, "read"),
        _llm_end(2, recv_ts=110.0),
    ])
    turns = mod.load_turns(tmp_path)
    assert len(turns) == 2
    t1, t2 = turns
    assert t1.prev_key == "(none)"          # first turn
    assert sorted(t2.prev_tools) == ["bash", "read"]
    assert t2.prev_key == "bash+read"


def test_load_turns_effective_input_includes_cache(mod, tmp_path):
    _write_profile(tmp_path, "ses1", [_llm_end(1, inp=100, cache_read=900)])
    (t,) = mod.load_turns(tmp_path)
    assert t.effective_input == 1000


def test_load_turns_classic_usage_shape(mod, tmp_path):
    ev = {"ev": "llm.end", "ts": 0, "step": 1, "duration_s": 1.0,
          "tokens": {"prompt_tokens": 42, "completion_tokens": 7}}
    _write_profile(tmp_path, "ses1", [ev])
    (t,) = mod.load_turns(tmp_path)
    assert t.input_tokens == 42
    assert t.output_tokens == 7


# ---------- load_sched ----------


def test_load_sched_prefill_decode_pair(mod, tmp_path):
    _write_worker_log(tmp_path, [
        _sched_line("rid-1", "prefill", 12.5, queued_ts=100.0),
        _sched_line("rid-1", "decode", 3.0, queued_ts=100.5),
    ])
    sched = mod.load_sched(tmp_path)
    rec = sched["rid-1"]
    assert rec.prefill_queue_ms == 12.5
    assert rec.decode_queue_ms == 3.0
    assert rec.total_queue_ms == 15.5
    assert rec.anchor_ts == 100.0           # prefill wins


def test_load_sched_without_queued_ts(mod, tmp_path):
    # older scheduling-log format: no queued_ts field
    _write_worker_log(tmp_path, ["SCHED_DELAY request_id=r1 role=decode queue_ms=5.0\n"])
    rec = mod.load_sched(tmp_path)["r1"]
    assert rec.decode_queue_ms == 5.0
    assert rec.anchor_ts is None


def test_load_sched_only_reads_vllm_glob(mod, tmp_path):
    _write_worker_log(tmp_path, [_sched_line("r1", "prefill", 1.0)], name="other.log")
    assert mod.load_sched(tmp_path) == {}


# ---------- exact join ----------


def test_exact_join(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "ses1", [
        _llm_end(1, request_id="rid-A", elapsed_s=2.0),
        _llm_end(2, request_id="rid-MISSING"),
    ])
    _write_worker_log(logs, [_sched_line("rid-A", "prefill", 100.0, queued_ts=50.0)])
    turns = mod.load_turns(prof)
    sched = mod.load_sched(logs)
    res = mod.join_exact(turns, sched)
    assert res.mode == "exact"
    assert len(res.matched) == 1
    assert res.matched[0][1] == "rid-A"
    assert res.unmatched_turns == 1


# ---------- timestamp join ----------


def test_timestamp_join_within_tolerance(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "ses1", [_llm_end(1, recv_ts=1000.0)])
    _write_worker_log(logs, [_sched_line("r1", "prefill", 10.0, queued_ts=1000.1)])
    res = mod.join_timestamp(mod.load_turns(prof), mod.load_sched(logs), 0.25)
    assert len(res.matched) == 1
    assert res.matched[0][1] == "r1"
    assert res.ambiguous == 0
    assert res.dt_abs[0] == pytest.approx(0.1)


def test_timestamp_join_outside_tolerance_unmatched(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "ses1", [_llm_end(1, recv_ts=1000.0)])
    _write_worker_log(logs, [_sched_line("r1", "prefill", 10.0, queued_ts=1001.0)])
    res = mod.join_timestamp(mod.load_turns(prof), mod.load_sched(logs), 0.25)
    assert res.matched == []
    assert res.unmatched_turns == 1


def test_timestamp_join_ambiguous_greedy_picks_nearest(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "ses1", [_llm_end(1, recv_ts=1000.0)])
    _write_worker_log(logs, [
        _sched_line("near", "prefill", 10.0, queued_ts=1000.05),
        _sched_line("far", "prefill", 10.0, queued_ts=1000.20),
    ])
    res = mod.join_timestamp(mod.load_turns(prof), mod.load_sched(logs), 0.25)
    assert len(res.matched) == 1
    assert res.matched[0][1] == "near"
    assert res.ambiguous == 1               # one turn had 2 candidates


def test_timestamp_join_one_to_one_assignment(mod, tmp_path):
    # two turns, two sched records: each rid used once, nearest-first
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "ses1", [
        _llm_end(1, recv_ts=1000.0),
        _llm_end(2, recv_ts=1000.1),
    ])
    _write_worker_log(logs, [
        _sched_line("rA", "prefill", 1.0, queued_ts=1000.0),
        _sched_line("rB", "prefill", 1.0, queued_ts=1000.1),
    ])
    res = mod.join_timestamp(mod.load_turns(prof), mod.load_sched(logs), 0.25)
    pairs = {(t.step, rid) for t, rid, _ in res.matched}
    assert pairs == {(1, "rA"), (2, "rB")}


def test_timestamp_join_isl_band_rejects(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "ses1", [_llm_end(1, recv_ts=1000.0, inp=100)])
    _write_worker_log(logs, [_sched_line("r1", "prefill", 10.0, queued_ts=1000.0)])
    turns = mod.load_turns(prof)
    sched = mod.load_sched(logs)
    # engine ISL 100 vs profile 100 -> within band
    ok = mod.join_timestamp(turns, sched, 0.25, isl={"r1": 150}, isl_band=512)
    assert len(ok.matched) == 1
    # engine ISL wildly off -> rejected
    bad = mod.join_timestamp(turns, sched, 0.25, isl={"r1": 5000}, isl_band=512)
    assert bad.matched == []


# ---------- aggregation ----------


def test_by_tool_rows_small_share(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "ses1", [
        _llm_end(1, request_id="r1", out=10, elapsed_s=1.0),   # small
        _tool_end(1, "bash"),
        _llm_end(2, request_id="r2", out=500, elapsed_s=1.0),  # large, prev=bash
        _tool_end(2, "bash"),
        _llm_end(3, request_id="r3", out=20, elapsed_s=1.0),   # small, prev=bash
    ])
    _write_worker_log(logs, [
        _sched_line("r1", "prefill", 100.0, queued_ts=1.0),
        _sched_line("r2", "prefill", 200.0, queued_ts=2.0),
        _sched_line("r3", "prefill", 300.0, queued_ts=3.0),
    ])
    res = mod.join_exact(mod.load_turns(prof), mod.load_sched(logs))
    rows = mod.by_tool_rows(res, small_tokens=64)
    by_key = {r["prev_tools"]: r for r in rows}
    assert set(by_key) == {"(none)", "bash"}
    bash = by_key["bash"]
    assert bash["count"] == 2
    assert bash["small_share"] == pytest.approx(0.5)
    # queue_share = total_queue_ms / (elapsed_s*1000): r2=0.2, r3=0.3
    assert bash["queue_share_small_p50"] == pytest.approx(0.3)
    assert bash["queue_share_large_p50"] == pytest.approx(0.2)


# ---------- away_s + cache_hit_ratio (KV-eviction cost analysis) ----------


def _llm_start(step, ts):
    return {"ev": "llm.start", "ts": ts, "step": step}


def test_away_s_from_llm_start_end_gap(mod, tmp_path):
    _write_profile(tmp_path, "s", [
        _llm_start(1, 100.0),
        {**_llm_end(1), "ts": 101.0},
        _llm_start(2, 131.0),                 # 30s off-GPU
        {**_llm_end(2), "ts": 132.0},
    ])
    t1, t2 = mod.load_turns(tmp_path)
    assert t1.away_s is None                  # first turn has no predecessor
    assert t2.away_s == pytest.approx(30.0)


def test_away_s_negative_gap_dropped(mod, tmp_path):
    # clock weirdness: start before previous end -> away_s stays None
    _write_profile(tmp_path, "s", [
        _llm_start(1, 100.0), {**_llm_end(1), "ts": 105.0},
        _llm_start(2, 104.0), {**_llm_end(2), "ts": 106.0},
    ])
    _, t2 = mod.load_turns(tmp_path)
    assert t2.away_s is None


def test_cache_hit_ratio(mod, tmp_path):
    _write_profile(tmp_path, "s", [_llm_end(1, inp=200, cache_read=800)])
    (t,) = mod.load_turns(tmp_path)
    assert t.cache_hit_ratio == pytest.approx(0.8)


def test_cache_hit_ratio_none_without_tokens(mod):
    t = mod.TurnRec(session_id="s", step=1)
    assert t.cache_hit_ratio is None


def test_away_cache_rows_bucketing(mod):
    def turn(away, hit, isl=1000):
        cache = int(isl * hit)
        return mod.TurnRec(session_id="s", step=1, away_s=away,
                           input_tokens=isl - cache, cache_read=cache)
    turns = [turn(0.5, 0.9), turn(2.0, 0.8), turn(30.0, 0.4), turn(120.0, 0.1),
             mod.TurnRec(session_id="s", step=1)]   # no away/cache -> excluded
    rows = mod.away_cache_rows(turns)
    by = {r["away_bucket"]: r for r in rows}
    assert by["<1s"]["count"] == 1
    assert by["<1s"]["cache_hit_p50"] == pytest.approx(0.9)
    assert by["15-60s"]["cache_hit_p50"] == pytest.approx(0.4)
    assert by[">60s"]["cache_hit_p50"] == pytest.approx(0.1)
    assert by["5-15s"]["count"] == 0


def test_away_cache_correlation_negative(mod):
    def turn(away, hit, isl=1000):
        cache = int(isl * hit)
        return mod.TurnRec(session_id="s", step=1, away_s=away,
                           input_tokens=isl - cache, cache_read=cache)
    turns = [turn(a, 0.9 - 0.005 * a) for a in (1, 10, 50, 100)]
    r, n = mod.away_cache_correlation(turns)
    assert n == 4
    assert r < -0.99


def test_away_cache_correlation_insufficient(mod):
    r, n = mod.away_cache_correlation([mod.TurnRec(session_id="s", step=1)])
    assert n == 0
    import math
    assert math.isnan(r)


def test_main_profile_only_when_no_sched(mod, tmp_path, capsys):
    # empty logs dir: away/cache analysis still produced, rc 0
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "s", [
        _llm_start(1, 100.0), {**_llm_end(1, inp=100, cache_read=900), "ts": 101.0},
        _llm_start(2, 111.0), {**_llm_end(2, inp=500, cache_read=500), "ts": 112.0},
    ])
    out = tmp_path / "o"
    rc = mod.main(["--profiles", str(prof), "--logs", str(logs), "--out", str(out)])
    assert rc == 0
    assert (out / "away_cache.csv").is_file()
    assert not (out / "turn_sched.csv").exists()
    err = capsys.readouterr().err
    assert "skipping" in err


def test_turn_csv_has_away_and_cache_columns(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "s", [
        _llm_start(1, 100.0),
        {**_llm_end(1, request_id="r1", inp=100, cache_read=300), "ts": 101.0},
    ])
    _write_worker_log(logs, [_sched_line("r1", "decode", 10.0, queued_ts=1.0)])
    out = tmp_path / "o"
    mod.main(["--profiles", str(prof), "--logs", str(logs), "--out", str(out)])
    (row,) = list(csv.DictReader((out / "turn_sched.csv").open()))
    assert row["cache_read"] == "300"
    assert float(row["cache_hit_ratio"]) == pytest.approx(0.75)
    assert row["away_s"] == ""                # first turn


# ---------- queue_share denominator fallback ----------


def test_denom_prefers_elapsed(mod):
    t = mod.TurnRec(session_id="s", step=1, elapsed_s=2.0, llm_wall_s=1.0)
    assert mod._denom_ms(t) == 2000.0


def test_denom_falls_back_to_llm_wall(mod):
    t = mod.TurnRec(session_id="s", step=1, elapsed_s=None, llm_wall_s=1.5)
    assert mod._denom_ms(t) == 1500.0


def test_denom_none_when_neither(mod):
    t = mod.TurnRec(session_id="s", step=1, elapsed_s=None, llm_wall_s=None)
    assert mod._denom_ms(t) is None


def test_by_tool_share_uses_fallback_when_no_elapsed(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    # no dynamo.elapsed_s; llm_wall 1.0s; decode queue 100ms -> share 0.1
    _write_profile(prof, "s", [_llm_end(1, request_id="r1", out=10,
                                        elapsed_s=None, duration_s=1.0)])
    _write_worker_log(logs, [_sched_line("r1", "decode", 100.0, queued_ts=1.0)])
    res = mod.join_exact(mod.load_turns(prof), mod.load_sched(logs))
    rows = mod.by_tool_rows(res, small_tokens=64)
    assert rows[0]["queue_share_p50"] == pytest.approx(0.1)


def test_turn_csv_records_share_basis(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "s", [
        _llm_end(1, request_id="rE", out=10, elapsed_s=2.0, duration_s=1.0),
        _llm_end(2, request_id="rW", out=10, elapsed_s=None, duration_s=1.0),
    ])
    _write_worker_log(logs, [
        _sched_line("rE", "decode", 10.0, queued_ts=1.0),
        _sched_line("rW", "decode", 10.0, queued_ts=2.0),
    ])
    out = tmp_path / "o"
    mod.main(["--profiles", str(prof), "--logs", str(logs), "--out", str(out)])
    rows = {r["request_id"]: r for r in csv.DictReader((out / "turn_sched.csv").open())}
    assert rows["rE"]["queue_share_basis"] == "elapsed_s"
    assert rows["rW"]["queue_share_basis"] == "llm_wall_s"


# ---------- main() ----------


def _setup_run(tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "sesX", [
        _llm_end(1, request_id="rid-1", out=5, elapsed_s=2.0),
        _tool_end(1, "read"),
        _llm_end(2, request_id="rid-2", out=100, elapsed_s=3.0),
    ])
    _write_worker_log(logs, [
        _sched_line("rid-1", "prefill", 50.0, queued_ts=10.0),
        _sched_line("rid-1", "decode", 5.0, queued_ts=10.5),
        _sched_line("rid-2", "prefill", 80.0, queued_ts=20.0),
    ])
    return prof, logs


def test_main_writes_outputs(mod, tmp_path, capsys):
    prof, logs = _setup_run(tmp_path)
    out = tmp_path / "out"
    rc = mod.main(["--profiles", str(prof), "--logs", str(logs), "--out", str(out)])
    assert rc == 0
    rows = list(csv.DictReader((out / "turn_sched.csv").open()))
    assert len(rows) == 2
    r1 = next(r for r in rows if r["request_id"] == "rid-1")
    assert r1["match"] == "exact"
    assert float(r1["total_queue_ms"]) == pytest.approx(55.0)
    assert float(r1["queue_share"]) == pytest.approx(55.0 / 2000.0)
    by_tool = list(csv.DictReader((out / "by_tool.csv").open()))
    assert {r["prev_tools"] for r in by_tool} == {"(none)", "read"}
    assert "join mode: exact" in capsys.readouterr().out


def test_main_emits_session_map(mod, tmp_path):
    prof, logs = _setup_run(tmp_path)
    smap = tmp_path / "req_to_session.csv"
    rc = mod.main(["--profiles", str(prof), "--logs", str(logs),
                   "--out", str(tmp_path / "o"), "--emit-session-map", str(smap)])
    assert rc == 0
    rows = list(csv.DictReader(smap.open()))
    assert {(r["request_id"], r["session_id"]) for r in rows} == {
        ("rid-1", "sesX"), ("rid-2", "sesX")}


def test_main_missing_profiles_returns_2(mod, tmp_path, capsys):
    rc = mod.main(["--profiles", str(tmp_path / "nope"), "--logs", str(tmp_path)])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_main_no_turns_returns_2(mod, tmp_path, capsys):
    # profiles dir exists but has no parseable llm.start/llm.end/tool.end
    # events -> load_turns() is empty -> hard error (distinct from the
    # no-sched-records case, which now degrades gracefully to rc 0; see
    # test_main_profile_only_when_no_sched).
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    (prof / "s.jsonl").write_text("", encoding="utf-8")
    rc = mod.main(["--profiles", str(prof), "--logs", str(logs)])
    assert rc == 2
    assert "no turns parsed" in capsys.readouterr().err
