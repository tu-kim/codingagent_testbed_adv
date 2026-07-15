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


def test_main_no_sched_returns_2(mod, tmp_path, capsys):
    prof = tmp_path / "profiles"; prof.mkdir()
    logs = tmp_path / "logs"; logs.mkdir()
    _write_profile(prof, "s", [_llm_end(1)])
    rc = mod.main(["--profiles", str(prof), "--logs", str(logs)])
    assert rc == 2
    assert "SCHED_DELAY" in capsys.readouterr().err
