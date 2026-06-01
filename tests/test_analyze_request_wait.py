"""Tests for scripts/analyze_request_wait.py.

Joins the dynamo frontend log (total request time) with per-worker
SCHED_DELAY lines (prefill/decode queue wait) by request_id, computing
the wait fraction of end-to-end. Risky bits: the frontend request_id
regex (ANSI + k=v tolerance), the per-role sum across multiple worker
logs, and the optional per-session rollup.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_request_wait.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_request_wait", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_request_wait"] = module
    spec.loader.exec_module(module)
    return module


def _frontend_line(rid: str, elapsed_ms: int, ttft_ms: float = 100.0) -> str:
    return (
        f"2026-06-01T12:00:00Z INFO http-request: "
        f"dynamo_llm::http::service::metrics: request completed "
        f"request_id={rid} model=qwen endpoint=chat_completions "
        f"request_type=stream status=success elapsed_ms={elapsed_ms} "
        f"method=POST uri=/v1/chat/completions request_id={rid} "
        f'model="qwen" input_tokens=1187 output_tokens=13 '
        f'ttft_ms="{ttft_ms}" avg_itl_ms="9.07"\n'
    )


def _sched_line(rid: str, role: str, queue_ms: float) -> str:
    return (
        f"INFO 2026-06-01 12:00:00,123 dynamo.vllm.handlers "
        f"SCHED_DELAY request_id={rid} role={role} queue_ms={queue_ms:.3f} "
        f"queued_ts=1000.000000 scheduled_ts=1000.001000\n"
    )


# ---------- frontend parse ----------


def test_parse_frontend_extracts_request_id_and_elapsed(mod, tmp_path):
    p = tmp_path / "frontend.log"
    p.write_text(_frontend_line("req-abc", 297, ttft_ms=187.48))
    fe = mod.parse_frontend(p)
    assert "req-abc" in fe
    assert fe["req-abc"].total_ms == pytest.approx(297.0)
    assert fe["req-abc"].ttft_ms == pytest.approx(187.48)


def test_parse_frontend_skips_non_completed_lines(mod, tmp_path):
    p = tmp_path / "frontend.log"
    p.write_text(
        "INFO some routing debug line request_id=ignored\n"
        + _frontend_line("req-1", 100)
    )
    fe = mod.parse_frontend(p)
    assert set(fe) == {"req-1"}


def test_parse_frontend_strips_ansi(mod, tmp_path):
    """tracing-subscriber wraps keys in SGR codes; the parser must strip
    them before matching request_id / elapsed_ms."""
    p = tmp_path / "frontend.log"
    raw = _frontend_line("req-ansi", 250)
    # Inject ANSI around the elapsed_ms key like the pretty formatter does.
    raw = raw.replace("elapsed_ms=250", "\x1b[3melapsed_ms\x1b[0m=250")
    p.write_text(raw)
    fe = mod.parse_frontend(p)
    assert "req-ansi" in fe
    assert fe["req-ansi"].total_ms == pytest.approx(250.0)


# ---------- worker SCHED_DELAY parse ----------


def test_parse_worker_waits_sums_per_role(mod, tmp_path):
    (tmp_path / "vllm-p0.log").write_text(_sched_line("req-1", "prefill", 5.0))
    (tmp_path / "vllm-d0.log").write_text(_sched_line("req-1", "decode", 40.0))
    waits = mod.parse_worker_waits(tmp_path)
    assert waits["req-1"].prefill_ms == [5.0]
    assert waits["req-1"].decode_ms == [40.0]


def test_parse_worker_waits_accumulates_across_multiple_worker_files(mod, tmp_path):
    """The fleet may have multiple prefill workers; a request that
    somehow logged on two of them sums. (Normally one, but the sum is
    defensive.)"""
    (tmp_path / "vllm-p0.log").write_text(_sched_line("req-1", "prefill", 3.0))
    (tmp_path / "vllm-p1.log").write_text(_sched_line("req-1", "prefill", 7.0))
    waits = mod.parse_worker_waits(tmp_path)
    assert sorted(waits["req-1"].prefill_ms) == [3.0, 7.0]


# ---------- join ----------


def test_join_computes_wait_fraction(mod):
    frontend = {"r1": mod.FrontendReq("r1", total_ms=200.0, ttft_ms=50.0)}
    waits = {"r1": mod.WorkerWait(prefill_ms=[10.0], decode_ms=[30.0])}
    rows = mod.join_requests(frontend, waits)
    assert len(rows) == 1
    r = rows[0]
    assert r.matched is True
    assert r.prefill_wait_ms == pytest.approx(10.0)
    assert r.decode_wait_ms == pytest.approx(30.0)
    assert r.total_wait_ms == pytest.approx(40.0)
    assert r.wait_fraction == pytest.approx(0.2)   # 40 / 200


def test_join_unmatched_request_has_none_waits(mod):
    """A frontend request with no SCHED_DELAY (patch off, or cache-only
    no-op) is kept with matched=False and zero total_wait."""
    frontend = {"r1": mod.FrontendReq("r1", total_ms=100.0, ttft_ms=None)}
    rows = mod.join_requests(frontend, {})
    r = rows[0]
    assert r.matched is False
    assert r.prefill_wait_ms is None
    assert r.decode_wait_ms is None
    assert r.total_wait_ms == pytest.approx(0.0)
    assert r.wait_fraction == pytest.approx(0.0)


def test_join_prefill_only_request(mod):
    """Only a prefill SCHED_DELAY (decode line missing) still counts as
    matched; decode_wait stays None."""
    frontend = {"r1": mod.FrontendReq("r1", total_ms=100.0, ttft_ms=None)}
    waits = {"r1": mod.WorkerWait(prefill_ms=[8.0], decode_ms=[])}
    rows = mod.join_requests(frontend, waits)
    r = rows[0]
    assert r.matched is True
    assert r.prefill_wait_ms == pytest.approx(8.0)
    assert r.decode_wait_ms is None
    assert r.total_wait_ms == pytest.approx(8.0)


def test_join_wait_fraction_none_on_zero_total(mod):
    frontend = {"r1": mod.FrontendReq("r1", total_ms=0.0, ttft_ms=None)}
    waits = {"r1": mod.WorkerWait(prefill_ms=[5.0], decode_ms=[])}
    r = mod.join_requests(frontend, waits)[0]
    assert r.wait_fraction is None


# ---------- session rollup ----------


def test_rollup_sessions_groups_and_sums(mod):
    rows = [
        mod.RequestWait("r1", total_ms=100.0, ttft_ms=None,
                        prefill_wait_ms=5.0, decode_wait_ms=15.0),   # wait 20
        mod.RequestWait("r2", total_ms=300.0, ttft_ms=None,
                        prefill_wait_ms=10.0, decode_wait_ms=20.0),  # wait 30
        mod.RequestWait("r3", total_ms=100.0, ttft_ms=None,
                        prefill_wait_ms=None, decode_wait_ms=None),  # other session
    ]
    req_to_session = {"r1": "ses_A", "r2": "ses_A", "r3": "ses_B"}
    sessions = mod.rollup_sessions(rows, req_to_session)
    by = {s.session_id: s for s in sessions}
    assert by["ses_A"].n_requests == 2
    assert by["ses_A"].total_ms == pytest.approx(400.0)      # 100 + 300
    assert by["ses_A"].total_wait_ms == pytest.approx(50.0)  # 20 + 30
    assert by["ses_A"].wait_fraction == pytest.approx(50.0 / 400.0)
    assert by["ses_B"].n_requests == 1
    assert by["ses_B"].total_wait_ms == pytest.approx(0.0)


def test_rollup_ignores_requests_without_session(mod):
    rows = [mod.RequestWait("r1", 100.0, None, 5.0, 5.0)]
    sessions = mod.rollup_sessions(rows, {})   # empty map → no sessions
    assert sessions == []


def test_load_session_map(mod, tmp_path):
    p = tmp_path / "map.csv"
    p.write_text("request_id,session_id\nr1,ses_A\nr2,ses_B\n,skip_blank\nr3,\n")
    m = mod.load_session_map(p)
    assert m == {"r1": "ses_A", "r2": "ses_B"}   # blank rid / blank sid dropped


# ---------- main end-to-end ----------


def test_main_request_level(mod, tmp_path, capsys):
    fe = tmp_path / "frontend.log"
    fe.write_text(_frontend_line("r1", 200) + _frontend_line("r2", 400))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "vllm-p0.log").write_text(
        _sched_line("r1", "prefill", 10.0) + _sched_line("r2", "prefill", 20.0)
    )
    (logs / "vllm-d0.log").write_text(
        _sched_line("r1", "decode", 30.0) + _sched_line("r2", "decode", 60.0)
    )
    out = tmp_path / "out"
    rc = mod.main(["--frontend", str(fe), "--logs", str(logs), "--output", str(out)])
    assert rc == 0

    rows = {r["request_id"]: r for r in
            csv.DictReader((out / "request_wait.csv").open())}
    assert float(rows["r1"]["wait_fraction"]) == pytest.approx(0.2)   # 40/200
    assert float(rows["r2"]["wait_fraction"]) == pytest.approx(0.2)   # 80/400
    assert rows["r1"]["matched"] == "1"
    # No session map → no session CSV.
    assert not (out / "session_wait.csv").exists()
    assert "wait_fraction" in capsys.readouterr().out


def test_main_with_session_map(mod, tmp_path):
    fe = tmp_path / "frontend.log"
    fe.write_text(_frontend_line("r1", 100) + _frontend_line("r2", 100))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "vllm-d0.log").write_text(
        _sched_line("r1", "decode", 10.0) + _sched_line("r2", "decode", 40.0)
    )
    smap = tmp_path / "map.csv"
    smap.write_text("request_id,session_id\nr1,ses_X\nr2,ses_X\n")
    out = tmp_path / "out"
    rc = mod.main(["--frontend", str(fe), "--logs", str(logs),
                   "--output", str(out), "--session-map", str(smap)])
    assert rc == 0
    sess = list(csv.DictReader((out / "session_wait.csv").open()))
    assert len(sess) == 1
    assert sess[0]["session_id"] == "ses_X"
    assert sess[0]["n_requests"] == "2"
    # total wait 10+40=50 over total 200 → 0.25
    assert float(sess[0]["wait_fraction"]) == pytest.approx(0.25)


# ---------- tail analysis ----------


def test_percentile_summary_keys_and_values(mod):
    s = mod.percentile_summary([float(i) for i in range(1, 101)])  # 1..100
    assert s["n"] == 100
    assert s["mean"] == pytest.approx(50.5)
    assert s["p50"] == pytest.approx(50.5, abs=1.0)
    assert s["p99"] == pytest.approx(99.0, abs=1.0)
    assert s["max"] == pytest.approx(100.0)
    # p99.9 key uses %g formatting -> "p99.9"
    assert "p99.9" in s


def test_percentile_summary_empty_is_none(mod):
    assert mod.percentile_summary([]) is None


def test_tail_buckets_concentration(mod):
    """20 requests: 18 fast with tiny wait, 2 slow with huge wait. The
    slowest 10% (k=2) must concentrate almost all the wait -- wait_share
    >> 0.10 -- which is exactly the 'tail is wait-dominated' signal."""
    rows = []
    # 18 fast: total 100ms, wait 5ms (fraction 0.05)
    for i in range(18):
        rows.append(mod.RequestWait(f"fast-{i}", total_ms=100.0, ttft_ms=None,
                                    prefill_wait_ms=2.0, decode_wait_ms=3.0))
    # 2 slow: total 1000ms, wait 800ms (fraction 0.8)
    for i in range(2):
        rows.append(mod.RequestWait(f"slow-{i}", total_ms=1000.0, ttft_ms=None,
                                    prefill_wait_ms=300.0, decode_wait_ms=500.0))
    buckets = {b.frac: b for b in mod.tail_buckets(rows, fracs=(0.10,))}
    b = buckets[0.10]
    assert b.k == 2                              # ceil(0.10 * 20)
    assert b.mean_total_ms == pytest.approx(1000.0)
    assert b.mean_wait_fraction == pytest.approx(0.8)
    # total wait = 18*5 + 2*800 = 90 + 1600 = 1690; tail holds 1600.
    assert b.wait_share == pytest.approx(1600.0 / 1690.0)
    assert b.wait_share > 0.90                   # 10% of reqs hold >90% of wait


def test_tail_buckets_k_is_ceil_and_at_least_one(mod):
    rows = [mod.RequestWait(f"r{i}", total_ms=float(i + 1), ttft_ms=None,
                            prefill_wait_ms=1.0, decode_wait_ms=1.0)
            for i in range(5)]
    # frac 0.01 * 5 = 0.05 -> ceil -> 1 (never zero)
    buckets = {b.frac: b for b in mod.tail_buckets(rows, fracs=(0.01,))}
    assert buckets[0.01].k == 1
    # the single slowest is r4 (total_ms=5)
    assert buckets[0.01].mean_total_ms == pytest.approx(5.0)


def test_tail_buckets_empty_when_no_matched(mod):
    rows = [mod.RequestWait("r1", total_ms=100.0, ttft_ms=None,
                            prefill_wait_ms=None, decode_wait_ms=None)]
    assert mod.tail_buckets(rows) == []


def test_main_writes_percentile_and_tail_csvs(mod, tmp_path, capsys):
    fe = tmp_path / "frontend.log"
    logs = tmp_path / "logs"
    logs.mkdir()
    fe_lines, sched_p, sched_d = [], [], []
    # 10 fast + 2 slow so the tail is clearly wait-heavy.
    for i in range(10):
        fe_lines.append(_frontend_line(f"f{i}", 100))
        sched_p.append(_sched_line(f"f{i}", "prefill", 1.0))
        sched_d.append(_sched_line(f"f{i}", "decode", 4.0))
    for i in range(2):
        fe_lines.append(_frontend_line(f"s{i}", 1000))
        sched_p.append(_sched_line(f"s{i}", "prefill", 300.0))
        sched_d.append(_sched_line(f"s{i}", "decode", 500.0))
    fe.write_text("".join(fe_lines))
    (logs / "vllm-p0.log").write_text("".join(sched_p))
    (logs / "vllm-d0.log").write_text("".join(sched_d))

    out = tmp_path / "out"
    rc = mod.main(["--frontend", str(fe), "--logs", str(logs), "--output", str(out)])
    assert rc == 0

    pct = {r["metric"]: r for r in csv.DictReader((out / "wait_percentiles.csv").open())}
    assert set(pct) == {"wait_fraction", "total_wait_ms", "total_ms"}
    assert pct["total_ms"]["max"] == "1000.0000"

    tail = list(csv.DictReader((out / "wait_tail.csv").open()))
    by_frac = {r["tail_frac"]: r for r in tail}
    # slowest 10% (k=ceil(0.1*12)=2) are the two slow reqs → wait_share high.
    assert by_frac["0.1"]["k_requests"] == "2"
    assert float(by_frac["0.1"]["mean_wait_fraction"]) == pytest.approx(0.8)
    assert float(by_frac["0.1"]["wait_share"]) > 0.9

    captured = capsys.readouterr().out
    assert "Tail (slowest X% by e2e)" in captured
    assert "wait_share" in captured


def test_main_missing_frontend_returns_2(mod, tmp_path):
    logs = tmp_path / "logs"; logs.mkdir()
    rc = mod.main(["--frontend", str(tmp_path / "nope.log"),
                   "--logs", str(logs), "--output", str(tmp_path / "o")])
    assert rc == 2


def test_main_no_completed_lines_returns_1(mod, tmp_path, capsys):
    fe = tmp_path / "frontend.log"
    fe.write_text("INFO nothing useful here\n")
    logs = tmp_path / "logs"; logs.mkdir()
    rc = mod.main(["--frontend", str(fe), "--logs", str(logs),
                   "--output", str(tmp_path / "o")])
    assert rc == 1
    assert "no 'request completed' lines" in capsys.readouterr().err
