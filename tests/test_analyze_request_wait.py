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
