"""Tests for scripts/compare_prefill_compute.py.

A/B comparison of queue-corrected TTFT between a baseline and a KVBM
run (the only per-request onboard-cost handle in this dynamo tag).
Key risks: frontend/SCHED parsing reuse, queue summing across
prefill+decode records, percentile/delta table, missing-input exits.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_prefill_compute.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("compare_prefill_compute", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_prefill_compute"] = module
    spec.loader.exec_module(module)
    return module


def _frontend_line(rid, elapsed_ms, ttft_ms):
    return (f"2026-07-15T00:00:00Z INFO request completed "
            f"request_id={rid} elapsed_ms={elapsed_ms} ttft_ms={ttft_ms}\n")


def _sched_line(rid, role, queue_ms):
    return f"SCHED_DELAY request_id={rid} role={role} queue_ms={queue_ms} queued_ts=0 scheduled_ts=0\n"


def _write_run(dirpath: Path, reqs: list[tuple[str, float, float, float]]):
    """reqs: (rid, elapsed, ttft, queue). Returns (frontend, logs_dir)."""
    frontend = dirpath / "frontend.log"
    logs = dirpath / "logs"; logs.mkdir(parents=True, exist_ok=True)
    frontend.write_text("".join(
        _frontend_line(r, e, t) for r, e, t, _ in reqs))
    (logs / "vllm-d0.log").write_text("".join(
        _sched_line(r, "decode", q) for r, _, _, q in reqs))
    return frontend, logs


def test_load_run_computes_compute(mod, tmp_path):
    fe, logs = _write_run(tmp_path, [("r1", 1000, 300.0, 100.0),
                                     ("r2", 2000, 500.0, 250.0)])
    rs = mod.load_run("x", fe, logs)
    assert rs.n == 2
    assert sorted(rs.compute) == [200.0, 250.0]


def test_load_run_sums_prefill_and_decode_queue(mod, tmp_path):
    fe = tmp_path / "frontend.log"
    logs = tmp_path / "logs"; logs.mkdir()
    fe.write_text(_frontend_line("r1", 1000, 400.0))
    (logs / "vllm-p0.log").write_text(_sched_line("r1", "prefill", 100.0))
    (logs / "vllm-d0.log").write_text(_sched_line("r1", "decode", 50.0))
    rs = mod.load_run("x", fe, logs)
    assert rs.compute == [250.0]           # 400 - (100+50)


def test_load_run_no_queue_join_still_counts_ttft(mod, tmp_path):
    fe = tmp_path / "frontend.log"
    logs = tmp_path / "logs"; logs.mkdir()
    fe.write_text(_frontend_line("r1", 1000, 400.0))
    rs = mod.load_run("x", fe, logs)
    assert rs.n == 1
    assert rs.ttft == [400.0]
    assert rs.compute == []


def test_build_rows_delta(mod, tmp_path):
    d1 = tmp_path / "a"; d1.mkdir()
    d2 = tmp_path / "b"; d2.mkdir()
    fe1, l1 = _write_run(d1, [("r1", 1000, 300.0, 100.0)])   # compute 200
    fe2, l2 = _write_run(d2, [("k1", 1000, 380.0, 100.0)])   # compute 280
    base = mod.load_run("baseline", fe1, l1)
    kvbm = mod.load_run("kvbm", fe2, l2)
    rows = mod.build_rows(base, kvbm)
    mean = next(r for r in rows
                if r["metric"] == "prefill_compute_ms" and r["percentile"] == "mean")
    assert mean["delta"] == pytest.approx(80.0)   # onboard-cost estimate


def test_main_end_to_end_csv(mod, tmp_path, capsys):
    d1 = tmp_path / "a"; d1.mkdir()
    d2 = tmp_path / "b"; d2.mkdir()
    fe1, l1 = _write_run(d1, [("r1", 1000, 300.0, 100.0), ("r2", 1000, 320.0, 100.0)])
    fe2, l2 = _write_run(d2, [("k1", 1000, 400.0, 100.0), ("k2", 1000, 420.0, 100.0)])
    out = tmp_path / "cmp"
    rc = mod.main(["--baseline-frontend", str(fe1), "--baseline-logs", str(l1),
                   "--kvbm-frontend", str(fe2), "--kvbm-logs", str(l2),
                   "--out", str(out)])
    assert rc == 0
    rows = list(csv.DictReader((out / "compare_prefill_compute.csv").open()))
    metrics = {r["metric"] for r in rows}
    assert metrics == {"ttft_ms", "prefill_compute_ms"}
    assert "onboard-cost estimate" in capsys.readouterr().out


def test_main_missing_frontend_returns_2(mod, tmp_path, capsys):
    rc = mod.main(["--baseline-frontend", str(tmp_path / "no.log"),
                   "--baseline-logs", str(tmp_path),
                   "--kvbm-frontend", str(tmp_path / "no2.log"),
                   "--kvbm-logs", str(tmp_path)])
    assert rc == 2
    assert "not found" in capsys.readouterr().err


def test_main_empty_run_returns_2(mod, tmp_path, capsys):
    fe1 = tmp_path / "a.log"; fe1.write_text("")
    fe2 = tmp_path / "b.log"; fe2.write_text("")
    rc = mod.main(["--baseline-frontend", str(fe1), "--baseline-logs", str(tmp_path),
                   "--kvbm-frontend", str(fe2), "--kvbm-logs", str(tmp_path)])
    assert rc == 2
