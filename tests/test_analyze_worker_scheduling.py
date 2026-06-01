"""Tests for scripts/analyze_worker_scheduling.py.

Parses SCHED_DELAY worker-log lines into per-request + per-(worker,role)
scheduling-delay stats. The risky bits: the regex against realistic
log lines (with logging prefixes), worker-label-from-filename, and the
per-role percentile rollup.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_worker_scheduling.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_worker_scheduling", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_worker_scheduling"] = module
    spec.loader.exec_module(module)
    return module


def _sched_line(rid: str, role: str, queue_ms: float,
                queued: float = 1000.0, scheduled: float | None = None) -> str:
    """A realistic worker-log line: a Python logging prefix then the
    SCHED_DELAY message (matching logger.info's format)."""
    if scheduled is None:
        scheduled = queued + queue_ms / 1000.0
    return (
        f"INFO 2026-06-01 12:00:00,123 dynamo.vllm.handlers "
        f"SCHED_DELAY request_id={rid} role={role} queue_ms={queue_ms:.3f} "
        f"queued_ts={queued:.6f} scheduled_ts={scheduled:.6f}\n"
    )


# ---------- regex / parsing ----------


def test_parse_single_line(mod, tmp_path):
    p = tmp_path / "vllm-d0.log"
    p.write_text(_sched_line("req-1", "decode", 12.5))
    recs = mod.load_records(p)
    assert len(recs) == 1
    r = recs[0]
    assert r.worker == "vllm-d0"          # from filename
    assert r.role == "decode"
    assert r.request_id == "req-1"
    assert r.queue_ms == pytest.approx(12.5)


def test_non_sched_lines_skipped(mod, tmp_path):
    """Ordinary worker log output must be ignored -- only SCHED_DELAY
    lines parse."""
    p = tmp_path / "vllm-p0.log"
    p.write_text(
        "INFO some unrelated startup log line\n"
        "DEBUG kv transfer params: {...}\n"
        + _sched_line("req-2", "prefill", 3.0)
        + "INFO Prefill completed for request req-2\n"
    )
    recs = mod.load_records(p)
    assert len(recs) == 1
    assert recs[0].request_id == "req-2"
    assert recs[0].role == "prefill"


def test_worker_label_from_filename(mod, tmp_path):
    """The worker comes from the log filename stem, NOT the line -- two
    files with the same role still separate by worker."""
    (tmp_path / "vllm-d0.log").write_text(_sched_line("a", "decode", 5.0))
    (tmp_path / "vllm-d1.log").write_text(_sched_line("b", "decode", 9.0))
    recs = mod.load_records(tmp_path)
    workers = {r.worker for r in recs}
    assert workers == {"vllm-d0", "vllm-d1"}


def test_directory_globs_only_vllm_logs(mod, tmp_path):
    """A non-vllm log in the same dir is ignored by the vllm-*.log glob."""
    (tmp_path / "vllm-p0.log").write_text(_sched_line("a", "prefill", 1.0))
    (tmp_path / "frontend.log").write_text(_sched_line("x", "decode", 99.0))
    recs = mod.load_records(tmp_path)
    assert len(recs) == 1
    assert recs[0].request_id == "a"


def test_malformed_sched_line_skipped(mod, tmp_path):
    """A line that contains SCHED_DELAY but lacks the full field set is
    dropped rather than crashing."""
    p = tmp_path / "vllm-d0.log"
    p.write_text(
        "INFO SCHED_DELAY request_id=onlypartial role=decode\n"   # missing queue_ms etc.
        + _sched_line("good", "decode", 7.0)
    )
    recs = mod.load_records(p)
    assert len(recs) == 1
    assert recs[0].request_id == "good"


# ---------- stats ----------


def test_stats_empty(mod):
    s = mod._stats([])
    assert s == {"n": 0, "mean": None, "p50": None,
                 "p90": None, "p99": None, "max": None}


def test_stats_basic(mod):
    s = mod._stats([10.0, 20.0, 30.0, 40.0, 50.0])
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(30.0)
    assert s["p50"] == pytest.approx(30.0)
    assert s["max"] == pytest.approx(50.0)


def test_by_worker_role_partitions(mod, tmp_path):
    """Stats keyed by (worker, role): a prefill and a decode worker are
    separate buckets even though they share request ids."""
    (tmp_path / "vllm-p0.log").write_text(
        _sched_line("r1", "prefill", 2.0) + _sched_line("r2", "prefill", 4.0)
    )
    (tmp_path / "vllm-d0.log").write_text(
        _sched_line("r1", "decode", 20.0) + _sched_line("r2", "decode", 40.0)
    )
    recs = mod.load_records(tmp_path)
    wr = mod.by_worker_role(recs)
    assert wr[("vllm-p0", "prefill")]["mean"] == pytest.approx(3.0)
    assert wr[("vllm-d0", "decode")]["mean"] == pytest.approx(30.0)


def test_by_role_aggregates_across_workers(mod, tmp_path):
    """Per-role rollup pools all workers of that role."""
    (tmp_path / "vllm-d0.log").write_text(_sched_line("r1", "decode", 10.0))
    (tmp_path / "vllm-d1.log").write_text(_sched_line("r2", "decode", 30.0))
    recs = mod.load_records(tmp_path)
    role = mod.by_role(recs)
    assert role["decode"]["n"] == 2
    assert role["decode"]["mean"] == pytest.approx(20.0)


# ---------- CSV / main ----------


def test_main_writes_csvs_and_summary(mod, tmp_path, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "vllm-p0.log").write_text(
        _sched_line("r1", "prefill", 2.5) + _sched_line("r2", "prefill", 7.5)
    )
    (logs / "vllm-d0.log").write_text(_sched_line("r1", "decode", 33.0))
    out = tmp_path / "out"
    rc = mod.main(["--logs", str(logs), "--output", str(out)])
    assert rc == 0

    per_req = list(csv.DictReader((out / "scheduling_per_request.csv").open()))
    assert len(per_req) == 3
    assert {r["role"] for r in per_req} == {"prefill", "decode"}

    by_wr = list(csv.DictReader((out / "scheduling_by_worker_role.csv").open()))
    rows = {(r["worker"], r["role"]): r for r in by_wr}
    assert rows[("vllm-p0", "prefill")]["n"] == "2"
    assert float(rows[("vllm-p0", "prefill")]["mean_ms"]) == pytest.approx(5.0)
    assert rows[("vllm-d0", "decode")]["n"] == "1"

    captured = capsys.readouterr().out
    assert "Scheduling delay by worker/role" in captured
    assert "Aggregate by role" in captured


def test_main_accepts_single_file(mod, tmp_path):
    f = tmp_path / "vllm-d0.log"
    f.write_text(_sched_line("r1", "decode", 5.0))
    out = tmp_path / "out"
    assert mod.main(["--logs", str(f), "--output", str(out)]) == 0
    assert (out / "scheduling_per_request.csv").exists()


def test_main_missing_logs_returns_2(mod, tmp_path, capsys):
    rc = mod.main(["--logs", str(tmp_path / "nope"), "--output", str(tmp_path / "o")])
    assert rc == 2
    assert "logs path not found" in capsys.readouterr().err


def test_main_no_sched_lines_returns_1(mod, tmp_path, capsys):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "vllm-d0.log").write_text("INFO just ordinary log output\n")
    rc = mod.main(["--logs", str(logs), "--output", str(tmp_path / "o")])
    assert rc == 1
    assert "no SCHED_DELAY lines found" in capsys.readouterr().err
