"""Tests for scripts/arm/e6_kv_capacity.py.

No network, no GPU. Loaded via importlib (script, not a package module),
matching this repo's convention for scripts/ tests (see
tests/test_e3_compare_runs.py).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "arm" / "e6_kv_capacity.py"
_E4_PATH = _REPO_ROOT / "scripts" / "arm" / "e4_prefill_decode.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e6():
    return _load_module("e6_kv_capacity", _SCRIPT_PATH)


@pytest.fixture(scope="module")
def e4():
    return _load_module("e4_prefill_decode", _E4_PATH)


def _line(rid, elapsed, ttft, out, input_tokens=None, ts=None):
    parts = [
        f'"request_id":"{rid}"',
        f'"elapsed_ms":{elapsed}',
        f'"ttft_ms":{ttft}',
        f'"output_tokens":{out}',
    ]
    if input_tokens is not None:
        parts.append(f'"input_tokens":{input_tokens}')
    body = "request completed {" + ",".join(parts) + "}"
    prefix = f"{ts} " if ts else ""
    return prefix + body + "\n"


# ---------- parse_frontend ----------


def test_parse_frontend_extracts_input_tokens(e6, e4, tmp_path):
    p = tmp_path / "frontend.log"
    p.write_text(_line("r1", 100, 20, 5, input_tokens=123))
    rows = e6.parse_frontend(e4, p)
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 123
    assert rows[0]["elapsed_ms"] == 100.0
    assert rows[0]["ttft_ms"] == 20.0
    assert rows[0]["output_tokens"] == 5


def test_parse_frontend_missing_input_tokens_is_none(e6, e4, tmp_path):
    p = tmp_path / "frontend.log"
    p.write_text(_line("r1", 100, 20, 5))
    rows = e6.parse_frontend(e4, p)
    assert len(rows) == 1
    assert rows[0]["input_tokens"] is None


def test_parse_frontend_drops_lines_missing_required_fields(e6, e4, tmp_path):
    p = tmp_path / "frontend.log"
    # no "request completed" marker at all -> ignored
    p.write_text('some unrelated line with "elapsed_ms":100\n')
    rows = e6.parse_frontend(e4, p)
    assert rows == []


def test_parse_frontend_last_write_wins(e6, e4, tmp_path):
    p = tmp_path / "frontend.log"
    p.write_text(
        _line("dup", 100, 20, 5, input_tokens=1)
        + _line("dup", 200, 40, 10, input_tokens=2)
    )
    rows = e6.parse_frontend(e4, p)
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 2
    assert rows[0]["elapsed_ms"] == 200.0


def test_parse_frontend_ansi_coded_line(e6, e4, tmp_path):
    p = tmp_path / "frontend.log"
    raw = _line("r1", 100, 20, 5, input_tokens=7).rstrip("\n")
    ansi = f"\x1b[32m{raw}\x1b[0m\n"
    p.write_text(ansi)
    rows = e6.parse_frontend(e4, p)
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == 7


# ---------- detect_host_kv_metric ----------


def _metrics_line(metrics: dict, ok=True, ts=1.0, role="decode"):
    return json.dumps({"ok": ok, "ts": ts, "role": role, "metrics": metrics}) + "\n"


def test_detect_host_kv_metric_prefers_cpu_local(e6, tmp_path):
    p = tmp_path / "vllm_metrics.ndjson"
    p.write_text(_metrics_line({
        "lmcache:remote_usage_bytes": [{"value": 1.0}],
        "lmcache:local_cpu_usage_bytes": [{"value": 2.0}],
    }))
    assert e6.detect_host_kv_metric(p) == "lmcache:local_cpu_usage_bytes"


def test_detect_host_kv_metric_generic_match(e6, tmp_path):
    p = tmp_path / "vllm_metrics.ndjson"
    p.write_text(_metrics_line({
        "lmcache:remote_usage_bytes": [{"value": 1.0}],
    }))
    assert e6.detect_host_kv_metric(p) == "lmcache:remote_usage_bytes"


def test_detect_host_kv_metric_none_when_absent(e6, tmp_path):
    p = tmp_path / "vllm_metrics.ndjson"
    p.write_text(_metrics_line({
        "vllm:kv_cache_usage_perc": [{"value": 0.5}],
    }))
    assert e6.detect_host_kv_metric(p) is None


def test_detect_host_kv_metric_ignores_non_usage_names(e6, tmp_path):
    p = tmp_path / "vllm_metrics.ndjson"
    p.write_text(_metrics_line({
        "lmcache:num_requests_total": [{"value": 3.0}],
    }))
    assert e6.detect_host_kv_metric(p) is None


# ---------- kv_series ----------


def test_kv_series_hbm_fraction_times_capacity(e6, tmp_path):
    p = tmp_path / "vllm_metrics.ndjson"
    p.write_text(_metrics_line({
        "vllm:kv_cache_usage_perc": [{"value": 0.5}],
    }, ts=1.0))
    series, unit = e6.kv_series(p, None, hbm_kv_gib=24.0, host_metric=None)
    assert unit == "GiB"
    assert series == [(1.0, 12.0)]


def test_kv_series_host_bytes_to_gib_summed(e6, tmp_path):
    p = tmp_path / "vllm_metrics.ndjson"
    gib_bytes = 2 * (2 ** 30)
    p.write_text(_metrics_line({
        "lmcache:local_cpu_usage_bytes": [
            {"value": float(gib_bytes)}, {"value": float(gib_bytes)},
        ],
    }, ts=1.0))
    series, unit = e6.kv_series(p, None, hbm_kv_gib=None,
                                host_metric="lmcache:local_cpu_usage_bytes")
    assert unit == "GiB"
    assert len(series) == 1
    ts, total = series[0]
    assert ts == 1.0
    assert total == pytest.approx(4.0)


def test_kv_series_hbm_and_host_combined(e6, tmp_path):
    p = tmp_path / "vllm_metrics.ndjson"
    gib_bytes = float(1 * (2 ** 30))
    p.write_text(_metrics_line({
        "vllm:kv_cache_usage_perc": [{"value": 0.5}],
        "lmcache:local_cpu_usage_bytes": [{"value": gib_bytes}],
    }, ts=1.0))
    series, unit = e6.kv_series(p, None, hbm_kv_gib=10.0,
                                host_metric="lmcache:local_cpu_usage_bytes")
    assert unit == "GiB"
    assert series == [(1.0, pytest.approx(5.0 + 1.0))]


def test_kv_series_unit_is_fraction_when_no_capacity_no_host(e6, tmp_path):
    p = tmp_path / "vllm_metrics.ndjson"
    p.write_text(_metrics_line({
        "vllm:kv_cache_usage_perc": [{"value": 0.5}],
    }, ts=1.0))
    series, unit = e6.kv_series(p, None, hbm_kv_gib=None, host_metric=None)
    assert unit == "fraction"
    assert series == [(1.0, 0.5)]


def test_kv_series_window_clips(e6, tmp_path):
    p = tmp_path / "vllm_metrics.ndjson"
    lines = (
        _metrics_line({"vllm:kv_cache_usage_perc": [{"value": 0.1}]}, ts=1.0)
        + _metrics_line({"vllm:kv_cache_usage_perc": [{"value": 0.9}]}, ts=5.0)
        + _metrics_line({"vllm:kv_cache_usage_perc": [{"value": 0.2}]}, ts=10.0)
    )
    p.write_text(lines)
    series, _unit = e6.kv_series(p, (2.0, 8.0), hbm_kv_gib=None, host_metric=None)
    assert [ts for ts, _ in series] == [5.0]


def test_kv_series_skips_ok_false_rows(e6, tmp_path):
    p = tmp_path / "vllm_metrics.ndjson"
    lines = (
        _metrics_line({"vllm:kv_cache_usage_perc": [{"value": 0.9}]}, ts=1.0, ok=False)
        + _metrics_line({"vllm:kv_cache_usage_perc": [{"value": 0.1}]}, ts=2.0, ok=True)
    )
    p.write_text(lines)
    series, _unit = e6.kv_series(p, None, hbm_kv_gib=None, host_metric=None)
    assert series == [(2.0, 0.1)]


# ---------- session_token_stats ----------


def _turn_events(sid: str, step: int, ts0: float, input_tok: int,
                 output_tok: int, cache_read: int = 0):
    return [
        json.dumps({"ev": "turn.start", "sessionID": sid, "ts": ts0, "step": step}),
        json.dumps({
            "ev": "llm.end", "sessionID": sid, "ts": ts0 + 0.5, "step": step,
            "tokens": {"input": input_tok, "output": output_tok,
                      "cache": {"read": cache_read}},
        }),
        json.dumps({"ev": "turn.end", "sessionID": sid, "ts": ts0 + 1.0,
                    "step": step, "duration_s": 1.0}),
    ]


def test_session_token_stats_start_and_end_tokens(e6, tmp_path):
    ap_mod = _load_module("analyze_profiles", _REPO_ROOT / "scripts" / "analyze_profiles.py")
    e0 = _load_module("e0_turn_characterization",
                      _REPO_ROOT / "scripts" / "arm" / "e0_turn_characterization.py")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    sid = "sesABC"
    events = []
    events += [json.dumps({"ev": "query.start", "sessionID": sid, "ts": 0.0})]
    events += _turn_events(sid, 0, 1.0, input_tok=100, output_tok=10, cache_read=5)
    events += _turn_events(sid, 1, 3.0, input_tok=50, output_tok=20, cache_read=0)
    events += [json.dumps({"ev": "query.end", "sessionID": sid, "ts": 5.0,
                           "duration_s": 5.0})]
    (profiles / f"{sid}.jsonl").write_text("\n".join(events) + "\n")

    st = e6.session_token_stats(ap_mod, e0, profiles, trace=None)
    assert st["n_sessions"] == 1
    # start = first turn's effective input = 100 + 5 = 105
    assert st["start_tokens"]["mean"] == pytest.approx(105.0)
    # end = last token-bearing turn effective input (50+0) + output (20) = 70
    assert st["end_tokens"]["mean"] == pytest.approx(70.0)
    assert st["turns_per_session"]["mean"] == pytest.approx(2.0)


def test_session_token_stats_trace_filters_sessions(e6, tmp_path):
    ap_mod = _load_module("analyze_profiles_2", _REPO_ROOT / "scripts" / "analyze_profiles.py")
    e0 = _load_module("e0_turn_characterization_2",
                      _REPO_ROOT / "scripts" / "arm" / "e0_turn_characterization.py")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    main_sid = "sesMAIN"
    sub_sid = "sesSUB"
    for sid in (main_sid, sub_sid):
        events = _turn_events(sid, 0, 1.0, input_tok=10, output_tok=1)
        (profiles / f"{sid}.jsonl").write_text("\n".join(events) + "\n")
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"session_id": main_sid}) + "\n")

    st = e6.session_token_stats(ap_mod, e0, profiles, trace=trace)
    assert st["n_sessions"] == 1


def test_session_token_stats_no_tokens_excluded_from_token_stats(e6, tmp_path):
    ap_mod = _load_module("analyze_profiles_3", _REPO_ROOT / "scripts" / "analyze_profiles.py")
    e0 = _load_module("e0_turn_characterization_3",
                      _REPO_ROOT / "scripts" / "arm" / "e0_turn_characterization.py")
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    sid = "sesNoTok"
    # turn.start/turn.end only, no llm.end -> no token-bearing turn but the
    # turn itself still counts toward turns_per_session.
    events = [
        json.dumps({"ev": "turn.start", "sessionID": sid, "ts": 1.0, "step": 0}),
        json.dumps({"ev": "turn.end", "sessionID": sid, "ts": 2.0, "step": 0,
                   "duration_s": 1.0}),
    ]
    (profiles / f"{sid}.jsonl").write_text("\n".join(events) + "\n")

    st = e6.session_token_stats(ap_mod, e0, profiles, trace=None)
    # counted toward turn counts (n_sessions/turns_per_session reflects it)...
    assert st["n_sessions"] == 1
    assert st["turns_per_session"]["mean"] == pytest.approx(1.0)
    # ...but excluded from token stats (no token-bearing turns at all)
    import math
    assert math.isnan(st["start_tokens"]["mean"])
    assert math.isnan(st["end_tokens"]["mean"])


# ---------- resolve_run ----------


def test_resolve_run_label_equals_path(e6, tmp_path):
    root = tmp_path / "runA"
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "frontend.log").write_text("")
    out = e6.resolve_run(f"mylabel={root}")
    assert out["label"] == "mylabel"
    assert out["root"] == root
    assert out["frontend"] == root / "logs" / "frontend.log"


def test_resolve_run_bare_path_label_is_dirname(e6, tmp_path):
    root = tmp_path / "runB"
    root.mkdir()
    out = e6.resolve_run(str(root))
    assert out["label"] == "runB"


def test_resolve_run_prefers_logs_subdir(e6, tmp_path):
    root = tmp_path / "runC"
    (root / "logs").mkdir(parents=True)
    (root / "logs" / "vllm_metrics.ndjson").write_text("")
    # a bare frontend.log at root should NOT be picked up once logs/ exists
    (root / "frontend.log").write_text("")
    out = e6.resolve_run(str(root))
    assert out["metrics"] == root / "logs" / "vllm_metrics.ndjson"
    assert "frontend" not in out


def test_resolve_run_bare_dir_no_logs_subdir(e6, tmp_path):
    root = tmp_path / "runD"
    root.mkdir()
    (root / "frontend.log").write_text("")
    out = e6.resolve_run(str(root))
    assert out["frontend"] == root / "frontend.log"


def test_resolve_run_profiles_dir_preferred_over_jsonl_file(e6, tmp_path):
    root = tmp_path / "runE"
    (root / "profiles").mkdir(parents=True)
    (root / "profiles.jsonl").write_text("")
    out = e6.resolve_run(str(root))
    assert out["profiles"] == root / "profiles"


def test_resolve_run_profiles_jsonl_file_used_when_no_subdir(e6, tmp_path):
    root = tmp_path / "runF"
    root.mkdir()
    (root / "profiles.jsonl").write_text("")
    out = e6.resolve_run(str(root))
    assert out["profiles"] == root / "profiles.jsonl"


def test_resolve_run_missing_path_raises(e6, tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        e6.resolve_run(str(missing))


def test_resolve_run_trace_detected(e6, tmp_path):
    root = tmp_path / "runG"
    root.mkdir()
    (root / "trace.jsonl").write_text("")
    out = e6.resolve_run(str(root))
    assert out["trace"] == root / "trace.jsonl"


# ---------- main smoke ----------


def test_main_smoke_writes_summary_and_session_tokens(e6, tmp_path):
    run_root = tmp_path / "runX"
    logs = run_root / "logs"
    logs.mkdir(parents=True)
    (logs / "frontend.log").write_text(
        _line("r1", 100, 20, 5, input_tokens=100, ts="2026-01-01T00:00:00Z")
        + _line("r2", 200, 40, 10, input_tokens=50, ts="2026-01-01T00:00:05Z")
    )
    (logs / "vllm_metrics.ndjson").write_text(
        _metrics_line({"vllm:kv_cache_usage_perc": [{"value": 0.4}],
                      "vllm:num_requests_running": [{"value": 2.0}]},
                     ts=1767225600.0, role="decode")
    )
    profiles = run_root / "profiles"
    profiles.mkdir()
    sid = "sesMain1"
    events = [json.dumps({"ev": "query.start", "sessionID": sid, "ts": 0.0})]
    events += _turn_events(sid, 0, 1.0, input_tok=100, output_tok=10)
    (profiles / f"{sid}.jsonl").write_text("\n".join(events) + "\n")

    out_dir = tmp_path / "e6_out"
    rc = e6.main([str(run_root), "--out", str(out_dir), "--no-figures"])
    assert rc == 0

    summary_csv = out_dir / "summary.csv"
    assert summary_csv.exists()
    with summary_csv.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["run"] == "runX"

    tokens_csv = out_dir / "session_tokens.csv"
    assert tokens_csv.exists()
    with tokens_csv.open() as f:
        trows = list(csv.DictReader(f))
    assert len(trows) == 1
    assert trows[0]["run"] == "runX"
    assert trows[0]["n_sessions"] == "1"

    # no figure file should exist under --no-figures
    assert not (out_dir / "fig_kv_over_time.pdf").exists()
