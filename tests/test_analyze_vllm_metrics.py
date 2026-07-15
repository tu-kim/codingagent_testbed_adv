"""Tests for scripts/analyze_vllm_metrics.py.

Pure file-based parsing + classification + per-series stats. No
external dependencies beyond numpy / json / csv. Histogram bucket
deltas / quantile interpolation are the trickiest bits — they get
dedicated tests with hand-computed expected values.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_vllm_metrics.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_vllm_metrics", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_vllm_metrics"] = module
    spec.loader.exec_module(module)
    return module


# ---------- fixtures ----------


def _scrape_row(ts: float, worker: str, role: str,
                metrics: dict, port: int = 21000) -> dict:
    return {
        "ts": ts,
        "interval_s": 1.0,
        "worker": worker,
        "role": role,
        "host": "127.0.0.1",
        "port": port,
        "ok": True,
        "metrics": metrics,
    }


def _g(value: float, labels: dict | None = None) -> list[dict]:
    """Single-point gauge/counter metric entry list."""
    return [{"labels": labels or {}, "value": value}]


def _bucket(le_to_count: dict, labels_extra: dict | None = None) -> list[dict]:
    """Histogram bucket entries, one per `le` upper bound."""
    out = []
    labels_extra = labels_extra or {}
    for le, cum in le_to_count.items():
        out.append({"labels": {**labels_extra, "le": str(le)}, "value": float(cum)})
    return out


def _write_ndjson(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# ---------- classify_metrics ----------


def test_classify_recognizes_histogram_family(mod):
    names = {
        "vllm:time_to_first_token_seconds_bucket",
        "vllm:time_to_first_token_seconds_count",
        "vllm:time_to_first_token_seconds_sum",
    }
    cls = mod.classify_metrics(names)
    assert cls["vllm:time_to_first_token_seconds_bucket"] == "histogram_bucket"
    assert cls["vllm:time_to_first_token_seconds_count"] == "histogram_count"
    assert cls["vllm:time_to_first_token_seconds_sum"] == "histogram_sum"


def test_classify_counter_suffix(mod):
    cls = mod.classify_metrics({"vllm:prompt_tokens_total"})
    assert cls["vllm:prompt_tokens_total"] == "counter"


def test_classify_gauge_default(mod):
    cls = mod.classify_metrics({"vllm:num_requests_running",
                                 "vllm:gpu_cache_usage_perc"})
    assert cls["vllm:num_requests_running"] == "gauge"
    assert cls["vllm:gpu_cache_usage_perc"] == "gauge"


def test_classify_count_without_matching_bucket_is_gauge(mod):
    """A lone `*_count` without a sibling `_bucket` is NOT a histogram.
    It's just a gauge that happens to end in `_count` (e.g. a custom
    metric name)."""
    cls = mod.classify_metrics({"vllm:something_count"})
    assert cls["vllm:something_count"] == "gauge"


# ---------- histogram_quantile ----------


def test_histogram_quantile_interpolates_within_bucket(mod):
    """Classic Prometheus example: 100 total observations distributed
    as 60 in [0, 0.5], 30 in (0.5, 1.0], 10 in (1.0, +Inf]. The 0.9
    quantile lands inside the (0.5, 1.0] bucket."""
    buckets = [(0.5, 60), (1.0, 90), (math.inf, 100)]
    p90 = mod.histogram_quantile(0.90, buckets)
    # target = 90, prev=(0.5, 60), bucket=(1.0, 90). Hits cum exactly at
    # upper bound -> interpolated value = 1.0
    assert p90 == pytest.approx(1.0)

    p50 = mod.histogram_quantile(0.50, buckets)
    # target = 50, falls in first bucket (0, 0.5). frac = 50/60.
    assert p50 == pytest.approx((50 / 60) * 0.5)


def test_histogram_quantile_returns_none_for_empty(mod):
    assert mod.histogram_quantile(0.5, []) is None
    assert mod.histogram_quantile(0.5, [(1.0, 0)]) is None  # total = 0


def test_histogram_quantile_inf_bucket_falls_back_to_prev_ub(mod):
    """If the quantile target lands in the +Inf overflow bucket, the
    real value is unknown — return the previous finite upper bound."""
    buckets = [(0.1, 50), (1.0, 90), (math.inf, 100)]
    # p99 target = 99, only +Inf has it -> prev_ub = 1.0
    p99 = mod.histogram_quantile(0.99, buckets)
    assert p99 == pytest.approx(1.0)


# ---------- gauge_stats ----------


def test_gauge_stats_basic(mod):
    s = mod.gauge_stats([10.0, 20.0, 30.0, 40.0, 50.0])
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(30.0)
    assert s["median"] == pytest.approx(30.0)
    assert s["max"] == pytest.approx(50.0)
    # p99 is between p90 and max; exact value depends on numpy's
    # percentile method (linear by default) but the range invariant
    # is what we care about across numpy versions.
    assert s["p90"] <= s["p99"] <= s["max"]


def test_gauge_stats_empty(mod):
    s = mod.gauge_stats([])
    assert s == {"n": 0, "mean": None, "median": None,
                 "p90": None, "p99": None, "max": None}


# ---------- counter_stats ----------


def test_counter_stats_delta_and_rate(mod):
    """Cumulative samples 100 → 250 over 30s window → delta=150, rate=5/s."""
    s = mod.counter_stats([(1000.0, 100.0), (1010.0, 180.0), (1030.0, 250.0)])
    assert s["delta"] == pytest.approx(150.0)
    assert s["rate_per_s"] == pytest.approx(150.0 / 30.0)


def test_counter_stats_clips_negative_delta_on_restart(mod):
    """If the counter resets (process restart) between scrapes, last <
    first. Clip delta to 0 (standard Prometheus rate() behaviour)."""
    s = mod.counter_stats([(0.0, 500.0), (10.0, 5.0)])
    assert s["delta"] == 0.0
    assert s["rate_per_s"] == 0.0


def test_counter_stats_single_sample_yields_none(mod):
    """A single scrape isn't enough to derive a delta — return None."""
    s = mod.counter_stats([(1.0, 100.0)])
    assert s["delta"] is None
    assert s["rate_per_s"] is None


# ---------- histogram_stats ----------


def test_histogram_stats_uses_bucket_deltas_for_quantiles(mod):
    """Cumulative buckets across two scrapes; quantiles should be over
    the WINDOW DELTA (last - first), not the absolute last sample."""
    hist = {
        "count": [(1.0, 100.0), (2.0, 200.0)],   # +100 obs in window
        "sum":   [(1.0, 50.0),  (2.0, 100.0)],   # +50 sum
        "buckets": {
            0.1: [(1.0, 50.0),  (2.0, 50.0)],    # +0
            0.5: [(1.0, 80.0),  (2.0, 140.0)],   # +60
            1.0: [(1.0, 95.0),  (2.0, 185.0)],   # +90
            math.inf: [(1.0, 100.0), (2.0, 200.0)],  # +100
        },
    }
    s = mod.histogram_stats(hist)
    assert s["delta"] == 100.0
    assert s["sum"]   == 50.0
    assert s["mean"]  == pytest.approx(0.5)   # 50 / 100

    # bucket deltas: [(0.1, 0), (0.5, 60), (1.0, 90), (inf, 100)]
    # p50: target=50, falls in (0.5, 60). prev=(0.1, 0). frac=50/60.
    # value = 0.1 + (50/60) * (0.5 - 0.1) = 0.1 + 0.333 = 0.4333
    assert s["median"] == pytest.approx(0.1 + (50/60) * (0.5 - 0.1), rel=1e-3)
    # p90: target=90, hits (1.0, 90) exactly -> 1.0
    assert s["p90"] == pytest.approx(1.0)
    # max: highest finite bucket that saw an increment -> 1.0
    assert s["max"] == pytest.approx(1.0)


def test_histogram_stats_no_observations_in_window(mod):
    """Buckets unchanged between scrapes → delta=0, no quantiles."""
    hist = {
        "count": [(1.0, 50.0), (2.0, 50.0)],
        "sum":   [(1.0, 25.0), (2.0, 25.0)],
        "buckets": {
            0.5: [(1.0, 30.0), (2.0, 30.0)],
            math.inf: [(1.0, 50.0), (2.0, 50.0)],
        },
    }
    s = mod.histogram_stats(hist)
    assert s["delta"] == 0.0
    assert s["median"] is None
    assert s["p90"] is None


def test_histogram_stats_single_sample_returns_none(mod):
    """Only one scrape -> can't compute deltas."""
    hist = {
        "count": [(1.0, 100.0)],
        "sum":   [(1.0, 50.0)],
        "buckets": {0.5: [(1.0, 60.0)]},
    }
    s = mod.histogram_stats(hist)
    assert s["delta"] is None
    assert s["mean"] is None


# ---------- collect_series ----------


def test_collect_series_partitions_by_type(mod):
    rows = [
        _scrape_row(1.0, "p0", "prefill", {
            "vllm:num_requests_running": _g(2.0),
            "vllm:prompt_tokens_total": _g(100.0),
            "vllm:time_to_first_token_seconds_bucket": _bucket(
                {0.1: 10, 0.5: 25, 1.0: 30, "+Inf": 32}),
            "vllm:time_to_first_token_seconds_count": _g(32.0),
            "vllm:time_to_first_token_seconds_sum": _g(8.5),
        }),
        _scrape_row(2.0, "p0", "prefill", {
            "vllm:num_requests_running": _g(3.0),
            "vllm:prompt_tokens_total": _g(250.0),
            "vllm:time_to_first_token_seconds_bucket": _bucket(
                {0.1: 12, 0.5: 35, 1.0: 50, "+Inf": 55}),
            "vllm:time_to_first_token_seconds_count": _g(55.0),
            "vllm:time_to_first_token_seconds_sum": _g(18.5),
        }),
    ]
    cls = mod.classify_metrics({
        "vllm:num_requests_running",
        "vllm:prompt_tokens_total",
        "vllm:time_to_first_token_seconds_bucket",
        "vllm:time_to_first_token_seconds_count",
        "vllm:time_to_first_token_seconds_sum",
    })
    g, c, h = mod.collect_series(rows, cls)

    gauge_key = ("p0", "prefill", "vllm:num_requests_running", "")
    assert g[gauge_key] == [(1.0, 2.0), (2.0, 3.0)]

    counter_key = ("p0", "prefill", "vllm:prompt_tokens_total", "")
    assert c[counter_key] == [(1.0, 100.0), (2.0, 250.0)]

    # Histogram key uses BASE metric (no `_bucket` suffix) so count/sum/buckets share it.
    hist_key = ("p0", "prefill", "vllm:time_to_first_token_seconds", "")
    assert hist_key in h
    assert h[hist_key]["count"] == [(1.0, 32.0), (2.0, 55.0)]
    assert h[hist_key]["sum"] == [(1.0, 8.5), (2.0, 18.5)]
    # +Inf bucket present
    assert math.inf in h[hist_key]["buckets"]


def test_collect_series_skips_nan_inf_gauges(mod):
    """vLLM emits +Inf for some gauges (e.g. avg_prompt_throughput
    before any request). These must not poison the percentile pool."""
    rows = [
        _scrape_row(1.0, "p0", "prefill", {
            "vllm:gpu_cache_usage_perc": [{"labels": {}, "value": float("nan")}],
        }),
        _scrape_row(2.0, "p0", "prefill", {
            "vllm:gpu_cache_usage_perc": _g(0.42),
        }),
    ]
    g, _, _ = mod.collect_series(rows, mod.classify_metrics(
        {"vllm:gpu_cache_usage_perc"}))
    key = ("p0", "prefill", "vllm:gpu_cache_usage_perc", "")
    assert g[key] == [(2.0, 0.42)]   # NaN row dropped


def test_collect_series_separates_label_sets(mod):
    """Same metric name with different non-le labels → separate series.
    Useful when vLLM splits a counter by `finished_reason` etc."""
    rows = [
        _scrape_row(1.0, "p0", "prefill", {
            "vllm:request_success_total": [
                {"labels": {"finished_reason": "stop"},   "value": 5.0},
                {"labels": {"finished_reason": "length"}, "value": 1.0},
            ],
        }),
        _scrape_row(2.0, "p0", "prefill", {
            "vllm:request_success_total": [
                {"labels": {"finished_reason": "stop"},   "value": 12.0},
                {"labels": {"finished_reason": "length"}, "value": 3.0},
            ],
        }),
    ]
    _, c, _ = mod.collect_series(rows, mod.classify_metrics(
        {"vllm:request_success_total"}))
    stop_key   = ("p0", "prefill", "vllm:request_success_total",
                  mod._label_key({"finished_reason": "stop"}))
    length_key = ("p0", "prefill", "vllm:request_success_total",
                  mod._label_key({"finished_reason": "length"}))
    assert c[stop_key] == [(1.0, 5.0), (2.0, 12.0)]
    assert c[length_key] == [(1.0, 1.0), (2.0, 3.0)]


def test_label_key_avoids_collision_on_separator_chars(mod):
    """Label values containing `,` or `=` (rare in vLLM but legal in
    Prometheus) MUST NOT collide with structurally-different label
    dicts. Regression-guard for the `a=b,c=d` vs `a=b,c=d` ambiguity."""
    collidey_value = {"a": "b,c=d"}
    structured     = {"a": "b", "c": "d"}
    assert mod._label_key(collidey_value) != mod._label_key(structured)


# ---------- role aggregation ----------


def test_aggregate_gauge_per_role_uses_per_tick_mean(mod):
    """Per-tick mean across workers in the role, then percentiles of
    THAT — not a flattened pool. Mirrors the resource analyzer's
    convention."""
    rows = [
        _scrape_row(1.0, "p0", "prefill", {"vllm:num_requests_running": _g(2.0)}),
        _scrape_row(1.0, "p1", "prefill", {"vllm:num_requests_running": _g(4.0)}),
        _scrape_row(2.0, "p0", "prefill", {"vllm:num_requests_running": _g(6.0)}),
        _scrape_row(2.0, "p1", "prefill", {"vllm:num_requests_running": _g(8.0)}),
    ]
    cls = mod.classify_metrics({"vllm:num_requests_running"})
    g, c, h = mod.collect_series(rows, cls)
    rg, rc, rh = mod.aggregate_by_role(g, c, h, cls)
    role_key = ("*", "prefill", "vllm:num_requests_running", "")
    # ts=1 mean=3, ts=2 mean=7
    assert rg[role_key] == [(1.0, 3.0), (2.0, 7.0)]


def test_aggregate_counter_sums_per_worker_deltas(mod):
    rows = [
        _scrape_row(0.0,  "p0", "prefill", {"vllm:prompt_tokens_total": _g(100.0)}),
        _scrape_row(10.0, "p0", "prefill", {"vllm:prompt_tokens_total": _g(300.0)}),
        _scrape_row(0.0,  "p1", "prefill", {"vllm:prompt_tokens_total": _g(50.0)}),
        _scrape_row(10.0, "p1", "prefill", {"vllm:prompt_tokens_total": _g(150.0)}),
    ]
    cls = mod.classify_metrics({"vllm:prompt_tokens_total"})
    g, c, h = mod.collect_series(rows, cls)
    _, rc, _ = mod.aggregate_by_role(g, c, h, cls)
    role_key = ("*", "prefill", "vllm:prompt_tokens_total", "")
    # p0 delta=200, p1 delta=100, sum=300, dur=10 -> rate=30/s
    assert rc[role_key]["delta"] == pytest.approx(300.0)
    assert rc[role_key]["rate_per_s"] == pytest.approx(30.0)


# ---------- CSV + main end-to-end ----------


def _write_profile(path: Path, *, session_id: str, start: float, end: float) -> None:
    lines = [
        {"ev": "query.start", "ts": start, "sessionID": session_id},
        {"ev": "query.end",   "ts": end,   "sessionID": session_id},
    ]
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n")


def test_main_all_points_writes_csv_and_table(mod, tmp_path, capsys):
    metrics_path = tmp_path / "vllm_metrics.ndjson"
    _write_ndjson(metrics_path, [
        _scrape_row(1.0, "p0", "prefill", {
            "vllm:num_requests_running": _g(2.0),
            "vllm:prompt_tokens_total":  _g(100.0),
        }),
        _scrape_row(2.0, "p0", "prefill", {
            "vllm:num_requests_running": _g(4.0),
            "vllm:prompt_tokens_total":  _g(300.0),
        }),
    ])
    out = tmp_path / "out"
    rc = mod.main(["--metrics", str(metrics_path), "--output", str(out)])
    assert rc == 0
    csv_path = out / "vllm_metrics_stats.csv"
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.read_text().splitlines()))
    by_metric = {(r["worker"], r["metric"]): r for r in rows}
    # gauge
    g_row = by_metric[("p0", "vllm:num_requests_running")]
    assert g_row["type"] == "gauge"
    assert float(g_row["mean"]) == pytest.approx(3.0)
    # counter
    c_row = by_metric[("p0", "vllm:prompt_tokens_total")]
    assert c_row["type"] == "counter"
    assert float(c_row["delta"]) == pytest.approx(200.0)
    assert float(c_row["rate_per_s"]) == pytest.approx(200.0 / 1.0)
    captured = capsys.readouterr().out
    assert "ALL_POINTS" in captured
    assert "vllm:num_requests_running" in captured


def test_main_session_window_filters_scrape_rows(mod, tmp_path, capsys):
    profile = tmp_path / "ses.jsonl"
    _write_profile(profile, session_id="ses_x", start=10.0, end=30.0)
    metrics_path = tmp_path / "vllm_metrics.ndjson"
    _write_ndjson(metrics_path, [
        _scrape_row(5.0,  "p0", "prefill", {"vllm:num_requests_running": _g(99.0)}),  # outside
        _scrape_row(15.0, "p0", "prefill", {"vllm:num_requests_running": _g(2.0)}),
        _scrape_row(25.0, "p0", "prefill", {"vllm:num_requests_running": _g(4.0)}),
        _scrape_row(35.0, "p0", "prefill", {"vllm:num_requests_running": _g(99.0)}),  # outside
    ])
    out = tmp_path / "out"
    rc = mod.main(["--profile", str(profile),
                   "--metrics", str(metrics_path),
                   "--output", str(out)])
    assert rc == 0
    rows = list(csv.DictReader((out / "vllm_metrics_stats.csv").read_text().splitlines()))
    g_row = next(r for r in rows if r["worker"] == "p0"
                                and r["metric"] == "vllm:num_requests_running")
    assert float(g_row["mean"]) == pytest.approx(3.0)   # mean(2, 4); 99s excluded
    assert g_row["session_id"] == "ses_x"


def test_main_drops_failed_scrapes(mod, tmp_path):
    """Rows with `ok=false` (e.g. worker /metrics unreachable) carry no
    `metrics` field and must be silently dropped, not crash on KeyError."""
    metrics_path = tmp_path / "vllm_metrics.ndjson"
    _write_ndjson(metrics_path, [
        _scrape_row(1.0, "p0", "prefill", {"vllm:num_requests_running": _g(2.0)}),
        {"ts": 2.0, "worker": "p0", "role": "prefill", "host": "127.0.0.1",
         "port": 21000, "ok": False, "error": "ConnectionRefused"},
        _scrape_row(3.0, "p0", "prefill", {"vllm:num_requests_running": _g(6.0)}),
    ])
    out = tmp_path / "out"
    rc = mod.main(["--metrics", str(metrics_path), "--output", str(out)])
    assert rc == 0
    rows = list(csv.DictReader((out / "vllm_metrics_stats.csv").read_text().splitlines()))
    g_row = next(r for r in rows if r["metric"] == "vllm:num_requests_running"
                                 and r["worker"] == "p0")
    assert int(g_row["n"]) == 2
    assert float(g_row["mean"]) == pytest.approx(4.0)   # mean(2, 6)


def test_main_returns_nonzero_when_no_rows_in_window(mod, tmp_path, capsys):
    profile = tmp_path / "ses.jsonl"
    _write_profile(profile, session_id="ses_x", start=10.0, end=20.0)
    metrics_path = tmp_path / "vllm_metrics.ndjson"
    _write_ndjson(metrics_path, [
        _scrape_row(5.0, "p0", "prefill", {"vllm:num_requests_running": _g(1.0)}),
        _scrape_row(25.0, "p0", "prefill", {"vllm:num_requests_running": _g(1.0)}),
    ])
    rc = mod.main(["--profile", str(profile),
                   "--metrics", str(metrics_path),
                   "--output", str(tmp_path / "out")])
    assert rc == 1
    assert "no scrape rows inside the window" in capsys.readouterr().err


def test_main_histogram_end_to_end(mod, tmp_path):
    """Histogram (bucket/count/sum) flows through main(): row appears
    with type=histogram and reconstructed mean / quantiles."""
    metrics_path = tmp_path / "vllm_metrics.ndjson"
    _write_ndjson(metrics_path, [
        _scrape_row(1.0, "p0", "prefill", {
            "vllm:ttft_seconds_bucket": _bucket({0.1: 0, 0.5: 0, 1.0: 0, "+Inf": 0}),
            "vllm:ttft_seconds_count":  _g(0.0),
            "vllm:ttft_seconds_sum":    _g(0.0),
        }),
        _scrape_row(11.0, "p0", "prefill", {
            "vllm:ttft_seconds_bucket": _bucket({0.1: 10, 0.5: 60, 1.0: 90, "+Inf": 100}),
            "vllm:ttft_seconds_count":  _g(100.0),
            "vllm:ttft_seconds_sum":    _g(50.0),
        }),
    ])
    out = tmp_path / "out"
    rc = mod.main(["--metrics", str(metrics_path), "--output", str(out)])
    assert rc == 0
    rows = list(csv.DictReader((out / "vllm_metrics_stats.csv").read_text().splitlines()))
    h_row = next(r for r in rows if r["metric"] == "vllm:ttft_seconds"
                                 and r["worker"] == "p0")
    assert h_row["type"] == "histogram"
    assert float(h_row["mean"]) == pytest.approx(0.5)
    # p90 lands exactly at upper bound of bucket (1.0, cum=90) -> 1.0
    assert float(h_row["p90"]) == pytest.approx(1.0)
    assert float(h_row["delta"]) == pytest.approx(100.0)


# ---------- window trimming (--trim-margin-s / --trim-head-s / --trim-tail-s) ----------


def _kv_rows(ts_vals):
    """One agg worker, kv_cache_usage gauge = 0 on the first/last two ticks
    (idle warmup/cooldown), 0.8 in the middle."""
    n = len(ts_vals)
    out = []
    for i, ts in enumerate(ts_vals):
        val = 0.0 if (i < 2 or i >= n - 2) else 0.8
        out.append(_scrape_row(float(ts), "a0", "agg",
                               {"vllm:kv_cache_usage_perc": _g(val)}))
    return out


def test_main_trim_margin_drops_edge_ticks(mod, tmp_path, capsys):
    p = tmp_path / "m.ndjson"
    _write_ndjson(p, _kv_rows(range(100, 111)))   # 11 ticks 100..110
    out = tmp_path / "o"
    rc = mod.main(["--metrics", str(p), "--output", str(out), "--trim-margin-s", "2"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "7 scrape rows across 6.00s" in captured
    assert "trimmed head=2.0s tail=2.0s" in captured
    # only the busy 0.8 ticks survive -> mean is 0.8, not diluted by idles
    row = next(r for r in csv.DictReader((out / "vllm_metrics_stats.csv")
                                         .read_text().splitlines())
               if r["metric"] == "vllm:kv_cache_usage_perc" and r["worker"] == "a0")
    assert float(row["mean"]) == pytest.approx(0.8)


def test_main_trim_per_side_overrides_margin(mod, tmp_path, capsys):
    p = tmp_path / "m.ndjson"
    _write_ndjson(p, _kv_rows(range(100, 111)))
    out = tmp_path / "o"
    rc = mod.main(["--metrics", str(p), "--output", str(out),
                   "--trim-margin-s", "2", "--trim-head-s", "3", "--trim-tail-s", "1"])
    assert rc == 0
    assert "trimmed head=3.0s tail=1.0s" in capsys.readouterr().out


def test_main_trim_no_op_when_zero(mod, tmp_path, capsys):
    p = tmp_path / "m.ndjson"
    _write_ndjson(p, _kv_rows(range(100, 106)))
    out = tmp_path / "o"
    rc = mod.main(["--metrics", str(p), "--output", str(out)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "6 scrape rows" in captured
    assert "trimmed" not in captured


def test_main_over_trim_returns_1(mod, tmp_path, capsys):
    p = tmp_path / "m.ndjson"
    _write_ndjson(p, _kv_rows(range(100, 111)))
    out = tmp_path / "o"
    rc = mod.main(["--metrics", str(p), "--output", str(out), "--trim-margin-s", "20"])
    assert rc == 1
    assert "exceeds" in capsys.readouterr().err


def test_main_negative_trim_returns_2(mod, tmp_path, capsys):
    p = tmp_path / "m.ndjson"
    _write_ndjson(p, _kv_rows(range(100, 106)))
    out = tmp_path / "o"
    rc = mod.main(["--metrics", str(p), "--output", str(out), "--trim-margin-s", "-1"])
    assert rc == 2
    assert "trim values must be >= 0" in capsys.readouterr().err


def test_main_trim_session_window(mod, tmp_path, capsys):
    # session window 100..110; trim 2s both -> 102..108
    p = tmp_path / "m.ndjson"
    _write_ndjson(p, _kv_rows(range(100, 111)))
    profile = tmp_path / "ses.jsonl"
    profile.write_text(
        json.dumps({"ev": "query.start", "sessionID": "s1", "ts": 100.0}) + "\n" +
        json.dumps({"ev": "query.end", "sessionID": "s1", "ts": 110.0}) + "\n")
    out = tmp_path / "o"
    rc = mod.main(["--metrics", str(p), "--profile", str(profile),
                   "--output", str(out), "--trim-margin-s", "2"])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "window=6.00s" in captured
    assert "trimmed head=2.0s tail=2.0s" in captured
