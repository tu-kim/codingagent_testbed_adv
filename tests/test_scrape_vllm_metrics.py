"""Tests for scripts/scrape_vllm_metrics.py.

Prometheus text-format parsing is the core risky bit (regex over
exposition format); we cover the relevant shapes. HTTP scraping is
mocked via urlopen monkeypatch — no real worker required.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "scrape_vllm_metrics.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("scrape_vllm_metrics", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["scrape_vllm_metrics"] = module
    spec.loader.exec_module(module)
    return module


# ---------- Prometheus parser ----------


SAMPLE_METRICS = """\
# HELP vllm:num_requests_running Number of running requests
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model="qwen3-coder-30b-a3b-instruct-fp8"} 4.0
# HELP vllm:kv_cache_usage_perc KV cache usage fraction (v1 renamed from gpu_cache_usage_perc)
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model="qwen3-coder-30b-a3b-instruct-fp8"} 0.6234
# HELP vllm:prompt_tokens_cached_total Cumulative input tokens that hit the prefix cache
# TYPE vllm:prompt_tokens_cached_total counter
vllm:prompt_tokens_cached_total{model="qwen3-coder-30b-a3b-instruct-fp8"} 9999
# HELP vllm:prompt_tokens_total Total prompt tokens generated
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{model="qwen3-coder-30b-a3b-instruct-fp8"} 12345
# HELP vllm:time_to_first_token_seconds TTFT histogram
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.005",model="qwen3-coder-30b-a3b-instruct-fp8"} 0.0
vllm:time_to_first_token_seconds_bucket{le="0.01",model="qwen3-coder-30b-a3b-instruct-fp8"} 0.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model="qwen3-coder-30b-a3b-instruct-fp8"} 142.0
vllm:time_to_first_token_seconds_count{model="qwen3-coder-30b-a3b-instruct-fp8"} 142.0
vllm:time_to_first_token_seconds_sum{model="qwen3-coder-30b-a3b-instruct-fp8"} 38.412
# HELP vllm:request_queue_time_seconds Scheduler queue wait (scheduling delay) histogram
# TYPE vllm:request_queue_time_seconds histogram
vllm:request_queue_time_seconds_bucket{le="0.01",model="qwen3-coder-30b-a3b-instruct-fp8"} 5.0
vllm:request_queue_time_seconds_bucket{le="0.1",model="qwen3-coder-30b-a3b-instruct-fp8"} 30.0
vllm:request_queue_time_seconds_bucket{le="+Inf",model="qwen3-coder-30b-a3b-instruct-fp8"} 42.0
vllm:request_queue_time_seconds_count{model="qwen3-coder-30b-a3b-instruct-fp8"} 42.0
vllm:request_queue_time_seconds_sum{model="qwen3-coder-30b-a3b-instruct-fp8"} 3.21
# HELP process_cpu_seconds_total Total CPU time
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 1234.5
# HELP python_info Python platform information
# TYPE python_info gauge
python_info{implementation="CPython",major="3",minor="11"} 1.0
"""


def test_parse_extracts_gauge_with_label(mod):
    out = mod.parse_prometheus(SAMPLE_METRICS, keep_names=None)
    rows = out["vllm:num_requests_running"]
    assert len(rows) == 1
    assert rows[0]["labels"]["model"] == "qwen3-coder-30b-a3b-instruct-fp8"
    assert rows[0]["value"] == 4.0


def test_parse_extracts_histogram_buckets_count_sum(mod):
    out = mod.parse_prometheus(SAMPLE_METRICS, keep_names=None)
    # buckets accumulate as separate entries because labels differ (le=)
    assert "vllm:time_to_first_token_seconds_bucket" in out
    buckets = out["vllm:time_to_first_token_seconds_bucket"]
    assert len(buckets) == 3
    le_values = {b["labels"].get("le") for b in buckets}
    assert le_values == {"0.005", "0.01", "+Inf"}
    # count + sum are separate metric names
    assert out["vllm:time_to_first_token_seconds_count"][0]["value"] == 142.0
    assert out["vllm:time_to_first_token_seconds_sum"][0]["value"] == pytest.approx(38.412)


def test_parse_default_allowlist_keeps_only_curated_names(mod):
    """DEFAULT_METRIC_NAMES is an EXACT-MATCH allowlist (not a prefix
    filter): it keeps the ~10 KV-cache/queue/throughput names and
    drops everything else, including Python internals AND vLLM
    latency histograms (per-request latency is already in
    nvext.timing + opencode profile NDJSON, so duplicating it here
    would just balloon the NDJSON size)."""
    out = mod.parse_prometheus(SAMPLE_METRICS, keep_names=mod.DEFAULT_METRIC_NAMES)
    # Kept: in DEFAULT_METRIC_NAMES (using vLLM v0.19.0 names -- v1
    # engine dropped the `gpu_` prefix from cache gauges).
    assert "vllm:num_requests_running" in out
    assert "vllm:kv_cache_usage_perc" in out        # headline KV cache memory
    assert "vllm:prompt_tokens_cached_total" in out # hit-token counter
    assert "vllm:prompt_tokens_total" in out
    # Kept: the scheduler queue-wait histogram (scheduling delay) -- all
    # three suffixed series, since analyze_vllm_metrics needs bucket +
    # count + sum to derive percentiles.
    assert "vllm:request_queue_time_seconds_bucket" in out
    assert "vllm:request_queue_time_seconds_count" in out
    assert "vllm:request_queue_time_seconds_sum" in out
    # Dropped: Python/process internals.
    assert "process_cpu_seconds_total" not in out
    assert "python_info" not in out
    # Dropped: OTHER latency histograms (TTFT etc.) -- recoverable from
    # nvext.timing + opencode profile, unlike queue-wait.
    assert "vllm:time_to_first_token_seconds_bucket" not in out
    assert "vllm:time_to_first_token_seconds_count" not in out
    assert "vllm:time_to_first_token_seconds_sum" not in out


def test_parse_keep_all_returns_everything(mod):
    """--keep-all passes keep_names=None → no filtering."""
    out = mod.parse_prometheus(SAMPLE_METRICS, keep_names=None)
    assert "process_cpu_seconds_total" in out
    assert "python_info" in out
    assert "vllm:num_requests_running" in out
    # Histograms also surface in keep-all mode.
    assert "vllm:time_to_first_token_seconds_bucket" in out


def test_parse_custom_allowlist_keeps_exactly_named(mod):
    """Custom set passed via --metric-names yields exact-match behavior:
    only names in the set survive; everything else (even other vllm:*)
    is dropped."""
    out = mod.parse_prometheus(
        SAMPLE_METRICS,
        keep_names=frozenset({"vllm:kv_cache_usage_perc"}),
    )
    assert list(out.keys()) == ["vllm:kv_cache_usage_perc"]


def test_default_metric_names_covers_headline_kv_cache_signals(mod):
    """Smoke check on the allowlist composition itself: the load-
    bearing KV-cache + queue depth fields must be present so users
    relying on the default get meaningful output without --metric-names."""
    assert isinstance(mod.DEFAULT_METRIC_NAMES, frozenset)
    # Names verified against vLLM 0.19.0 (the dynamo-pinned version).
    # The v1 engine renamed `gpu_cache_usage_perc` -> `kv_cache_usage_perc`
    # and dropped the `gpu_` prefix from prefix-cache counters.
    must_include = {
        "vllm:kv_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prompt_tokens_cached_total",
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        # Scheduler queue-wait histogram (prefill/decode scheduling delay).
        "vllm:request_queue_time_seconds_bucket",
        "vllm:request_queue_time_seconds_count",
        "vllm:request_queue_time_seconds_sum",
    }
    assert must_include.issubset(mod.DEFAULT_METRIC_NAMES)
    # Stale v0 names must be ABSENT (caught a real regression once --
    # v0 prefix-cache-and-gpu-prefix names silently produced empty
    # scrape output against a v1 engine).
    stale_v0_names = {
        "vllm:gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc",
        "vllm:gpu_prefix_cache_queries_total",
        "vllm:gpu_prefix_cache_hits_total",
    }
    assert stale_v0_names.isdisjoint(mod.DEFAULT_METRIC_NAMES)
    # Latency histograms are explicitly NOT in the default (duplicates
    # nvext.timing + opencode profile, balloons NDJSON size).
    assert "vllm:time_to_first_token_seconds_bucket" not in mod.DEFAULT_METRIC_NAMES


def test_parse_handles_inf_and_nan_values(mod):
    text = (
        'vllm:test_inf 1.5e+10\n'
        'vllm:test_nan_or_neg{le="+Inf"} +Inf\n'
        'vllm:test_neg{model="x"} -1.5\n'
    )
    out = mod.parse_prometheus(text, keep_names=None)
    assert out["vllm:test_inf"][0]["value"] == 1.5e10
    assert out["vllm:test_neg"][0]["value"] == -1.5
    # +Inf parses to math.inf
    assert out["vllm:test_nan_or_neg"][0]["value"] == float("inf")


def test_parse_skips_comments_and_blank_lines(mod):
    text = "\n# a comment\n  \n"
    out = mod.parse_prometheus(text, keep_names=None)
    assert out == {}


def test_parse_unescapes_label_quotes_and_backslashes(mod):
    text = r'vllm:test{path="a\"b\\c"} 1'
    out = mod.parse_prometheus(text + "\n", keep_names=None)
    assert out["vllm:test"][0]["labels"]["path"] == 'a"b\\c'


# ---------- workers from testbed.yaml ----------


def _write_yaml(path: Path, system_port_base=21000,
                 prefill=None, decode=None) -> None:
    pytest.importorskip("yaml")
    import yaml
    yaml.dump({
        "vllm": {
            "system_port_base": system_port_base,
            "prefill_workers": prefill if prefill is not None else
                [{"name": "p0", "host": "127.0.0.1", "gpus": "0,1", "tp": 2, "pp": 1}],
            "decode_workers": decode if decode is not None else
                [{"name": "d0", "host": "127.0.0.1", "gpus": "2,3", "tp": 2, "pp": 1}],
        }
    }, path.open("w"))


def test_load_workers_assigns_sequential_ports(mod, tmp_path):
    pytest.importorskip("yaml")
    yaml_path = tmp_path / "testbed.yaml"
    _write_yaml(yaml_path, system_port_base=21000,
                 prefill=[{"name": "p0", "host": "127.0.0.1", "gpus": "0,1", "tp": 2, "pp": 1},
                          {"name": "p1", "host": "10.0.0.5", "gpus": "0,1", "tp": 2, "pp": 1}],
                 decode=[{"name": "d0", "host": "127.0.0.1", "gpus": "2,3", "tp": 2, "pp": 1}])
    workers = mod.load_workers(yaml_path)
    assert workers == [
        {"worker": "p0", "role": "prefill", "host": "127.0.0.1", "port": 21000},
        {"worker": "p1", "role": "prefill", "host": "10.0.0.5",  "port": 21001},
        {"worker": "d0", "role": "decode",  "host": "127.0.0.1", "port": 21002},
    ]


def test_load_workers_rejects_disabled_system_port(mod, tmp_path):
    pytest.importorskip("yaml")
    yaml_path = tmp_path / "testbed.yaml"
    _write_yaml(yaml_path, system_port_base=-1)
    with pytest.raises(SystemExit, match="system_port_base"):
        mod.load_workers(yaml_path)


def test_load_workers_rejects_empty_worker_list(mod, tmp_path):
    pytest.importorskip("yaml")
    yaml_path = tmp_path / "testbed.yaml"
    _write_yaml(yaml_path, prefill=[], decode=[])
    with pytest.raises(SystemExit, match="no vllm workers"):
        mod.load_workers(yaml_path)


# ---------- scrape_one (HTTP mocked) ----------


def _mk_response(text: str):
    class FakeResp:
        def __init__(self, body: bytes):
            self._body = body
        def read(self): return self._body
        def __enter__(self): return self
        def __exit__(self, *a): return False
    return FakeResp(text.encode("utf-8"))


def test_scrape_one_returns_parsed_metrics_on_200(mod):
    with patch.object(mod.urllib.request, "urlopen",
                       side_effect=lambda *a, **kw: _mk_response(SAMPLE_METRICS)):
        ok, payload = mod.scrape_one("127.0.0.1", 21000, 2.0,
                                       keep_names=mod.DEFAULT_METRIC_NAMES)
    assert ok is True
    assert "vllm:num_requests_running" in payload
    assert payload["vllm:num_requests_running"][0]["value"] == 4.0


def test_scrape_one_returns_error_on_connection_failure(mod):
    import urllib.error
    def _raise(*a, **kw):
        raise urllib.error.URLError("Connection refused")
    with patch.object(mod.urllib.request, "urlopen", side_effect=_raise):
        ok, payload = mod.scrape_one("127.0.0.1", 21000, 2.0,
                                       keep_names=mod.DEFAULT_METRIC_NAMES)
    assert ok is False
    assert "Connection refused" in payload


def test_scrape_one_returns_error_on_http_5xx(mod):
    """HTTPError is a subclass of URLError so the except clause catches
    it; ensure we surface a meaningful error string instead of the
    parsed-but-empty-body silent-success path."""
    import urllib.error
    def _raise(*a, **kw):
        raise urllib.error.HTTPError(
            url="http://127.0.0.1:21000/metrics", code=500,
            msg="Internal Server Error", hdrs=None, fp=None,
        )
    with patch.object(mod.urllib.request, "urlopen", side_effect=_raise):
        ok, payload = mod.scrape_one("127.0.0.1", 21000, 2.0,
                                       keep_names=mod.DEFAULT_METRIC_NAMES)
    assert ok is False
    assert "HTTPError" in payload
    assert "500" in payload


def test_scrape_one_returns_ok_with_empty_metrics_on_blank_body(mod):
    """A worker that's up but hasn't observed any requests yet may
    expose /metrics returning only HELP/TYPE comments (or nothing).
    Boundary value: ok=True with metrics={}, NOT ok=False."""
    with patch.object(mod.urllib.request, "urlopen",
                       side_effect=lambda *a, **kw: _mk_response("")):
        ok, payload = mod.scrape_one("127.0.0.1", 21000, 2.0,
                                       keep_names=mod.DEFAULT_METRIC_NAMES)
    assert ok is True
    assert payload == {}


# ---------- run_scraper end-to-end ----------


def test_run_scraper_writes_one_row_per_worker_per_tick(mod, tmp_path):
    out_path = tmp_path / "metrics.ndjson"
    workers = [
        {"worker": "p0", "role": "prefill", "host": "127.0.0.1", "port": 21000},
        {"worker": "d0", "role": "decode",  "host": "127.0.0.1", "port": 21001},
    ]
    state = {"count": 0}
    def stop():
        state["count"] += 1
        return state["count"] > 2  # 2 ticks

    # `side_effect=lambda ...` builds a fresh FakeResp per call instead
    # of reusing one instance across all 4 urlopen invocations -- safer
    # for tests that touch read-once stream-like wrappers.
    with patch.object(mod.urllib.request, "urlopen",
                       side_effect=lambda *a, **kw: _mk_response(SAMPLE_METRICS)):
        n = mod.run_scraper(workers, 0.0, out_path,
                             mod.DEFAULT_METRIC_NAMES, 2.0, stop)
    assert n == 4  # 2 workers × 2 ticks
    lines = [json.loads(l) for l in out_path.read_text().splitlines() if l]
    assert len(lines) == 4
    for row in lines:
        assert row["ok"] is True
        assert "metrics" in row
        assert "vllm:num_requests_running" in row["metrics"]
        assert row["worker"] in ("p0", "d0")
        assert row["role"] in ("prefill", "decode")


def test_run_scraper_records_error_when_worker_unreachable(mod, tmp_path):
    """One dead worker must not kill the loop; record `ok:false` +
    error string and continue."""
    import urllib.error
    out_path = tmp_path / "metrics.ndjson"
    workers = [{"worker": "p0", "role": "prefill", "host": "127.0.0.1", "port": 21000}]
    state = {"count": 0}
    def stop():
        state["count"] += 1
        return state["count"] > 1
    with patch.object(mod.urllib.request, "urlopen",
                       side_effect=urllib.error.URLError("ECONNREFUSED")):
        n = mod.run_scraper(workers, 0.0, out_path,
                             mod.DEFAULT_METRIC_NAMES, 2.0, stop)
    assert n == 1
    row = json.loads(out_path.read_text().strip())
    assert row["ok"] is False
    assert "ECONNREFUSED" in row["error"]
    assert "metrics" not in row
