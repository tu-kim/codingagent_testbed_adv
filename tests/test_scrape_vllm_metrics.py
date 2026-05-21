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
# HELP vllm:gpu_cache_usage_perc GPU KV cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model="qwen3-coder-30b-a3b-instruct-fp8"} 0.6234
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
# HELP process_cpu_seconds_total Total CPU time
# TYPE process_cpu_seconds_total counter
process_cpu_seconds_total 1234.5
# HELP python_info Python platform information
# TYPE python_info gauge
python_info{implementation="CPython",major="3",minor="11"} 1.0
"""


def test_parse_extracts_gauge_with_label(mod):
    out = mod.parse_prometheus(SAMPLE_METRICS, keep_prefixes=("vllm:",))
    rows = out["vllm:num_requests_running"]
    assert len(rows) == 1
    assert rows[0]["labels"]["model"] == "qwen3-coder-30b-a3b-instruct-fp8"
    assert rows[0]["value"] == 4.0


def test_parse_extracts_histogram_buckets_count_sum(mod):
    out = mod.parse_prometheus(SAMPLE_METRICS, keep_prefixes=("vllm:",))
    # buckets accumulate as separate entries because labels differ (le=)
    assert "vllm:time_to_first_token_seconds_bucket" in out
    buckets = out["vllm:time_to_first_token_seconds_bucket"]
    assert len(buckets) == 3
    le_values = {b["labels"].get("le") for b in buckets}
    assert le_values == {"0.005", "0.01", "+Inf"}
    # count + sum are separate metric names
    assert out["vllm:time_to_first_token_seconds_count"][0]["value"] == 142.0
    assert out["vllm:time_to_first_token_seconds_sum"][0]["value"] == pytest.approx(38.412)


def test_parse_default_prefix_filter_drops_python_internals(mod):
    """Default keep_prefixes hides process_cpu_seconds_total and python_info."""
    out = mod.parse_prometheus(SAMPLE_METRICS, keep_prefixes=mod.DEFAULT_PREFIXES)
    assert "process_cpu_seconds_total" not in out
    assert "python_info" not in out
    assert "vllm:num_requests_running" in out


def test_parse_keep_all_returns_everything(mod):
    """--keep-all passes keep_prefixes=None → no filtering."""
    out = mod.parse_prometheus(SAMPLE_METRICS, keep_prefixes=None)
    assert "process_cpu_seconds_total" in out
    assert "python_info" in out
    assert "vllm:num_requests_running" in out


def test_parse_handles_inf_and_nan_values(mod):
    text = (
        'vllm:test_inf 1.5e+10\n'
        'vllm:test_nan_or_neg{le="+Inf"} +Inf\n'
        'vllm:test_neg{model="x"} -1.5\n'
    )
    out = mod.parse_prometheus(text, keep_prefixes=("vllm:",))
    assert out["vllm:test_inf"][0]["value"] == 1.5e10
    assert out["vllm:test_neg"][0]["value"] == -1.5
    # +Inf parses to math.inf
    assert out["vllm:test_nan_or_neg"][0]["value"] == float("inf")


def test_parse_skips_comments_and_blank_lines(mod):
    text = "\n# a comment\n  \n"
    out = mod.parse_prometheus(text, keep_prefixes=None)
    assert out == {}


def test_parse_unescapes_label_quotes_and_backslashes(mod):
    text = r'vllm:test{path="a\"b\\c"} 1'
    out = mod.parse_prometheus(text + "\n", keep_prefixes=None)
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
                       return_value=_mk_response(SAMPLE_METRICS)):
        ok, payload = mod.scrape_one("127.0.0.1", 21000, 2.0,
                                       keep_prefixes=mod.DEFAULT_PREFIXES)
    assert ok is True
    assert "vllm:num_requests_running" in payload
    assert payload["vllm:num_requests_running"][0]["value"] == 4.0


def test_scrape_one_returns_error_on_connection_failure(mod):
    import urllib.error
    def _raise(*a, **kw):
        raise urllib.error.URLError("Connection refused")
    with patch.object(mod.urllib.request, "urlopen", side_effect=_raise):
        ok, payload = mod.scrape_one("127.0.0.1", 21000, 2.0,
                                       keep_prefixes=mod.DEFAULT_PREFIXES)
    assert ok is False
    assert "Connection refused" in payload


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

    with patch.object(mod.urllib.request, "urlopen",
                       return_value=_mk_response(SAMPLE_METRICS)):
        n = mod.run_scraper(workers, 0.0, out_path,
                             mod.DEFAULT_PREFIXES, 2.0, stop)
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
                             mod.DEFAULT_PREFIXES, 2.0, stop)
    assert n == 1
    row = json.loads(out_path.read_text().strip())
    assert row["ok"] is False
    assert "ECONNREFUSED" in row["error"]
    assert "metrics" not in row
