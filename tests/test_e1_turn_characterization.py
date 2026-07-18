"""Tests for scripts/arm/e1_turn_characterization.py — the E0 views
adapted to concurrent (interleaved-session) runs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT = (Path(__file__).resolve().parents[1]
           / "scripts" / "arm" / "e1_turn_characterization.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("e1_turn_characterization", _SCRIPT)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    sys.modules["e1_turn_characterization"] = m
    spec.loader.exec_module(m)
    return m


class _T:
    def __init__(self, sid, step, start, end):
        self.session_id = sid
        self.step = step
        self.llm_start_ts = start
        self.llm_end_ts = end


def test_session_first_ordinals_no_overlap_filter(mod):
    # A and B OVERLAP in time (concurrent run); both must still yield a
    # boundary — E0's window filter would have dropped B as "nested".
    ordered = [_T("A", 1, 0.0, 1.0), _T("B", 1, 0.5, 1.5),
               _T("A", 2, 3.0, 4.0), _T("B", 2, 3.5, 4.5)]
    assert mod.session_first_ordinals(ordered) == [0, 1]
    assert mod.session_first_times_abs(ordered) == [0.0, 0.5]


def _turn(step, a, b, inp, out, cache, tool):
    evs = [{"ev": "llm.start", "ts": a, "step": step},
           {"ev": "llm.end", "ts": b, "step": step, "duration_s": b - a,
            "tokens": {"input": inp, "output": out, "cache": {"read": cache}},
            "dynamo": {}}]
    if tool:
        evs.append({"ev": "tool.end", "ts": b + 0.05, "step": step,
                    "name": tool, "callID": f"c{step}", "duration_s": 0.5,
                    "ok": True})
    return evs


def test_main_interleaved_sessions_no_figures(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    (prof / "A.jsonl").write_text("".join(
        json.dumps(e) + "\n" for e in
        _turn(1, 0, 1, 500, 50, 0, "read") + _turn(2, 3, 4, 600, 60, 500, None)))
    (prof / "B.jsonl").write_text("".join(
        json.dumps(e) + "\n" for e in
        _turn(1, 0.5, 1.5, 400, 40, 0, "bash") + _turn(2, 3.5, 4.5, 500, 50, 400, None)))
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"session_id": "A"}) + "\n"
                     + json.dumps({"session_id": "B"}) + "\n")
    out = tmp_path / "o"
    rc = mod.main(["--profiles", str(prof), "--trace", str(trace),
                   "--out", str(out), "--no-figures"])
    assert rc == 0
    assert not list(out.glob("*.pdf"))
    assert not list(out.glob("*.csv"))


class _TE:
    def __init__(self, sid, step, inp, cache, out, away=None, disp=None):
        self.session_id = sid
        self.step = step
        self.input_tokens = inp
        self.cache_read = cache
        self.output_tokens = out
        self.away_s = away
        self.away_displaced_tokens = disp
        self.llm_start_ts = float(step)
        self.llm_end_ts = float(step) + 0.5

    @property
    def effective_input(self):
        return None if self.input_tokens is None else self.input_tokens + self.cache_read


def test_eviction_events_discriminates_eviction_vs_compaction(mod):
    # step1: prev_cached = 1000 + 100 = 1100
    # step2: prompt GREW (eff 1600 > 1000) but cache_read 200 -> eviction
    # step3: prompt SHRANK (eff 300 < 0.6*1600) -> compaction
    turns = [_TE("A", 1, 1000, 0, 100),
             _TE("A", 2, 1400, 200, 80, away=5.0, disp=4000),
             _TE("A", 3, 300, 0, 50, away=2.0, disp=100)]
    ev = {e["step"]: e for e in mod.eviction_events(turns)}
    assert ev[2]["label"] == "eviction"
    assert ev[2]["shortfall"] == 1100 - 200
    assert ev[3]["label"] == "compaction"


def test_eviction_events_min_shortfall_noise_is_ok(mod):
    # a tiny shortfall (block granularity) is neither eviction nor compaction
    turns = [_TE("A", 1, 1000, 0, 0),
             _TE("A", 2, 1050, 950, 0)]     # prev_cached 1000, cr 950 -> short 50
    ev = mod.eviction_events(turns, min_shortfall=128)
    assert ev[0]["label"] == "ok"


def test_window_avg_speed_delta_sum_over_delta_count(mod):
    # window mean = delta(_sum)/delta(_count); zero-delta and gap skipped
    recs = [{"ts": 0, "s": 100.0, "c": 2},
            {"ts": 1, "s": 400.0, "c": 4},     # (400-100)/(4-2) = 150
            {"ts": 2, "s": 400.0, "c": 4},     # dcount 0 -> skip
            {"ts": 3, "s": None, "c": None},   # gap -> break the chain
            {"ts": 4, "s": 500.0, "c": 5}]     # prev reset -> no point
    assert mod.window_avg_speed(recs, "s", "c") == [(1, 150.0)]


def test_window_avg_speed_skips_counter_reset(mod):
    # a Prometheus counter reset (e.g. worker restart) makes _sum go DOWN;
    # that window must be dropped (ds < 0), not reported as negative speed,
    # but the reset value still seeds the next window's baseline.
    recs = [{"ts": 0, "s": 1000.0, "c": 10},
            {"ts": 1, "s": 50.0, "c": 1},      # reset: ds=-950 -> skip
            {"ts": 2, "s": 150.0, "c": 3}]     # (150-50)/(3-1) = 50, post-reset
    assert mod.window_avg_speed(recs, "s", "c") == [(2, 50.0)]


def _lmc_row(ts, worker, cpu_b, disk_b, r_sum, r_cnt, s_sum, s_cnt,
             ok=True, role="agg"):
    return {"ts": ts, "worker": worker, "role": role, "ok": ok, "metrics": {
        "vllm:kv_cache_usage_perc": [{"labels": {}, "value": 0.5}],
        "lmcache:local_cache_usage": [{"labels": {}, "value": cpu_b}],
        "lmcache:local_storage_usage": [{"labels": {}, "value": disk_b}],
        "lmcache:retrieve_speed_sum": [{"labels": {}, "value": r_sum}],
        "lmcache:retrieve_speed_count": [{"labels": {}, "value": r_cnt}],
        "lmcache:store_speed_sum": [{"labels": {}, "value": s_sum}],
        "lmcache:store_speed_count": [{"labels": {}, "value": s_cnt}]}}


def test_lmcache_series_reads_lmcache_rows_only(mod, tmp_path):
    p = tmp_path / "m.ndjson"
    rows = [
        _lmc_row(10.0, "a0", 2 << 30, 1 << 30, 1000.0, 10, 500.0, 5),
        _lmc_row(11.0, "a0", 3 << 30, 1 << 30, 3200.0, 20, 900.0, 9),
        # a plain vLLM row (no lmcache: metric) must be skipped
        {"ts": 10.5, "worker": "a0", "role": "agg", "ok": True,
         "metrics": {"vllm:kv_cache_usage_perc": [{"labels": {}, "value": 0.9}]}},
        # not-ok row dropped
        _lmc_row(12.0, "a0", 9 << 30, 0, 0.0, 0, 0.0, 0, ok=False),
        "not json",
    ]
    p.write_text("".join(
        (r if isinstance(r, str) else json.dumps(r)) + "\n" for r in rows))
    lm = mod.lmcache_series(p)
    assert list(lm) == ["a0"] and len(lm["a0"]) == 2
    assert lm["a0"][0]["local_usage_bytes"] == 2 << 30
    # retrieve = host->device onboard speed, window-avg tokens/sec
    assert mod.window_avg_speed(lm["a0"], "retrieve_sum", "retrieve_count") \
        == [(11.0, (3200.0 - 1000.0) / (20 - 10))]


def test_main_with_lmcache_metrics_no_figures(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    (prof / "A.jsonl").write_text("".join(
        json.dumps(e) + "\n" for e in
        _turn(1, 0, 1, 500, 50, 0, "read") + _turn(2, 3, 4, 600, 60, 500, None)))
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"session_id": "A"}) + "\n")
    met = tmp_path / "m.ndjson"
    met.write_text(json.dumps(_lmc_row(0.5, "a0", 1 << 30, 0, 10.0, 1, 5.0, 1)) + "\n")
    rc = mod.main(["--profiles", str(prof), "--trace", str(trace),
                   "--metrics", str(met), "--cpu-cache-gb", "8",
                   "--out", str(tmp_path / "o"), "--no-figures"])
    assert rc == 0


def test_main_missing_trace_returns_2(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    rc = mod.main(["--profiles", str(prof), "--trace", str(tmp_path / "no")])
    assert rc == 2
