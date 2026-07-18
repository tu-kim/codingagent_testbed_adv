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


def test_counter_rate_delta_over_dt_skips_reset_and_gap(mod):
    # rate = delta(value)/delta(ts); zero-delta -> 0, reset (dv<0) skipped,
    # None breaks the chain (no cross-gap rate).
    recs = [{"ts": 0, "k": 10.0}, {"ts": 2, "k": 30.0},   # 20/2 = 10
            {"ts": 4, "k": 30.0},                          # flat -> 0.0
            {"ts": 6, "k": 5.0},                           # reset -> skip
            {"ts": 7, "k": None},                          # gap -> break
            {"ts": 8, "k": 100.0}]                         # prev reset -> none
    assert mod.counter_rate(recs, "k") == [(2, 10.0), (4, 0.0)]


def _lmc_row(ts, worker, cpu_b, disk_b, r_sum, r_cnt, s_sum, s_cnt,
             evict_keys=0, evict_failed=0, hit_tokens=0, stored_tokens=0,
             ok=True, role="agg"):
    return {"ts": ts, "worker": worker, "role": role, "ok": ok, "metrics": {
        "vllm:kv_cache_usage_perc": [{"labels": {}, "value": 0.5}],
        "lmcache:local_cache_usage": [{"labels": {}, "value": cpu_b}],
        "lmcache:local_storage_usage": [{"labels": {}, "value": disk_b}],
        "lmcache:retrieve_speed_sum": [{"labels": {}, "value": r_sum}],
        "lmcache:retrieve_speed_count": [{"labels": {}, "value": r_cnt}],
        "lmcache:store_speed_sum": [{"labels": {}, "value": s_sum}],
        "lmcache:store_speed_count": [{"labels": {}, "value": s_cnt}],
        "lmcache:local_cpu_evict_keys_count": [{"labels": {}, "value": evict_keys}],
        "lmcache:local_cpu_evict_failed_count": [{"labels": {}, "value": evict_failed}],
        "lmcache:num_hit_tokens": [{"labels": {}, "value": hit_tokens}],
        "lmcache:num_stored_tokens": [{"labels": {}, "value": stored_tokens}]}}


def test_lmcache_series_maps_transferred_token_counters(mod, tmp_path):
    # locks the field-name mapping: lmcache:num_hit_tokens -> hit_tokens,
    # lmcache:num_stored_tokens -> stored_tokens (an upstream rename must
    # fail here, not silently zero out fig8).
    p = tmp_path / "m.ndjson"
    p.write_text(
        json.dumps(_lmc_row(10.0, "a0", 1 << 30, 0, 1000.0, 10, 500.0, 5,
                            hit_tokens=300, stored_tokens=400)) + "\n"
        + json.dumps(_lmc_row(11.0, "a0", 2 << 30, 0, 3200.0, 20, 900.0, 9,
                              hit_tokens=500, stored_tokens=800)) + "\n")
    lm = mod.lmcache_series(p)
    assert [r["hit_tokens"] for r in lm["a0"]] == [300, 500]
    assert [r["stored_tokens"] for r in lm["a0"]] == [400, 800]
    # end-to-end through transfer_batches with the real field names:
    # window tokens = 500-300 = 200, speed = (3200-1000)/(20-10) = 220
    tb = mod.transfer_batches(lm["a0"], "hit_tokens",
                              "retrieve_sum", "retrieve_count")
    assert tb == [(11.0, 200, pytest.approx(200 / 220.0))]


def test_transfer_batches_seconds_from_tokens_over_speed(mod):
    # window t=1: tokens moved = 1200-1000 = 200; window mean speed =
    # (3200-1000)/(20-10) = 220 tok/s; seconds = 200/220. dc=0 window skip.
    recs = [{"ts": 0, "tok": 1000, "s": 1000.0, "c": 10},
            {"ts": 1, "tok": 1200, "s": 3200.0, "c": 20},
            {"ts": 2, "tok": 1200, "s": 3200.0, "c": 20}]  # dc 0 -> skip
    tb = mod.transfer_batches(recs, "tok", "s", "c")
    assert len(tb) == 1
    ts, tokens, secs = tb[0]
    assert ts == 1 and tokens == 200
    assert secs == pytest.approx(200 / 220.0)


def test_transfer_batches_skips_missing_series_and_gap(mod):
    # a None in any of the three series breaks the delta chain
    recs = [{"ts": 0, "tok": 100, "s": 10.0, "c": 1},
            {"ts": 1, "tok": None, "s": 20.0, "c": 2},   # gap
            {"ts": 2, "tok": 300, "s": 30.0, "c": 3}]     # prev reset -> none
    assert mod.transfer_batches(recs, "tok", "s", "c") == []


def test_lmcache_series_reads_eviction_counters(mod, tmp_path):
    p = tmp_path / "m.ndjson"
    p.write_text(
        json.dumps(_lmc_row(10.0, "a0", 1 << 30, 0, 0.0, 0, 0.0, 0,
                            evict_keys=0)) + "\n"
        + json.dumps(_lmc_row(11.0, "a0", 2 << 30, 0, 0.0, 0, 0.0, 0,
                              evict_keys=30, evict_failed=2)) + "\n")
    lm = mod.lmcache_series(p)
    assert [r["evict_keys"] for r in lm["a0"]] == [0, 30]
    # chunks/sec across the 1s window
    assert mod.counter_rate(lm["a0"], "evict_keys") == [(11.0, 30.0)]


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
