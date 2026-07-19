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
    def __init__(self, sid, step, inp, cache, out, away=None, disp=None,
                 reasoning=0):
        self.session_id = sid
        self.step = step
        self.input_tokens = inp
        self.cache_read = cache
        self.output_tokens = out
        self.reasoning_tokens = reasoning
        self.away_s = away
        self.away_displaced_tokens = disp
        self.llm_start_ts = float(step)
        self.llm_end_ts = float(step) + 0.5

    @property
    def effective_input(self):
        return None if self.input_tokens is None else self.input_tokens + self.cache_read


def test_session_spans_first_start_to_last_end_ordered(mod):
    turns = [_T("A", 1, 0.0, 1.0), _T("A", 2, 3.0, 4.0),
             _T("B", 1, 0.5, 1.5),
             _T("X", 1, None, None)]     # missing timing -> dropped
    sp = mod.session_spans(turns)
    assert [d["session_id"] for d in sp] == ["A", "B"]   # by start time
    assert sp[0]["start"] == 0.0 and sp[0]["end"] == 4.0
    assert sp[0]["segments"] == [(0.0, 1.0), (3.0, 4.0)]
    assert sp[1]["start"] == 0.5 and sp[1]["end"] == 1.5


def test_session_spans_end_is_max_not_last_segment(mod):
    # out-of-order / overlapping turns: end must be the MAX end, not the
    # last-by-start segment's end
    turns = [_T("A", 1, 0.0, 5.0), _T("A", 2, 1.0, 2.0)]
    sp = mod.session_spans(turns)
    assert sp[0]["start"] == 0.0 and sp[0]["end"] == 5.0


def test_session_spans_drops_only_the_none_turn_not_the_whole_session(mod):
    # A has one valid turn and one turn with missing timing; the session
    # must still appear, built from just the valid turn (per-turn skip,
    # not a session-level drop).
    turns = [_T("A", 1, 0.0, 1.0), _T("A", 2, None, None)]
    sp = mod.session_spans(turns)
    assert [d["session_id"] for d in sp] == ["A"]
    assert sp[0]["segments"] == [(0.0, 1.0)]
    assert sp[0]["start"] == 0.0 and sp[0]["end"] == 1.0


def test_eviction_events_excludes_prior_turn_reasoning(mod):
    # prev_cached must exclude prior-turn reasoning (dropped from the next
    # prompt): eff(N-1)=1000, out=200, reasoning=150 -> prev_cached =
    # 1000 + (200-150) = 1050 (not 1200). cache_read 1000 -> shortfall 50.
    turns = [_TE("A", 1, 1000, 0, 200, reasoning=150),
             _TE("A", 2, 1400, 1000, 80)]
    ev = {e["step"]: e for e in mod.eviction_events(turns)}
    assert ev[2]["prev_cached"] == 1050
    assert ev[2]["shortfall"] == 50
    assert ev[2]["label"] == "ok"       # below the 128 min-shortfall


def test_prefix_mismatch_pct_excludes_compaction(mod):
    ev = [{"prev_cached": 900, "shortfall": 100, "label": "eviction"},
          {"prev_cached": 500, "shortfall": 400, "label": "compaction"},  # out
          {"prev_cached": 600, "shortfall": 50, "label": "ok"}]
    missed, reusable, pct = mod.prefix_mismatch_pct(ev)
    assert missed == 150 and reusable == 1500
    assert pct == pytest.approx(10.0)
    assert mod.prefix_mismatch_pct([]) == (0, 0, None)


def test_eviction_loss_pct(mod):
    # missed = eviction-turn shortfall only; total = prev_cached over ALL
    ev = [{"prev_cached": 1000, "shortfall": 300, "label": "eviction"},
          {"prev_cached": 500, "shortfall": 100, "label": "compaction"},
          {"prev_cached": 800, "shortfall": 0, "label": "ok"}]
    missed, total, pct = mod.eviction_loss_pct(ev)
    assert missed == 300 and total == 2300
    assert pct == pytest.approx(300 / 2300 * 100)
    assert mod.eviction_loss_pct([]) == (0, 0, None)


class _TS:
    """Turn double for session_spans with dynamo timing + request_id."""
    def __init__(self, sid, step, start, end, elapsed=None, rid=None):
        self.session_id = sid
        self.step = step
        self.llm_start_ts = start
        self.llm_end_ts = end
        self.elapsed_s = elapsed
        self.request_id = rid


def test_session_spans_prefill_decode_segments_with_queue_join(mod):
    # turn1: llm_end 110, elapsed 10, queue 4s -> active 6 -> seg (104, 110)
    # turn2: llm_end 120, elapsed 5, queue 1s -> active 4 -> seg (116, 120)
    turns = [_TS("A", 1, 100.0, 110.0, elapsed=10.0, rid="r1"),
             _TS("A", 2, 115.0, 120.0, elapsed=5.0, rid="r2")]
    sp = mod.session_spans(turns, {"r1": 4000.0, "r2": 1000.0})
    assert sp[0]["segments"] == [(104.0, 110.0), (116.0, 120.0)]
    assert sp[0]["queue_s"] == pytest.approx(5.0)
    # qseg1 = (llm_end - elapsed, llm_end - active) = (110-10, 110-6) = (100, 104)
    # qseg2 = (120-5, 120-4) = (115, 116)
    assert sp[0]["queue_segments"] == [(100.0, 104.0), (115.0, 116.0)]
    # start now includes the first turn's queue wait: min(104, 100) = 100
    assert sp[0]["start"] == pytest.approx(100.0)
    # span = 100..120 = 20; active 10; queue 5; others 5
    bd = mod.session_breakdown(sp)[0]
    assert bd["gpu_active"] == pytest.approx(10 / 20)
    assert bd["queue"] == pytest.approx(5 / 20)
    assert bd["others"] == pytest.approx(5 / 20)


def test_session_spans_no_queue_join_counts_queue_as_active(mod):
    # elapsed present but no queue map -> active = elapsed (queue inside)
    turns = [_TS("A", 1, 100.0, 110.0, elapsed=10.0, rid="r1")]
    sp = mod.session_spans(turns)
    assert sp[0]["segments"] == [(100.0, 110.0)]
    assert sp[0]["queue_s"] == 0.0


def test_session_spans_legacy_fallback_without_elapsed(mod):
    # no dynamo timing -> decode-only legacy segment (llm_start..llm_end)
    turns = [_TS("A", 1, 100.0, 110.0)]
    sp = mod.session_spans(turns)
    assert sp[0]["segments"] == [(100.0, 110.0)]


def test_session_utilizations_active_over_span(mod):
    spans = [
        {"session_id": "A", "start": 0.0, "end": 4.0,
         "segments": [(0.0, 1.0), (3.0, 4.0)]},   # active 2 / span 4 = 0.5
        {"session_id": "B", "start": 0.0, "end": 2.0,
         "segments": [(0.0, 2.0)]},                # 1.0
        {"session_id": "C", "start": 5.0, "end": 5.0,
         "segments": [(5.0, 5.0)]},                # span 0 -> skipped
    ]
    assert mod.session_utilizations(spans) == [0.5, 1.0]


def test_percentile_interpolates_and_handles_empty(mod):
    assert mod._percentile([], 90) is None
    assert mod._percentile([0.7], 50) == 0.7
    assert mod._percentile([0.5, 1.0], 90) == pytest.approx(0.95)
    assert mod._percentile([0.0, 0.5, 1.0], 50) == pytest.approx(0.5)


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


def test_order_turns_grouped_interleaved_sessions(mod):
    # A@0, B@1, A@2, B@3 chronologically interleaved -> grouped keeps each
    # session contiguous, ordered by first-turn start (A first).
    turns = [_T("A", 1, 0.0, 0.5), _T("B", 1, 1.0, 1.5),
             _T("A", 2, 2.0, 2.5), _T("B", 2, 3.0, 3.5)]
    ordered, boundaries = mod.order_turns_grouped(turns)
    assert [(t.session_id, t.step) for t in ordered] == \
        [("A", 1), ("A", 2), ("B", 1), ("B", 2)]
    assert boundaries == [0, 2]


def test_order_turns_grouped_missing_start_falls_back_to_end_ts(mod):
    # llm_start_ts is None -> key falls back to llm_end_ts for both the
    # per-turn sort key and the session's first-turn ordinal.
    turns = [_T("A", 1, None, 5.0), _T("B", 1, 1.0, 1.5)]
    ordered, boundaries = mod.order_turns_grouped(turns)
    # B's start (1.0) < A's fallback key (5.0) -> B first
    assert [(t.session_id, t.step) for t in ordered] == [("B", 1), ("A", 1)]
    assert boundaries == [0, 1]


def test_counter_delta_total_sums_workers_and_tolerates_reset(mod, tmp_path):
    def row(ts, worker, val):
        return {"ts": ts, "worker": worker, "ok": True,
                "metrics": {"vllm:x_seconds_sum": [{"labels": {}, "value": val}]}}
    p = tmp_path / "m.ndjson"
    p.write_text(
        json.dumps(row(0.0, "w0", 5.0)) + "\n"
        + json.dumps(row(1.0, "w0", 12.5)) + "\n"      # monotonic: 7.5
        + json.dumps(row(0.0, "w1", 3.0)) + "\n"
        + json.dumps(row(1.0, "w1", 1.0)) + "\n")       # reset: degrades to last (1.0)
    assert mod.counter_delta_total(p, "vllm:x_seconds_sum") == pytest.approx(8.5)


def test_counter_delta_total_window_filter_excludes_ticks(mod, tmp_path):
    def row(ts, val):
        return {"ts": ts, "worker": "w0", "ok": True,
                "metrics": {"vllm:x_seconds_sum": [{"labels": {}, "value": val}]}}
    p = tmp_path / "m.ndjson"
    p.write_text(
        json.dumps(row(0.0, 5.0)) + "\n"
        + json.dumps(row(1.0, 10.0)) + "\n"
        + json.dumps(row(2.0, 20.0)) + "\n")     # excluded by hi=1.5
    assert mod.counter_delta_total(p, "vllm:x_seconds_sum", lo=0.0, hi=1.5) \
        == pytest.approx(5.0)


def test_counter_delta_total_total_suffix_fallback(mod, tmp_path):
    def row(ts, val):
        return {"ts": ts, "worker": "w0", "ok": True,
                "metrics": {"vllm:x_seconds_sum_total": [{"labels": {}, "value": val}]}}
    p = tmp_path / "m.ndjson"
    p.write_text(json.dumps(row(0.0, 1.0)) + "\n" + json.dumps(row(1.0, 4.0)) + "\n")
    assert mod.counter_delta_total(p, "vllm:x_seconds_sum") == pytest.approx(3.0)


def test_counter_delta_total_none_when_absent_and_skips_not_ok(mod, tmp_path):
    p = tmp_path / "m.ndjson"
    p.write_text(
        json.dumps({"ts": 0.0, "worker": "w0", "ok": False,
                    "metrics": {"vllm:x_seconds_sum": [{"labels": {}, "value": 99.0}]}}) + "\n"
        + json.dumps({"ts": 1.0, "worker": "w0", "ok": True,
                      "metrics": {"vllm:other_metric": [{"labels": {}, "value": 1.0}]}}) + "\n")
    assert mod.counter_delta_total(p, "vllm:x_seconds_sum") is None


def test_session_spans_no_elapsed_join_carves_queue_from_bracket_head(mod):
    # elapsed_s is None (no dynamo timing) but the turn DOES have a
    # request_id in queue_ms_by_rid -- the queue wait sits at the FRONT of
    # the client llm bracket, so it is carved out of the bracket's head:
    # (s, s+q) queue / (s+q, e) active. No double count: queue_s equals
    # the drawn queue segment and active shrinks by the same amount.
    class _TQ:
        session_id = "A"
        step = 1
        llm_start_ts = 100.0
        llm_end_ts = 110.0
        elapsed_s = None
        request_id = "r1"
    sp = mod.session_spans([_TQ()], {"r1": 4000.0})
    assert sp[0]["queue_segments"] == [(100.0, 104.0)]
    assert sp[0]["segments"] == [(104.0, 110.0)]
    assert sp[0]["queue_s"] == pytest.approx(4.0)
    bd = mod.session_breakdown(sp)[0]
    assert bd["queue"] == pytest.approx(0.4)
    assert bd["gpu_active"] == pytest.approx(0.6)
    assert bd["others"] == pytest.approx(0.0)


def test_session_spans_no_elapsed_queue_clamped_to_bracket(mod):
    # joined queue (20s) exceeds the whole client bracket (10s): clamp to
    # the bracket so shares still tile the span (active collapses to 0).
    class _TQ:
        session_id = "A"
        step = 1
        llm_start_ts = 100.0
        llm_end_ts = 110.0
        elapsed_s = None
        request_id = "r1"
    sp = mod.session_spans([_TQ()], {"r1": 20000.0})
    assert sp[0]["queue_segments"] == [(100.0, 110.0)]
    assert sp[0]["segments"] == [(110.0, 110.0)]
    assert sp[0]["queue_s"] == pytest.approx(10.0)


def test_session_spans_queued_ts_anchor_beats_collapsed_bracket(mod):
    # BUFFERED turn: start-step fires only after the stream is consumed,
    # so the client bracket collapses (109.9..110.0) even though the true
    # queue wait was 30s. With the SCHED_DELAY queued_ts anchor the queue
    # segment is placed absolutely: (qts, qts+q) then active to llm_end.
    class _TQ:
        session_id = "A"
        step = 1
        llm_start_ts = 109.9
        llm_end_ts = 110.0
        elapsed_s = None
        request_id = "r1"
    sp = mod.session_spans([_TQ()], {"r1": 30000.0}, {"r1": 75.0})
    assert sp[0]["queue_segments"] == [(75.0, 105.0)]
    assert sp[0]["segments"] == [(105.0, 110.0)]
    assert sp[0]["queue_s"] == pytest.approx(30.0)
    assert sp[0]["start"] == pytest.approx(75.0)
    bd = mod.session_breakdown(sp)[0]
    assert bd["queue"] == pytest.approx(30.0 / 35.0)
    assert bd["gpu_active"] == pytest.approx(5.0 / 35.0)


def test_session_spans_queued_ts_anchor_clamps_at_llm_end(mod):
    # qts + queue would run past llm_end (clock skew / over-report):
    # the queue segment clamps at llm_end and active collapses to zero.
    class _TQ:
        session_id = "A"
        step = 1
        llm_start_ts = None
        llm_end_ts = 110.0
        elapsed_s = None
        request_id = "r1"
    sp = mod.session_spans([_TQ()], {"r1": 50000.0}, {"r1": 100.0})
    assert sp[0]["queue_segments"] == [(100.0, 110.0)]
    assert sp[0]["segments"] == [(110.0, 110.0)]
    assert sp[0]["queue_s"] == pytest.approx(10.0)


def test_session_spans_clamps_queue_to_elapsed(mod):
    # queue (6s) EXCEEDS elapsed (4s) -- SCHED_DELAY queue can't exceed the
    # dynamo server wall, so q_eff clamps to elapsed: active == 0, and the
    # queue segment covers the full elapsed window.
    turns = [_TS("A", 1, 100.0, 110.0, elapsed=4.0, rid="r1")]
    sp = mod.session_spans(turns, {"r1": 6000.0})
    assert sp[0]["queue_s"] == pytest.approx(4.0)
    # active = max(0, 4 - 4) = 0 -> segment collapses to a point at e
    assert sp[0]["segments"] == [(110.0, 110.0)]
    assert sp[0]["queue_segments"] == [(106.0, 110.0)]


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


def test_prefix_hit_rate_series_windowed_ratio(mod, tmp_path):
    # per tick sum hits/queries across workers, then delta(hits)/delta(queries)
    def row(ts, worker, hits, queries):
        return {"ts": ts, "worker": worker, "role": "agg", "ok": True,
                "metrics": {
                    "vllm:prefix_cache_hits_total": [{"labels": {}, "value": hits}],
                    "vllm:prefix_cache_queries_total": [{"labels": {}, "value": queries}]}}
    p = tmp_path / "m.ndjson"
    p.write_text(
        # tick 10: two workers -> hits 100, queries 200
        json.dumps(row(10.0, "a0", 60, 120)) + "\n"
        + json.dumps(row(10.0, "a1", 40, 80)) + "\n"
        # tick 11: hits 100+80=180, queries 200+100=300 ->
        #          delta hits 80 / delta queries 100 = 0.8
        + json.dumps(row(11.0, "a0", 100, 160)) + "\n"
        + json.dumps(row(11.0, "a1", 80, 140)) + "\n")
    assert mod.prefix_hit_rate_series(p) == [(11.0, pytest.approx(0.8))]


def test_prefix_hit_rate_series_skips_counter_reset(mod, tmp_path):
    def row(ts, hits, queries):
        return {"ts": ts, "ok": True, "metrics": {
            "vllm:prefix_cache_hits_total": [{"labels": {}, "value": hits}],
            "vllm:prefix_cache_queries_total": [{"labels": {}, "value": queries}]}}
    p = tmp_path / "m.ndjson"
    p.write_text(
        json.dumps(row(10.0, 100, 200)) + "\n"
        # worker restarted: counters reset lower -> tick 11 must be skipped
        # (dh < 0), not produce a negative/garbage ratio
        + json.dumps(row(11.0, 5, 10)) + "\n"
        + json.dumps(row(12.0, 25, 40)) + "\n")
    # tick 12 measured against tick 11's raw values: dh=20, dq=30
    assert mod.prefix_hit_rate_series(p) == [(12.0, pytest.approx(20 / 30))]


def test_prefix_hit_rate_series_empty_without_counters(mod, tmp_path):
    p = tmp_path / "m.ndjson"
    p.write_text(json.dumps({"ts": 1, "ok": True, "metrics": {
        "vllm:kv_cache_usage_perc": [{"labels": {}, "value": 0.5}]}}) + "\n")
    assert mod.prefix_hit_rate_series(p) == []


def test_lmcache_metric_names_collects_lmcache_prefix_only(mod, tmp_path):
    p = tmp_path / "m.ndjson"
    p.write_text(
        json.dumps({"ts": 1, "ok": True, "metrics": {
            "vllm:x": [{"labels": {}, "value": 1}],
            "lmcache:a": [{"labels": {}, "value": 1}]}}) + "\n"
        + json.dumps({"ts": 2, "ok": True, "metrics": {
            "lmcache:b": [{"labels": {}, "value": 1}]}}) + "\n"
        # not-ok rows are ignored
        + json.dumps({"ts": 3, "ok": False, "metrics": {
            "lmcache:c": [{"labels": {}, "value": 1}]}}) + "\n")
    assert mod.lmcache_metric_names(p) == {"lmcache:a", "lmcache:b"}


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


def test_lmcache_series_tolerates_openmetrics_total_suffix(mod, tmp_path):
    # the installed LMCache exposes counters with the OpenMetrics `_total`
    # suffix (num_hit_tokens_total, local_cpu_evict_keys_count_total, ...);
    # _counter_series must pick those up so fig8/eviction aren't empty.
    def row(ts, hit, ev):
        return {"ts": ts, "worker": "a0", "role": "agg", "ok": True,
                "metrics": {
                    "lmcache:num_hit_tokens_total": [{"labels": {}, "value": hit}],
                    "lmcache:num_stored_tokens_total": [{"labels": {}, "value": 0}],
                    "lmcache:local_cpu_evict_keys_count_total": [{"labels": {}, "value": ev}],
                    "lmcache:local_cpu_evict_failed_count_total": [{"labels": {}, "value": 0}]}}
    p = tmp_path / "m.ndjson"
    p.write_text(json.dumps(row(10.0, 300, 10)) + "\n"
                 + json.dumps(row(11.0, 500, 40)) + "\n")
    lm = mod.lmcache_series(p)
    assert [r["hit_tokens"] for r in lm["a0"]] == [300, 500]
    assert [r["evict_keys"] for r in lm["a0"]] == [10, 40]
    # and the diagnostic counts them as present despite the suffix
    assert "lmcache:num_hit_tokens" not in mod.lmcache_missing_metrics(
        mod.lmcache_metric_names(p))


def test_lmcache_series_reads_hit_rate_gauges(mod, tmp_path):
    # LMCache tier hit-rate gauges (retrieve/lookup, 0-1) surface on fig7
    def row(ts, rhr, lhr):
        return {"ts": ts, "worker": "a0", "role": "agg", "ok": True,
                "metrics": {
                    "lmcache:retrieve_hit_rate": [{"labels": {}, "value": rhr}],
                    "lmcache:lookup_hit_rate": [{"labels": {}, "value": lhr}]}}
    p = tmp_path / "m.ndjson"
    p.write_text(json.dumps(row(10.0, 0.2, 0.3)) + "\n"
                 + json.dumps(row(11.0, 0.5, 0.4)) + "\n")
    lm = mod.lmcache_series(p)
    assert [r["retrieve_hit_rate"] for r in lm["a0"]] == [0.2, 0.5]
    assert [r["lookup_hit_rate"] for r in lm["a0"]] == [0.3, 0.4]


def test_lmcache_missing_metrics_flags_truly_absent(mod, tmp_path):
    p = tmp_path / "m.ndjson"
    # only a gauge present -> counters + histograms all reported missing
    p.write_text(json.dumps({"ts": 1, "ok": True, "metrics": {
        "lmcache:local_cache_usage": [{"labels": {}, "value": 1}]}}) + "\n")
    missing = mod.lmcache_missing_metrics(mod.lmcache_metric_names(p))
    assert "lmcache:num_hit_tokens" in missing
    assert "lmcache:local_cache_usage" not in missing


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
