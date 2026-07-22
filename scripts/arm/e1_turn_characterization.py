#!/usr/bin/env python3
"""E1 turn characterization: the E0 views adapted to CONCURRENT runs
(max-in-flight > 1, Poisson arrivals).

Under concurrency, samples overlap in time, which breaks two E0
assumptions:
  * E0's sample-boundary detection treats a session that starts inside
    another's window as a nested helper — with overlap that erases every
    boundary. E1 instead marks each trace session's FIRST turn (turns are
    already trace-filtered to main sessions, so every session IS a sample).
  * E0's fig3 breaks a session's line at ordinal discontinuities (the
    sequential pause semantics) — with interleaving, no session has
    consecutive ordinals, so that left only isolated dots. E1 connects
    each session's points across the interleave.

Everything else (fig1 panels, fig2 CDF, fig4 reuse-vs-gap) reuses the E0
implementations. fig3 top = cached-vs-reused per turn (cached tokens vs
hit tokens, eviction gap in red); fig3 bottom = GPU KV-usage (left y) +
the vLLM prefix-cache hit rate (right y), both from the scrape NDJSON.
fig8 (needs --logs [+ --frontend]) plots per-request engine queue wait
vs time and its share of the frontend elapsed_ms (request_id join).
fig9 (needs --logs + --frontend) joins each LMCache retrieve transfer
(worker-log "Retrieved ... cost N ms" lines) to the same request's
frontend ttft_ms — how much of TTFT was host->device KV onboarding.

Usage:
  scripts/arm/e1_turn_characterization.py \\
      --profiles <workspace_root>/profiles \\
      --trace results/<run>/trace.jsonl \\
      [--metrics logs/vllm_metrics.ndjson] \\
      [--cpu-cache-gb N] [--disk-cache-gb N] [--out <dir>] [--no-figures]

When the scrape NDJSON carries lmcache:* metrics (an LMCache offload run),
fig7_lmcache is a 3-panel figure: (1) KV Cache Tier Occupancy — host CPU /
disk tier usage + CPU-tier eviction rate + LMCache hit rate (GPU KV usage
is in fig3, not repeated here); (2) transfer speed (retrieve = onboard,
store = offload; window-avg tokens/sec); (3) transfer time (seconds to
move the transferred tokens per window = tokens / speed).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

_E0_PATH = Path(__file__).resolve().parent / "e0_turn_characterization.py"


def _load_e0():
    spec = importlib.util.spec_from_file_location("e0_turn_characterization",
                                                  _E0_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e0_turn_characterization"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- concurrent-run sample boundaries (no overlap filter) ----------


def session_first_ordinals(ordered: list) -> list[int]:
    """Ordinal of each session's FIRST turn — the sample boundaries when
    `ordered` is already trace-filtered to main sessions. No overlap
    filter: under a concurrent run samples legitimately overlap in time,
    and E0's top-level window filter would misread all but the first as
    nested, erasing every boundary."""
    seen: set[str] = set()
    out: list[int] = []
    for i, t in enumerate(ordered):
        if t.session_id not in seen:
            seen.add(t.session_id)
            out.append(i)
    return out


def order_turns_grouped(turns: list) -> tuple[list, list[int]]:
    """Session-grouped ordering for per-turn ORDINAL figures under a
    CONCURRENT run: sessions in first-turn start order, each session's
    turns contiguous (chronological within the session). Returns
    (ordered, boundaries) with boundaries = ordinal of each session's
    first turn. Under mif>1 the global chronological order interleaves
    sessions, so first-turn ordinals stop being block edges — grouping
    restores contiguous per-session blocks (same convention as fig3's
    top panel)."""
    def key(t):
        return (t.llm_start_ts if t.llm_start_ts is not None
                else t.llm_end_ts if t.llm_end_ts is not None
                else float("inf"))
    by: dict[str, list] = {}
    for t in turns:
        by.setdefault(t.session_id, []).append(t)
    sess = sorted(by, key=lambda s: min(key(t) for t in by[s]))
    ordered: list = []
    boundaries: list[int] = []
    for sid in sess:
        boundaries.append(len(ordered))
        ordered.extend(sorted(by[sid], key=key))
    return ordered, boundaries


def session_first_times_abs(ordered: list) -> list[float]:
    """ABSOLUTE start ts of each session's first turn (no overlap filter,
    see session_first_ordinals)."""
    first: dict[str, float] = {}
    for t in ordered:
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if st is not None and t.session_id not in first:
            first[t.session_id] = st
    return sorted(first.values())


# ---------- LMCache tier occupancy + transfer speed (scrape NDJSON) ----------


def _sum_series(row: dict, name: str) -> float | None:
    series = (row.get("metrics") or {}).get(name)
    if not series:
        return None
    vals = [e.get("value") for e in series
            if isinstance(e.get("value"), (int, float))
            and math.isfinite(e["value"])]
    return sum(vals) if vals else None


def _counter_series(row: dict, base: str) -> float | None:
    """Sum a Prometheus counter, tolerating the OpenMetrics `_total`
    suffix: newer prometheus_client (what LMCache ships) exposes
    `lmcache:num_hit_tokens_total`, older builds `lmcache:num_hit_tokens`.
    Try the bare name first, then `_total`."""
    v = _sum_series(row, base)
    return v if v is not None else _sum_series(row, base + "_total")


def lmcache_series(metrics_path: Path) -> dict[str, list[dict]]:
    """Per-worker LMCache series from any scrape row carrying lmcache:*
    metrics (LMCache rides the worker's own /metrics, so these appear on
    the normal vLLM target rows, NOT a separate role like KVBM).

    Returns {worker: [rec, ...]} sorted by ts, each rec:
      ts, local_usage_bytes  (lmcache:local_cache_usage  = CPU tier occupancy)
          storage_usage_bytes(lmcache:local_storage_usage = disk tier occupancy)
          retrieve_sum/retrieve_count (lmcache:retrieve_speed histogram
              _sum/_count; retrieve = host->device onboard, tokens/sec)
          store_sum/store_count       (lmcache:store_speed histogram
              _sum/_count; store = device->host offload, tokens/sec)
    Window-average speed is derived downstream as delta(_sum)/delta(_count)
    between successive scrape ticks. Metric names are LMCache's (verified
    against a local LMCache checkout; re-confirm against the installed
    version via the first run's vllm_metrics.ndjson)."""
    out: dict[str, list[dict]] = {}
    with metrics_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("ok"):
                continue
            metrics = row.get("metrics") or {}
            if not any(k.startswith("lmcache:") for k in metrics):
                continue
            ts = row.get("ts")
            if ts is None:
                continue
            out.setdefault(str(row.get("worker", "?")), []).append({
                "ts": float(ts),
                "local_usage_bytes": _sum_series(row, "lmcache:local_cache_usage"),
                "storage_usage_bytes": _sum_series(row, "lmcache:local_storage_usage"),
                "retrieve_sum": _sum_series(row, "lmcache:retrieve_speed_sum"),
                "retrieve_count": _sum_series(row, "lmcache:retrieve_speed_count"),
                "store_sum": _sum_series(row, "lmcache:store_speed_sum"),
                "store_count": _sum_series(row, "lmcache:store_speed_count"),
                # CPU-tier eviction counters (host tier full -> LRU drop):
                # evict_keys = chunks evicted, evict_failed = allocate
                # attempts that found NO evictable candidate (pure pressure).
                # Counters -> OpenMetrics `_total` suffix (see _counter_series).
                "evict_keys": _counter_series(row, "lmcache:local_cpu_evict_keys_count"),
                "evict_failed": _counter_series(row, "lmcache:local_cpu_evict_failed_count"),
                # transferred-token counters (pairs with the speed
                # histograms to derive transfer TIME): hit = tokens
                # retrieved host->device, stored = tokens offloaded
                # device->host.
                "hit_tokens": _counter_series(row, "lmcache:num_hit_tokens"),
                "stored_tokens": _counter_series(row, "lmcache:num_stored_tokens"),
                # LMCache hit-rate gauges (0-1, sliding window): retrieve =
                # fraction of retrieve requests served from the tier, lookup
                # = fraction of lookups that found a match. The tier-side
                # analogue of fig3-1's GPU prefix-cache hit rate.
                "retrieve_hit_rate": _sum_series(row, "lmcache:retrieve_hit_rate"),
                "lookup_hit_rate": _sum_series(row, "lmcache:lookup_hit_rate"),
            })
    for recs in out.values():
        recs.sort(key=lambda r: r["ts"])
    return out


# The lmcache:* names fig7/fig8 consume. Kept here (not buried in
# lmcache_series) so the diagnostic can report which ones the run is
# actually missing -- LMCache metric names drift across versions, and a
# renamed histogram/counter silently empties a panel. Counters carry an
# OpenMetrics `_total` suffix on newer prometheus_client (a name is
# considered present if EITHER the bare or the `_total` form appears; see
# _counter_series / lmcache_missing_metrics).
# Only the names fig7's CURRENT panels consume (host usage + hit rate +
# transfer speed/time). Disk usage, eviction counters, and retrieve_hit_rate
# are still parsed by lmcache_series (available if re-enabled) but not
# required, so they are omitted here to keep the diagnostic quiet.
LMCACHE_EXPECTED_METRICS = (
    "lmcache:local_cache_usage",         # gauge (host tier usage)
    "lmcache:retrieve_speed_sum",        # histogram (transfer speed)
    "lmcache:retrieve_speed_count",
    "lmcache:store_speed_sum",
    "lmcache:store_speed_count",
    "lmcache:num_hit_tokens",            # counter (+/- _total; transfer time)
    "lmcache:num_stored_tokens",
    "lmcache:lookup_hit_rate",           # gauge (LMCache hit %)
)


def lmcache_missing_metrics(seen: set[str]) -> list[str]:
    """Expected names absent from `seen`, tolerating the counter `_total`
    suffix (a bare expected name is satisfied by either form)."""
    return [n for n in LMCACHE_EXPECTED_METRICS
            if n not in seen and (n + "_total") not in seen]


def lmcache_metric_names(metrics_path: Path) -> set[str]:
    """Every distinct lmcache:* metric name present in the scrape NDJSON —
    the ground truth for reconciling LMCACHE_EXPECTED_METRICS against the
    installed LMCache version when a panel comes up empty."""
    names: set[str] = set()
    with metrics_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("ok"):
                continue
            names.update(k for k in (row.get("metrics") or {})
                         if k.startswith("lmcache:"))
    return names


def counter_rate(recs: list[dict], key: str) -> list[tuple[float, float]]:
    """Per-window rate (units/sec) of a cumulative counter:
    delta(value)/delta(ts), stamped at the later tick. Skips negative
    deltas (counter reset), non-positive dt, and breaks the chain on a
    None gap. Used for LMCache eviction counters (chunks evicted/sec)."""
    out: list[tuple[float, float]] = []
    prev = None
    for r in recs:
        v = r.get(key)
        if v is None:
            prev = None
            continue
        if prev is not None:
            dv, dt = v - prev[1], r["ts"] - prev[0]
            if dt > 0 and dv >= 0:
                out.append((r["ts"], dv / dt))
        prev = (r["ts"], v)
    return out


def counter_delta_total(metrics_path: Path, name: str,
                        lo: float | None = None,
                        hi: float | None = None) -> float | None:
    """Total increase of a cumulative counter over the [lo, hi] window,
    summed across workers, from the scrape NDJSON. Tries the bare name
    then the OpenMetrics `_total` variant. Per worker: last - first value
    inside the window; a counter reset (last < first) degrades to the
    last value. None when the metric never appears in the window."""
    per: dict[str, tuple[float, float]] = {}   # worker -> (first, last)
    with metrics_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("ok"):
                continue
            ts = row.get("ts")
            if ts is None or (lo is not None and ts < lo) \
                    or (hi is not None and ts > hi):
                continue
            v = _sum_series(row, name)
            if v is None:
                v = _sum_series(row, name + "_total")
            if v is None:
                continue
            w = str(row.get("worker", "?"))
            first, _last = per.get(w, (v, v))
            per[w] = (first, v)
    if not per:
        return None
    tot = 0.0
    for first, last in per.values():
        tot += (last - first) if last >= first else last
    return tot


def transfer_batches(recs: list[dict], tok_key: str, sum_key: str,
                     count_key: str) -> list[tuple[float, float, float]]:
    """Per window: (ts, tokens_transferred, seconds_to_transfer).

    tokens = delta(tok_key) actually moved in the window; the window's
    mean speed = delta(_sum)/delta(_count) tokens/sec from the paired
    speed histogram; seconds = tokens / speed = how long it took to
    transfer THOSE tokens. Skips windows with no completed ops
    (delta_count <= 0) or non-positive speed, and breaks the chain on a
    None gap (any of the three series missing)."""
    out: list[tuple[float, float, float]] = []
    prev = None
    for r in recs:
        tok, s, c = r.get(tok_key), r.get(sum_key), r.get(count_key)
        if tok is None or s is None or c is None:
            prev = None
            continue
        if prev is not None:
            dtok, ds, dc = tok - prev[1], s - prev[2], c - prev[3]
            if dc > 0 and ds > 0 and dtok >= 0:
                speed = ds / dc
                out.append((r["ts"], dtok, dtok / speed))
        prev = (r["ts"], tok, s, c)
    return out


def window_avg_speed(recs: list[dict], sum_key: str,
                     count_key: str) -> list[tuple[float, float]]:
    """Per-window mean transfer speed (tokens/sec) from a Prometheus
    histogram's _sum/_count: between two ticks, delta(_sum)/delta(_count)
    is the mean of the operations that completed in that window. Points
    are stamped at the LATER tick. Skips windows with no new operations
    (delta_count <= 0) and any tick missing the series."""
    out: list[tuple[float, float]] = []
    prev = None
    for r in recs:
        s, c = r.get(sum_key), r.get(count_key)
        if s is None or c is None:
            prev = None      # break the delta chain across a gap
            continue
        if prev is not None:
            ds, dc = s - prev[1], c - prev[2]
            if dc > 0 and ds >= 0:
                out.append((r["ts"], ds / dc))
        prev = (r["ts"], s, c)
    return out


# ---------- eviction evidence ----------


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sxx ** 0.5 * syy ** 0.5)


def eviction_events(turns: list, compaction_drop_ratio: float = 0.6,
                    min_shortfall: int = 128) -> list[dict]:
    """Per-turn re-use shortfall, labeled eviction vs compaction.

    For turn N with a resolvable previous turn N-1 in the SAME session:
      prev_cached = effective_input(N-1) + output(N-1)   # KV that WAS cached
      shortfall   = prev_cached - cache_read(N)           # what it failed to reuse
    A shortfall means the session did NOT reuse KV it had produced. WHY:
      * compaction — the prompt SHRANK (effective_input(N) < ratio *
        effective_input(N-1)); opencode summarized the history.
      * eviction   — the prompt did NOT shrink (still growing) yet the
        middle KV was gone, so it had to be re-prefilled. This is the LRU
        overwrite we want to prove; GPU-usage never hitting 100% does not
        rule it out because freed-but-cached blocks aren't counted in usage.
    `away_displaced_tokens` (KV other sessions allocated during this turn's
    away window) is the displacement pressure that drives the eviction."""
    by_key = {(t.session_id, t.step): t for t in turns}
    out: list[dict] = []
    for t in turns:
        prev = by_key.get((t.session_id, t.step - 1))
        if prev is None:
            continue
        pe = prev.effective_input
        if pe is None:
            continue
        # prev_cached = KV from turn N-1 that STAYS reusable in turn N's
        # prompt = effective_input(N-1) + non-reasoning output(N-1).
        # opencode drops prior-turn reasoning from the next prompt, so
        # reasoning tokens are never re-fed and must not count as
        # "cached but not reused" (else every turn shows a phantom miss,
        # even at mif=1 where no real eviction can happen).
        prev_out = (prev.output_tokens or 0) - getattr(prev, "reasoning_tokens", 0)
        prev_cached = pe + max(0, prev_out)
        if prev_cached <= 0:
            continue
        cr = t.cache_read or 0
        shortfall = prev_cached - cr
        ce = t.effective_input
        compaction = (ce is not None and pe > 0
                      and ce < compaction_drop_ratio * pe)
        is_evict = (not compaction) and shortfall > min_shortfall
        out.append({
            "session_id": t.session_id,
            "step": t.step,
            "prev_cached": prev_cached,
            "cache_read": cr,
            "shortfall": shortfall,
            "eff_prev": pe,
            "eff_cur": ce,
            "label": "compaction" if compaction
                     else ("eviction" if is_evict else "ok"),
            "away_s": t.away_s,
            "displaced": t.away_displaced_tokens,
        })
    return out


def eviction_loss_pct(events: list[dict]) -> tuple[float, float, float | None]:
    """(missed, total_cached, pct): missed = tokens that had been cached
    but were re-prefilled because of EVICTION (sum of eviction-turn
    shortfall, the red gap in fig3's top panel); total_cached = all KV
    that WAS cached and available for reuse across turns (sum of
    prev_cached over every resolved turn). pct = 100*missed/total_cached
    = share of reusable KV lost to eviction (None if nothing cached)."""
    total_cached = sum(e["prev_cached"] for e in events)
    missed = sum(e["shortfall"] for e in events if e["label"] == "eviction")
    pct = (100.0 * missed / total_cached) if total_cached > 0 else None
    return missed, total_cached, pct


def prefix_mismatch_pct(events: list[dict]) -> tuple[float, float, float | None]:
    """(missed, reusable, pct): prefix-mismatch rate — of the reusable KV
    across NON-compaction turns (compaction is a legitimate prompt shrink,
    not a mismatch), what fraction failed to be prefix-matched and had to
    be re-prefilled. missed = sum of positive shortfall, reusable = sum of
    prev_cached, over label != 'compaction'. With prior-turn reasoning
    already excluded from prev_cached, this is the true prefix-match miss
    rate (reasoning drop no longer inflates it)."""
    rel = [e for e in events if e["label"] != "compaction"]
    reusable = sum(e["prev_cached"] for e in rel)
    missed = sum(max(0, e["shortfall"]) for e in rel)
    pct = (100.0 * missed / reusable) if reusable > 0 else None
    return missed, reusable, pct


def session_spans(turns: list,
                  queue_ms_by_rid: dict[str, float] | None = None,
                  queued_ts_by_rid: dict[str, float] | None = None,
                  ) -> list[dict]:
    """Per session: {session_id, start, end, segments, queue_segments,
    queue_s} ordered by start. Segments are per-turn GPU-ACTIVE intervals
    = PREFILL + DECODE: anchored at llm_end and extending back by (dynamo
    elapsed_s - engine queue wait), where queue wait comes from the
    SCHED_DELAY join (queue_ms_by_rid, keyed by request_id).
    queue_segments are the per-turn engine QUEUE-WAIT intervals sitting
    immediately before their active segment: [llm_end - elapsed,
    llm_end - (elapsed - queue)] (empty when no join). Fallback chain per
    turn:
      elapsed - queue  ->  elapsed (queue unknown: queue counted as active)
      -> SCHED_DELAY queued_ts anchor (no dynamo timing but
         queued_ts_by_rid joined): queue = (qts, qts+q), active =
         (qts+q, llm_end) — robust against buffered turns where the
         client bracket collapses
      -> llm_end - llm_start (client llm bracket; with a queue join the
         bracket is split (s, s+q) queue / (s+q, e) active; without a
         join the whole bracket counts as active).
    queue_s = summed engine queue wait actually carved out of active
    (per-turn clamped to the bracket it was carved from; 0.0 when no
    join), so it always equals the drawn queue_segments total and the 3
    breakdown shares tile the span. span - active - queue = others
    (tool + scaffold)."""
    by: dict[str, list[tuple[float, float]]] = {}
    qsegs_by_sess: dict[str, list[tuple[float, float]]] = {}
    queue_by_sess: dict[str, float] = {}
    for t in turns:
        s, e = t.llm_start_ts, t.llm_end_ts
        if e is None:
            continue
        q_s = None
        rid = getattr(t, "request_id", None)
        if queue_ms_by_rid and rid and rid in queue_ms_by_rid:
            q_s = queue_ms_by_rid[rid] / 1000.0
        elapsed = getattr(t, "elapsed_s", None)
        if elapsed is not None:
            # clamp: SCHED_DELAY queue can't exceed dynamo's server wall
            q_eff = min(q_s, elapsed) if q_s else 0.0
            active = max(0.0, elapsed - q_eff)
            seg = (e - active, e)
            if q_eff > 0:
                qsegs_by_sess.setdefault(t.session_id, []).append(
                    (e - elapsed, e - active))
                queue_by_sess[t.session_id] = \
                    queue_by_sess.get(t.session_id, 0.0) + q_eff
        else:
            # No dynamo timing. PREFERRED: anchor on the SCHED_DELAY
            # absolute queued_ts — queue = (qts, qts+q), active
            # (prefill+decode) = (qts+q, llm_end). This is immune to the
            # buffered-turn collapse of the client llm bracket (start-step
            # fires only after the stream is consumed, so llm.start ≈
            # llm.end and a bracket-clamped queue leaks into others).
            qts = None
            if queued_ts_by_rid and rid and rid in queued_ts_by_rid:
                qts = queued_ts_by_rid[rid]
            if q_s and qts is not None and qts < e:
                q_end = min(qts + q_s, e)
                qsegs_by_sess.setdefault(t.session_id, []).append(
                    (qts, q_end))
                queue_by_sess[t.session_id] = \
                    queue_by_sess.get(t.session_id, 0.0) + (q_end - qts)
                seg = (q_end, e)
            elif s is not None and q_s and e > s:
                # bracket-head carve fallback (no queued_ts): clamp to
                # the client llm bracket.
                q_eff = min(q_s, e - s)
                qsegs_by_sess.setdefault(t.session_id, []).append(
                    (s, s + q_eff))
                queue_by_sess[t.session_id] = \
                    queue_by_sess.get(t.session_id, 0.0) + q_eff
                seg = (s + q_eff, e)
            elif s is not None:
                seg = (s, e)
            else:
                continue
        by.setdefault(t.session_id, []).append(seg)
    out: list[dict] = []
    for sid, segs in by.items():
        segs = sorted(segs)
        qsegs = sorted(qsegs_by_sess.get(sid, []))
        # the first turn's queue wait precedes its active segment; include
        # it in the span so the 3 shares tile the bar.
        start = min(segs[0][0], qsegs[0][0]) if qsegs else segs[0][0]
        out.append({"session_id": sid, "start": start,
                    "end": max(e for _, e in segs), "segments": segs,
                    "queue_segments": qsegs,
                    "queue_s": queue_by_sess.get(sid, 0.0)})
    out.sort(key=lambda d: d["start"])
    return out


def session_utilizations(spans: list[dict]) -> list[float]:
    """Per-session utilization = LLM-active time / total span (active =
    prefill + decode when the spans were built with dynamo timing/queue
    join; see session_spans). Sessions with a non-positive span skipped."""
    out: list[float] = []
    for sp in spans:
        span = sp["end"] - sp["start"]
        if span <= 0:
            continue
        active = sum(e - s for s, e in sp["segments"])
        out.append(active / span)
    return out


def session_breakdown(spans: list[dict]) -> list[dict]:
    """Per-session 3-way share of the wall span:
      gpu_active = prefill + decode          (the segments)
      queue      = engine queue wait          (SCHED_DELAY join; 0 without --logs)
      others     = tool + scaffold            (span - active - queue, clamped;
                   scaffold = post overhead + agent-loop plumbing)
    Each entry: {session_id, span_s, gpu_active, queue, others} with the
    three shares summing to <= 1 (clamp guards timing noise). Sessions
    with a non-positive span are skipped."""
    out: list[dict] = []
    for sp in spans:
        span = sp["end"] - sp["start"]
        if span <= 0:
            continue
        active = sum(e - s for s, e in sp["segments"])
        queue = sp.get("queue_s", 0.0)
        others = max(0.0, span - active - queue)
        out.append({"session_id": sp["session_id"], "span_s": span,
                    "gpu_active": active / span, "queue": queue / span,
                    "others": others / span})
    return out


def _percentile(xs: list[float], p: float) -> float | None:
    """Linear-interpolated p-th percentile (0-100) of xs, or None if empty."""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _draw_mean_p50(ax, vals: list[float], unit: str = "") -> None:
    """Dashed red mean + dotted purple p50 horizontal lines on `ax`,
    labeled with their values at the right edge (y-axis transform, so the
    x-position is axis-relative). No-op on empty vals."""
    if not vals:
        return
    mean = sum(vals) / len(vals)
    p50 = _percentile(vals, 50)
    tr = ax.get_yaxis_transform()
    ax.axhline(mean, color="tab:red", lw=1.0, ls="--", zorder=3)
    ax.text(0.995, mean, f" mean {mean:.2f}{unit}", color="tab:red",
            fontsize=8, va="bottom", ha="right", transform=tr)
    ax.axhline(p50, color="tab:purple", lw=1.0, ls=":", zorder=3)
    ax.text(0.995, p50, f" p50 {p50:.2f}{unit}", color="tab:purple",
            fontsize=8, va="top", ha="right", transform=tr)


_ARW_PATH = Path(__file__).resolve().parents[1] / "analyze_request_wait.py"


def _load_arw():
    spec = importlib.util.spec_from_file_location("analyze_request_wait",
                                                  _ARW_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_request_wait"] = mod
    spec.loader.exec_module(mod)
    return mod


# LMCache per-request retrieve join (vllm-*.log). Two line shapes:
#   lookup (vllm_v1_adapter, repeated once per scheduler step while the
#   request sits in WAITING — dedup by Reqid):
#     LMCache INFO: Reqid: <rid>-<suffix>, Total tokens N, Inference
#     Engine computed tokens: C, LMCache hit tokens: H, need to load: L
#   retrieve completion (cache_engine, once per actual transfer; carries
#   NO Reqid, so it is attributed to the most recent NEW lookup whose
#   hit-token count matches "from H total tokens"):
#     Retrieved R out of Q required tokens (from H total tokens).
#     size: S gb, cost M ms, throughput: T GB/s
_LMC_LOOKUP_RE = re.compile(
    r"LMCache INFO: Reqid: (?P<reqid>\S+?),\s+Total tokens (?P<total>\d+).*?"
    r"LMCache hit tokens: (?P<hit>\d+),\s+need to load: (?P<need>\d+)")
_LMC_RETR_RE = re.compile(
    r"Retrieved (?P<got>\d+) out of (?P<req>\d+) required tokens "
    r"\(from (?P<hit>\d+) total tokens\)\..*?cost (?P<cost_ms>[\d.]+) ms")


_LMC_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJ]")
# LMCache log-line timestamp: [2026-07-18 16:18:42,810]
_LMC_TS_RE = re.compile(
    r"\[(?P<d>\d{4}-\d{2}-\d{2}) (?P<t>\d{2}:\d{2}:\d{2}),(?P<ms>\d{3})\]")


def _lmc_line_ts(line: str) -> float | None:
    from datetime import datetime
    m = _LMC_TS_RE.search(line)
    if not m:
        return None
    try:
        dt = datetime.strptime(f"{m.group('d')} {m.group('t')}",
                               "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return dt.timestamp() + int(m.group("ms")) / 1000.0


def parse_lmcache_retrieves(logs: Path,
                            stats: dict | None = None) -> dict[str, dict]:
    """{request_id: {cost_ms, tokens, hit_tokens, waits}} per request from
    the worker logs. request_id = the lookup Reqid with the engine's
    trailing '-<suffix>' stripped (the remaining 5-group UUID equals the
    frontend/SCHED_DELAY Context UUID). `waits` counts the repeated
    lookup lines for the Reqid (~ scheduler steps spent in WAITING before
    the transfer). A Retrieved line is attributed to the most recent
    lookup whose hit count matches its "from N total tokens", falling
    back to the most recent lookup of ANY hit count (hit counts can
    shift between lookup and transfer when more chunks land in the
    meantime); only retrieves with no prior lookup at all are dropped.
    Last retrieve wins per request. `stats` (optional dict) receives
    {lookup_lines, retrieve_lines, matched, fallback_matched} for the
    caller's diagnostics.

    `pre_wait_ms` (when both line timestamps parse): first lookup line ts
    -> Retrieved line ts, minus the transfer cost itself = the WAITING-
    side overhead BEFORE the transfer (GPU KV block allocation / eviction
    + scheduler re-tries), clamped >= 0. None when timestamps missing."""
    out: dict[str, dict] = {}
    st = stats if stats is not None else {}
    st.setdefault("lookup_lines", 0)
    st.setdefault("retrieve_lines", 0)
    st.setdefault("matched", 0)
    st.setdefault("fallback_matched", 0)
    files = [logs] if logs.is_file() else sorted(logs.glob("vllm-*.log"))
    for fpath in files:
        last_by_hit: dict[int, str] = {}     # hit tokens -> reqid
        last_reqid: str | None = None
        waits: dict[str, int] = {}
        first_ts: dict[str, float] = {}      # reqid -> first lookup line ts
        with fpath.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = _LMC_ANSI_RE.sub("", line)
                m = _LMC_LOOKUP_RE.search(line)
                if m:
                    reqid = m.group("reqid")
                    st["lookup_lines"] += 1
                    waits[reqid] = waits.get(reqid, 0) + 1
                    if reqid not in first_ts:
                        lts = _lmc_line_ts(line)
                        if lts is not None:
                            first_ts[reqid] = lts
                    last_by_hit[int(m.group("hit"))] = reqid
                    last_reqid = reqid
                    continue
                m = _LMC_RETR_RE.search(line)
                if not m:
                    continue
                st["retrieve_lines"] += 1
                reqid = last_by_hit.get(int(m.group("hit")))
                if reqid is None:
                    if last_reqid is None:
                        continue
                    reqid = last_reqid
                    st["fallback_matched"] += 1
                else:
                    st["matched"] += 1
                rid = reqid.rsplit("-", 1)[0]
                cost_ms = float(m.group("cost_ms"))
                pre_wait_ms = None
                rts = _lmc_line_ts(line)
                if rts is not None and reqid in first_ts:
                    pre_wait_ms = max(
                        0.0, (rts - first_ts[reqid]) * 1000.0 - cost_ms)
                out[rid] = {
                    "cost_ms": cost_ms,
                    "tokens": int(m.group("got")),
                    "hit_tokens": int(m.group("hit")),
                    "waits": waits.get(reqid, 0),
                    "pre_wait_ms": pre_wait_ms,
                }
    return out


def _stat_line(name: str, vals: list[float], fmt: str = "{:.3f}") -> None:
    if not vals:
        print(f"  {name:<26} (no data)")
        return
    mean = sum(vals) / len(vals)
    print(f"  {name:<26} mean {fmt.format(mean)}  "
          f"p50 {fmt.format(_percentile(vals, 50))}  "
          f"p90 {fmt.format(_percentile(vals, 90))}  "
          f"p99 {fmt.format(_percentile(vals, 99))}  n={len(vals)}")


def fig_queue_share(e0, sched: dict, frontend: dict, path: Path) -> None:
    """fig8: per-request engine queue wait joined to the frontend by
    request_id. Top panel: queue_ms vs time (x = SCHED_DELAY queued_ts).
    Bottom panel: queue_ms / frontend elapsed_ms share vs time — how much
    of each request's end-to-end wall was scheduler queue. Stats printed
    by the caller."""
    plt = e0._mpl()
    pts = []                                   # (queued_ts, queue_ms, share)
    for rid, rec in sched.items():
        fr = frontend.get(rid)
        qts = rec.anchor_ts
        q = rec.total_queue_ms
        if qts is None or q <= 0:
            continue
        share = (q / fr.total_ms) if fr and fr.total_ms > 0 else None
        pts.append((qts, q, share))
    fig, (ax_q, ax_s) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    if not pts:
        ax_q.text(0.5, 0.5, "no SCHED_DELAY records", transform=ax_q.transAxes,
                  ha="center", va="center", color="grey")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    pts.sort()
    t0 = pts[0][0]
    xs = [p[0] - t0 for p in pts]
    ax_q.scatter(xs, [p[1] / 1000.0 for p in pts], s=6, alpha=0.5,
                 color="tab:orange")
    ax_q.set_yscale("log")
    ax_q.set_ylabel("queue wait (s, log)")
    ax_q.set_title("Engine queue wait per request vs time")
    _draw_mean_p50(ax_q, [p[1] / 1000.0 for p in pts], "s")
    sh = [(x, p[2]) for x, p in zip(xs, pts) if p[2] is not None]
    if sh:
        ax_s.scatter([x for x, _ in sh], [v for _, v in sh], s=6, alpha=0.5,
                     color="tab:red")
        _draw_mean_p50(ax_s, [v for _, v in sh])
    else:
        ax_s.text(0.5, 0.5, "no frontend join (missing --frontend?)",
                  transform=ax_s.transAxes, ha="center", va="center",
                  color="grey")
    ax_s.set_ylim(0, 1.05)
    ax_s.set_ylabel("queue / frontend elapsed")
    ax_s.set_xlabel("time (s)")
    ax_s.set_title("Queue share of end-to-end elapsed per request")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_retrieve_ttft(e0, retrieves: dict, frontend: dict, path: Path,
                      sched: dict | None = None) -> list[float]:
    """fig9: LMCache retrieve transfer cost vs the SAME request's frontend
    ttft_ms. NOTE the frontend TTFT includes the engine queue wait
    (ttft = first_token - request_received), so with a SCHED_DELAY join
    (`sched`) a second panel plots retrieve vs (TTFT - queue_ms) — the
    queue-removed prefill-side wall, the honest denominator for "how much
    of prefill was KV onboarding". Returns the raw-TTFT share list."""
    plt = e0._mpl()
    pts = []                     # (cost_ms, ttft_ms, net_ttft_ms|None)
    shares: list[float] = []
    for rid, rec in retrieves.items():
        fr = frontend.get(rid)
        if fr is None or fr.ttft_ms is None or fr.ttft_ms <= 0:
            continue
        net = None
        if sched and rid in sched:
            q = sched[rid].total_queue_ms
            if q is not None and 0 < q < fr.ttft_ms:
                net = fr.ttft_ms - q
        shares.append(rec["cost_ms"] / fr.ttft_ms)
        pts.append((rec["cost_ms"], fr.ttft_ms, net))
    have_net = any(p[2] is not None for p in pts)
    fig, axes = plt.subplots(1, 2 if have_net else 1, figsize=(15, 5))
    axes = axes if have_net else [axes]

    def _panel(ax, xs_ms, ys_ms, xlabel):
        ax.scatter([x / 1000.0 for x in xs_ms], [y / 1000.0 for y in ys_ms],
                   s=10, alpha=0.6, color="tab:purple")
        lim = max(max(xs_ms), max(ys_ms)) / 1000.0
        for frac, style in ((1.0, "-"), (0.5, "--"), (0.1, ":")):
            ax.plot([0, lim], [0, lim * frac], color="grey", lw=0.8,
                    ls=style, label=f"retrieve = {int(frac*100)}%")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("LMCache retrieve cost (s)")
        ax.legend(fontsize=8, loc="upper left", framealpha=0.7)

    if pts:
        _panel(axes[0], [p[1] for p in pts], [p[0] for p in pts],
               "frontend TTFT (s)  [includes queue wait]")
        axes[0].set_title("retrieve vs raw TTFT")
        if have_net:
            np_ = [(p[0], p[2]) for p in pts if p[2] is not None]
            _panel(axes[1], [p[1] for p in np_], [p[0] for p in np_],
                   "TTFT - queue_ms (s)  [prefill-side wall]")
            axes[1].set_title("retrieve vs queue-removed TTFT")
    else:
        axes[0].text(0.5, 0.5, "no retrieve<->frontend joins",
                     transform=axes[0].transAxes, ha="center", va="center",
                     color="grey")
    fig.suptitle("LMCache retrieve transfer vs TTFT per request")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return shares


def fig_eviction_vs_displacement(events: list[dict], path: Path, e0) -> None:
    """Scatter: eviction shortfall (tokens the session had to re-prefill,
    NON-compaction turns only) vs away_displaced_tokens (KV other sessions
    allocated during this turn's away window). A positive correlation is
    the causal evidence that the reuse collapse is LRU eviction driven by
    concurrent traffic — not compaction, and not visible in GPU usage."""
    plt = e0._mpl()
    pts = [(e["displaced"], e["shortfall"]) for e in events
           if e["label"] == "eviction"
           and isinstance(e["displaced"], (int, float))]
    fig, ax = plt.subplots(figsize=(7, 5))
    r = None
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=16, alpha=0.6, color="tab:red")
        r = _pearson(xs, ys)
    ax.set_xlabel("away_displaced_tokens (other sessions' KV during the gap)")
    ax.set_ylabel("eviction shortfall (re-prefilled tokens)")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    rtxt = f"  (pearson r={r:.3f}, n={len(pts)})" if r is not None else ""
    ax.set_title("Eviction shortfall vs displacement pressure" + rtxt)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- concurrent-run figures ----------


def fig_turn_llm_time(e0, ordered: list, path: Path,
                      sample_ordinals: list[int]) -> None:
    """E0's three stacked fig1 panels over SESSION-GROUPED turn order
    (see order_turns_grouped), with boundary lines at each session
    block's first ordinal — contiguous per-session blocks even when
    mif>1 interleaves sessions in time."""
    plt = e0._mpl()
    import math as _math
    n = len(ordered)
    xs = list(range(n))
    llm, tool, tool_colors, ratio, ratio_colors = [], [], [], [], []
    for t in ordered:
        lw = e0._llm_time(t)
        tw = e0._tool_time(t)
        llm.append(lw if lw is not None and lw > 0 else float("nan"))
        tool.append(tw if tw is not None and tw > 0 else float("nan"))
        is_task = any(str(name) == "task" for name in t.tool_names)
        tool_colors.append("tab:orange" if is_task else "tab:green")
        if lw and tw and lw > 0 and tw > 0:
            r = _math.log2(lw / tw)
            ratio.append(r)
            ratio_colors.append("tab:blue" if r >= 0 else "tab:orange")
        else:
            ratio.append(float("nan"))
            ratio_colors.append("none")

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

    lpos = [v for v in llm if v == v and v > 0]
    axes[0].set_yscale("log")
    axes[0].vlines(xs, min(lpos) if lpos else 1e-3, llm, color="tab:blue",
                   linewidth=0.7)
    if lpos:
        axes[0].set_ylim(min(lpos) * 0.8, max(lpos) * 1.2)
        _draw_mean_p50(axes[0], lpos, "s")
    axes[0].set_ylabel("LLM time / turn (s)")
    axes[0].set_title("Per-turn LLM Time vs turn")

    tpos = [v for v in tool if v == v and v > 0]
    axes[1].set_yscale("log")
    axes[1].vlines(xs, min(tpos) if tpos else 1e-3, tool, color=tool_colors,
                   linewidth=0.7)
    if tpos:
        axes[1].set_ylim(min(tpos) * 0.8, max(tpos) * 1.2)
        _draw_mean_p50(axes[1], tpos, "s")
    axes[1].set_ylabel("tool exec time / turn (s)")
    axes[1].set_title("Per-turn Tool Execution Time vs turn "
                      "(orange = task sub-agent)")
    axes[1].plot([], [], color="tab:orange", label="task tool")
    axes[1].plot([], [], color="tab:green", label="other tools")
    axes[1].legend(fontsize=8, loc="upper right", framealpha=0.7)

    axes[2].vlines(xs, 0.0, ratio, color=ratio_colors, linewidth=0.7)
    axes[2].axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
    rpos = [v for v in ratio if v == v]
    if rpos:
        lim = max(abs(min(rpos)), abs(max(rpos))) * 1.1 or 1.0
        axes[2].set_ylim(-lim, lim)
        _draw_mean_p50(axes[2], rpos)
    axes[2].set_ylabel("log2(LLM / tool)")
    axes[2].set_title("LLM Time / Tool Execution vs turn")
    axes[2].plot([], [], color="tab:blue", label="LLM-bound (>0)")
    axes[2].plot([], [], color="tab:orange", label="tool-bound (<0)")
    axes[2].legend(fontsize=8, loc="upper right", framealpha=0.7)

    for ax in axes:
        for b in sample_ordinals:
            ax.axvline(b, color="crimson", linewidth=0.5, alpha=0.7)
    axes[2].set_xlabel("turn")
    axes[2].set_xlim(0, max(n - 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def prefix_hit_rate_series(metrics_path: Path) -> list[tuple[float, float]]:
    """Windowed vLLM prefix-cache hit rate (0-1) from the scrape NDJSON:
    per tick, sum vllm:prefix_cache_hits_total / _queries_total across
    workers, then delta(hits)/delta(queries) between ticks. Shares the
    scrape clock with kv_usage_series, so it overlays fig3's GPU KV-usage
    panel directly (unlike the worker-log-based hit rate, a different
    clock). Empty if the run didn't scrape those counters."""
    per_ts: dict[float, list[float]] = {}
    with metrics_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("ok"):
                continue
            h = _sum_series(row, "vllm:prefix_cache_hits_total")
            q = _sum_series(row, "vllm:prefix_cache_queries_total")
            ts = row.get("ts")
            if h is None or q is None or ts is None:
                continue
            agg = per_ts.setdefault(float(ts), [0.0, 0.0])
            agg[0] += h
            agg[1] += q
    out: list[tuple[float, float]] = []
    prev = None
    for ts, (h, q) in sorted(per_ts.items()):
        if prev is not None:
            dh, dq = h - prev[1], q - prev[2]
            if dq > 0 and dh >= 0:
                out.append((ts, dh / dq))
        prev = (ts, h, q)
    return out


def queue_len_series(metrics_path: Path) -> list[tuple[float, float]]:
    """Engine queue LENGTH over time from the scrape NDJSON: per tick,
    vllm:num_requests_waiting (requests sitting in the scheduler WAITING
    queue) summed across workers. A plain gauge — no windowing needed.
    Shares the scrape clock with kv_usage_series so it stacks under
    fig3's panels."""
    per_ts: dict[float, float] = {}
    with metrics_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("ok"):
                continue
            v = _sum_series(row, "vllm:num_requests_waiting")
            ts = row.get("ts")
            if v is None or ts is None:
                continue
            per_ts[float(ts)] = per_ts.get(float(ts), 0.0) + v
    return sorted(per_ts.items())


def fig_hit_vs_kv(e0, ordered: list, kv: list, path: Path,
                  sample_ordinals: list[int],
                  sample_times: list[float],
                  queue_len: list[tuple[float, float]] | None = None,
                  prefix_hit: list[tuple[float, float]] | None = None,
                  events: list[dict] | None = None,
                  turns: list | None = None) -> None:
    """Two stacked fig3 panels for a CONCURRENT run. Top panel (cached-vs-
    reused per turn, sessions in start order as consecutive blocks): the
    KV each turn HAD cached (cached tokens) vs what it reused (hit tokens),
    with the gap on eviction turns marked red = tokens missed due to
    eviction. Middle panel: GPU KV-cache usage (left y) + the vLLM
    prefix-cache hit rate (right y, `prefix_hit`) on the scrape time
    axis. Bottom panel (`queue_len`): engine queue length
    (num_requests_waiting summed across workers), same time axis."""
    plt = e0._mpl()
    fig, (ax_top, ax_bot, ax_q) = plt.subplots(3, 1, figsize=(18, 14))

    # top panel: cached-vs-reused per turn (formerly fig5)
    starts: dict[str, float] = {}
    for t in (turns or []):
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if st is not None and t.session_id not in starts:
            starts[t.session_id] = st
    ev_by_sess: dict[str, list[dict]] = {}
    for e in (events or []):
        ev_by_sess.setdefault(e["session_id"], []).append(e)
    offset = 0
    first = True
    for sid in sorted(ev_by_sess, key=lambda s: starts.get(s, 0.0)):
        evs = sorted(ev_by_sess[sid], key=lambda e: e["step"])
        xs = [offset + j for j in range(len(evs))]
        ax_top.axvline(offset, color="crimson", linewidth=0.4, alpha=0.5,
                       zorder=1)
        ax_top.plot(xs, [e["prev_cached"] for e in evs], color="tab:orange",
                    lw=0.8, marker=".", ms=2, zorder=2,
                    label="cached tokens" if first else None)
        ax_top.plot(xs, [e["cache_read"] for e in evs], color="tab:blue",
                    lw=0.8, marker=".", ms=2, zorder=2,
                    label="hit tokens (reused)" if first else None)
        for x, e in zip(xs, evs):
            if e["label"] == "eviction":
                ax_top.plot([x, x], [e["cache_read"], e["prev_cached"]],
                            color="tab:red", lw=0.8, alpha=0.7, zorder=3)
        first = False
        offset += len(evs)
    ax_top.plot([], [], color="tab:red", lw=1.0, label="missed token")
    ax_top.set_xlim(0, max(offset - 1, 1))
    ax_top.set_ylim(bottom=0)
    ax_top.set_xlabel("turn")
    ax_top.set_ylabel("tokens")
    ax_top.set_title("Cached-vs-reused per turn")
    ax_top.legend(fontsize=8, loc="upper right", framealpha=0.7)

    ts_all = [t.llm_end_ts for t in ordered if t.llm_end_ts is not None] + \
             [t.llm_start_ts for t in ordered if t.llm_start_ts is not None]
    if ts_all:
        lo, hi = min(ts_all), max(ts_all)
        # Trim both scrape series to the profile window [lo, hi]: the scrape
        # (kv + prefix_hit) often keeps running after the last turn ends, so
        # without this the panel shows a long residual tail past the run.
        kv = [(t, v) for t, v in kv if lo <= t <= hi]
        t0 = min([lo] + [p[0] for p in kv])
        hi_x = hi - t0
    else:
        t0 = min((p[0] for p in kv), default=0.0)
        hi_x = max((t - t0 for t, _ in kv), default=1.0)
    for b in sample_times:
        ax_bot.axvline(b - t0, color="crimson", linewidth=0.5, alpha=0.7,
                       zorder=1)
    if kv:
        kx = [t - t0 for t, _ in kv]
        ax_bot.plot(kx, [v * 100.0 for _, v in kv], color="tab:green",
                    lw=0.1, zorder=2, label="GPU KV usage %")
    # x-axis ends at the profile's last turn (hi), NOT the scrape end.
    ax_bot.set_xlim(0, hi_x if hi_x > 0 else 1)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("GPU KV-cache usage (%)")
    ax_bot.set_title("GPU KV-cache Usage vs time")

    # prefix-cache hit rate on the right y-axis (same scrape clock) —
    # the former fig3-1 curve, folded into fig3. Trimmed to [t0, hi] too.
    ph = prefix_hit or []
    hi_ts = t0 + hi_x
    ph = [(t, r) for t, r in ph if t0 <= t <= hi_ts]
    if ph:
        ax_ph = ax_bot.twinx()
        px = [t - t0 for t, _ in ph]
        ax_ph.plot(px, [r * 100.0 for _, r in ph], color="tab:orange",
                   lw=0.9, zorder=3, label="prefix-cache hit rate %")
        ax_ph.set_ylim(0, 105)
        ax_ph.set_ylabel("prefix-cache hit rate (%)", color="tab:orange")
        h1, l1 = ax_bot.get_legend_handles_labels()
        h2, l2 = ax_ph.get_legend_handles_labels()
        # legend inside the axes (upper right), drawn ON TOP of the curves
        leg = ax_ph.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right",
                           framealpha=0.9)
        leg.set_zorder(5)

    # bottom panel: engine queue length, same scrape clock, trimmed to
    # the profile window like the middle panel.
    qw = [(t, v) for t, v in (queue_len or []) if t0 <= t <= t0 + hi_x]
    for b in sample_times:
        ax_q.axvline(b - t0, color="crimson", linewidth=0.5, alpha=0.7,
                     zorder=1)
    if qw:
        ax_q.plot([t - t0 for t, _ in qw], [v for _, v in qw],
                  color="tab:purple", lw=0.8, zorder=2)
    else:
        ax_q.text(0.5, 0.5, "no num_requests_waiting in scrape",
                  transform=ax_q.transAxes, ha="center", va="center",
                  color="grey")
    ax_q.set_xlim(0, hi_x if hi_x > 0 else 1)
    ax_q.set_ylim(bottom=0)
    ax_q.set_xlabel("time (s)")
    ax_q.set_ylabel("waiting requests")
    ax_q.set_title("Engine queue length vs time")

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_lmcache(e0, lmcache: dict[str, list[dict]], gpu_kv: list,
                path: Path, sample_times: list[float],
                cpu_cache_gb: float | None = None,
                disk_cache_gb: float | None = None,
                profile_end: float | None = None) -> None:
    """LMCache CPU/disk tier occupancy + transfer speed + transfer time
    vs time — the fig3 analogue for a KVBM-alternative run (LMCache
    exposes a REAL occupancy gauge, unlike KVBM). Three stacked panels on
    a shared scrape time axis:

    Panel 1 (KV Cache Tier Occupancy): host CPU tier usage
        (lmcache:local_cache_usage; % of --cpu-cache-gb when given, else
        GB) on the left y, and the LMCache hit rate
        (lmcache:lookup_hit_rate = fraction of the prompt the tier covers)
        on the right y. GPU KV usage is intentionally NOT drawn here — it
        lives in fig3. (Disk tier + eviction-rate overlays were removed;
        the counters are still parsed and available in lmcache_series.)
    Panel 2 (LMCache Transfer Speed): tokens/sec window-avg from the speed
        histograms; retrieve = host->device onboard, store = device->host
        offload.
    Panel 3 (LMCache Transfer Time): seconds it took to move the
        transferred tokens per window (tokens / speed) — a time spike
        without a matching speed drop is a slow-transfer window.

    gpu_kv is still used to anchor t0 to the earliest scrape tick (so the
    occupancy curve's real host-tier onset lag is visible), but is not
    plotted."""
    plt = e0._mpl()
    fig, (ax_use, ax_spd, ax_tt) = plt.subplots(3, 1, figsize=(18, 15),
                                                sharex=True)

    # Trim scrape series to the profile end: the scrape keeps running after
    # the last turn, so without this the panels show a long residual tail.
    if profile_end is not None:
        gpu_kv = [(t, v) for t, v in gpu_kv if t <= profile_end]
        lmcache = {w: [r for r in recs if r["ts"] <= profile_end]
                   for w, recs in lmcache.items()}
        lmcache = {w: recs for w, recs in lmcache.items() if recs}

    # Anchor t0 to the EARLIEST scrape tick overall (GPU KV appears from
    # worker startup); the lmcache:* metrics only register after LMCache's
    # first transfer, so the tier curves legitimately start LATER (real
    # host-tier onset) — that lag is the signal.
    all_ts = [r["ts"] for recs in lmcache.values() for r in recs]
    all_ts += [t for t, _ in gpu_kv]
    t0 = min(all_ts) if all_ts else 0.0
    GB = float(1 << 30)
    x_right = 0.0

    for ax in (ax_use, ax_spd, ax_tt):
        for b in sample_times:
            ax.axvline(b - t0, color="crimson", lw=0.5, alpha=0.7, zorder=1)

    # --- panel 1: KV Cache Tier Occupancy ---
    # left y = host tier usage; right y = LMCache hit rate (lookup).
    ax_hr = ax_use.twinx()
    hr_labeled = False
    for worker in sorted(lmcache):
        recs = lmcache[worker]
        pts = [(r["ts"] - t0, r["lookup_hit_rate"] * 100.0) for r in recs
               if r.get("lookup_hit_rate") is not None]
        if not pts:
            continue
        ax_use.plot([x for x, _ in pts], [y for _, y in pts], color="tab:olive",
                    lw=0.9, alpha=0.85, zorder=2,
                    label=None if hr_labeled else "LMCache hit %")
        hr_labeled = True
    ax_hr.set_ylim(0, 105)
    ax_hr.set_ylabel("LMCache hit rate (%)")

    as_pct = cpu_cache_gb and cpu_cache_gb > 0
    first = True
    for worker in sorted(lmcache):
        recs = lmcache[worker]
        ux = [r["ts"] - t0 for r in recs]
        cpu = [r["local_usage_bytes"] for r in recs]
        if as_pct:
            cap = cpu_cache_gb * GB
            ax_use.plot(ux, [b / cap * 100.0 if b is not None else float("nan")
                             for b in cpu],
                        color="tab:purple", lw=0.9, zorder=2,
                        label="host tier usage %" if first else None)
        else:
            ax_use.plot(ux, [b / GB if b is not None else float("nan")
                             for b in cpu],
                        color="tab:purple", lw=0.9, zorder=2,
                        label="host tier usage (GB)" if first else None)
        if ux:
            x_right = max(x_right, ux[-1])
        first = False

    ax_use.set_ylim(bottom=0)
    ax_use.set_ylabel("Host KV usage "
                      + ("(%)" if as_pct else "(GB)"))
    ax_use.set_title("KV Cache Tier Occupancy")
    h1, l1 = ax_use.get_legend_handles_labels()
    h2, l2 = ax_hr.get_legend_handles_labels()
    if l1 + l2:
        ax_use.legend(h1 + h2, l1 + l2, fontsize=8,
                      loc="upper left", framealpha=0.7)

    # --- panel 2: transfer speed (tokens/sec, window-avg) ---
    spd_labeled: set[str] = set()
    for worker in sorted(lmcache):
        recs = lmcache[worker]
        ret = window_avg_speed(recs, "retrieve_sum", "retrieve_count")
        sto = window_avg_speed(recs, "store_sum", "store_count")
        if ret:
            rx = [t - t0 for t, _ in ret]
            lab = "retrieve  host->device (onboard)"
            ax_spd.plot(rx, [v for _, v in ret], color="tab:blue", lw=0.9,
                        marker=".", ms=3, zorder=2,
                        label=None if lab in spd_labeled else lab)
            spd_labeled.add(lab)
            x_right = max(x_right, rx[-1])
        if sto:
            sx = [t - t0 for t, _ in sto]
            lab = "store  device->host (offload)"
            ax_spd.plot(sx, [v for _, v in sto], color="tab:red", lw=0.9,
                        marker=".", ms=3, zorder=2,
                        label=None if lab in spd_labeled else lab)
            spd_labeled.add(lab)
            x_right = max(x_right, sx[-1])
    ax_spd.set_ylim(bottom=0)
    ax_spd.set_ylabel("transfer speed (tokens/sec)")
    ax_spd.set_title("LMCache Transfer Speed vs time")
    if spd_labeled:
        ax_spd.legend(fontsize=8, loc="upper right", framealpha=0.7)

    # --- panel 3: transfer time (seconds to move those tokens) ---
    specs = [("hit_tokens", "retrieve_sum", "retrieve_count", "tab:blue",
              "retrieve  host->device"),
             ("stored_tokens", "store_sum", "store_count", "tab:red",
              "store  device->host")]
    tt_labeled: set[str] = set()
    for worker in sorted(lmcache):
        recs = lmcache[worker]
        for tok_key, sk, ck, color, label in specs:
            batches = transfer_batches(recs, tok_key, sk, ck)
            if not batches:
                continue
            lab = None if label in tt_labeled else label
            tt_labeled.add(label)
            bx = [t - t0 for t, _, _ in batches]
            ax_tt.plot(bx, [sec for _, _, sec in batches], color=color,
                       lw=0.9, marker=".", ms=3, zorder=2, label=lab)
            x_right = max(x_right, bx[-1])
    ax_tt.set_ylim(bottom=0)
    # end the x-axis at the profile end (not the last scrape point), so the
    # panels close exactly where the run ends — matches fig3.
    if profile_end is not None:
        x_right = profile_end - t0
    ax_tt.set_xlim(0, x_right if x_right > 0 else 1)
    ax_tt.set_xlabel("time (s)")
    ax_tt.set_ylabel("seconds to transfer those tokens")
    ax_tt.set_title("LMCache Transfer Time per window")
    if tt_labeled:
        ax_tt.legend(fontsize=8, loc="upper right", framealpha=0.7)
    else:
        msg = ("no transfer batches: lmcache num_hit/stored_tokens or "
               "retrieve/store_speed histograms absent or renamed "
               "(see the lmcache metric-name diagnostic on stderr)")
        ax_tt.text(0.5, 0.5, msg, transform=ax_tt.transAxes, ha="center",
                   va="center", fontsize=11, color="grey", wrap=True)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--metrics", type=Path, default=None,
                    help="vLLM scrape NDJSON for the KV-usage + prefix-hit "
                         "panel (fig3) and the LMCache panels (fig7). "
                         "Default: <logs>/vllm_metrics.ndjson when --logs "
                         "is given and the file exists")
    ap.add_argument("--frontend", type=Path, default=None,
                    help="dynamo frontend.log: joins elapsed_ms/ttft_ms by "
                         "request_id for fig8 (queue share) and fig9 "
                         "(retrieve/TTFT share). Default: <logs>/frontend.log "
                         "when --logs is given and the file exists")
    ap.add_argument("--logs", type=Path, default=None,
                    help="logs/ dir: vllm-*.log (SCHED_DELAY + LMCache "
                         "lines) and, unless overridden, frontend.log and "
                         "vllm_metrics.ndjson are auto-picked from here too")
    ap.add_argument("--compaction-drop-ratio", type=float, default=0.6,
                    help="fig5/6: a reuse shortfall is COMPACTION (not "
                         "eviction) when effective_input(N) < ratio * "
                         "effective_input(N-1). Default 0.6")
    ap.add_argument("--min-shortfall", type=int, default=128,
                    help="fig5/6: ignore reuse shortfalls <= this many "
                         "tokens (block-granularity noise). Default 128")
    ap.add_argument("--cpu-cache-gb", type=float, default=None,
                    help="LMCache CPU tier capacity GB (vllm.lmcache."
                         "cpu_cache_gb) -> fig_lmcache usage panel in %% "
                         "of capacity instead of raw GB")
    ap.add_argument("--disk-cache-gb", type=float, default=None,
                    help="LMCache disk tier capacity GB "
                         "(vllm.lmcache.disk_cache_gb)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    # --logs is the one-stop logs/ dir: auto-pick the frontend log and the
    # scrape NDJSON from it unless explicitly overridden.
    if args.logs is not None and args.logs.is_dir():
        if args.frontend is None and (args.logs / "frontend.log").exists():
            args.frontend = args.logs / "frontend.log"
        if args.metrics is None \
                and (args.logs / "vllm_metrics.ndjson").exists():
            args.metrics = args.logs / "vllm_metrics.ndjson"

    if not args.profiles.is_dir():
        print(f"error: profiles dir not found: {args.profiles}", file=sys.stderr)
        return 2
    if not args.trace.is_file():
        print(f"error: trace not found: {args.trace}", file=sys.stderr)
        return 2

    e0 = _load_e0()
    ats = e0._load_ats()
    turns = ats.load_turns(args.profiles)
    if not turns:
        print("error: no turns parsed from profiles", file=sys.stderr)
        return 2
    main_ids = e0.trace_session_ids(args.trace)
    if not main_ids:
        print("error: no session_id in trace.jsonl", file=sys.stderr)
        return 2
    turns = [t for t in turns if t.session_id in main_ids]
    if not turns:
        print("error: no turns left after trace filter", file=sys.stderr)
        return 2

    # total run wall time: earliest turn start -> latest turn end across
    # all main sessions.
    _starts = [t.llm_start_ts for t in turns if t.llm_start_ts is not None]
    _ends = [t.llm_end_ts for t in turns if t.llm_end_ts is not None]
    if _starts and _ends:
        run_lo = min(_starts)
        run_elapsed = max(_ends) - run_lo
        print(f"total elapsed time: {run_elapsed:.1f}s "
              f"({run_elapsed/60:.1f} min) over {len(turns)} turns, "
              f"{len(main_ids)} sessions")

    ordered = e0.order_turns(turns)
    # per-turn ordinal figures (fig1) use SESSION-GROUPED order so each
    # session is a contiguous block even when mif>1 interleaves them.
    grouped, grouped_bounds = order_turns_grouped(turns)
    samples_abs = session_first_times_abs(ordered)
    reuse = e0.gap_reuse_pairs(turns)
    events = eviction_events(turns, args.compaction_drop_ratio,
                             args.min_shortfall)
    have_metrics = args.metrics is not None and args.metrics.exists()
    kv = e0.kv_usage_series(args.metrics) if have_metrics else []
    prefix_hit = prefix_hit_rate_series(args.metrics) if have_metrics else []
    queue_len = queue_len_series(args.metrics) if have_metrics else []
    lmc = lmcache_series(args.metrics) if have_metrics else {}
    if lmc:
        # LMCache metric names drift across versions; when a panel is empty
        # this tells you exactly which expected name is absent so it can be
        # reconciled against the installed lmcache.
        seen = lmcache_metric_names(args.metrics)
        missing = lmcache_missing_metrics(seen)
        print(f"lmcache metrics: {len(seen)} names seen across "
              f"{len(lmc)} worker(s)", file=sys.stderr)
        if missing:
            print("  MISSING expected names (fig7 panels using them "
                  "will be empty): " + ", ".join(missing), file=sys.stderr)
            extra = sorted(n for n in seen
                           if n not in LMCACHE_EXPECTED_METRICS)
            if extra:
                print("  other lmcache: names present (candidate renames): "
                      + ", ".join(extra), file=sys.stderr)

    # LMCache retrieve transfer time as a share of total prefill compute.
    # Onboard (host->device) rides TTFT, so its cost surfaces inside the
    # engine's prefill wall; this quantifies how much of that wall was
    # spent moving KV in from the CPU tier.
    if lmc:
        ts_win = [t.llm_end_ts for t in turns if t.llm_end_ts is not None] \
            + [t.llm_start_ts for t in turns if t.llm_start_ts is not None]
        lo_w = min(ts_win) if ts_win else None
        hi_w = max(ts_win) if ts_win else None
        xfer_s = 0.0
        for recs in lmc.values():
            for bts, _tok, sec in transfer_batches(
                    recs, "hit_tokens", "retrieve_sum", "retrieve_count"):
                if (lo_w is None or bts >= lo_w) \
                        and (hi_w is None or bts <= hi_w):
                    xfer_s += sec
        prefill_s = counter_delta_total(
            args.metrics, "vllm:request_prefill_time_seconds_sum",
            lo_w, hi_w)
        if prefill_s and prefill_s > 0:
            print(f"lmcache retrieve transfer: {xfer_s:.1f}s = "
                  f"{100.0 * xfer_s / prefill_s:.1f}% of total prefill "
                  f"compute {prefill_s:.1f}s")
        else:
            print(f"lmcache retrieve transfer: {xfer_s:.1f}s "
                  f"(no vllm:request_prefill_time_seconds_sum in scrape — "
                  f"re-run scrape_metrics with the updated allowlist for "
                  f"the prefill share)")

    out_dir = args.out or (args.profiles / "e1")
    out_dir.mkdir(parents=True, exist_ok=True)

    # eviction evidence summary (this is the analysis the run is for)
    n_ev = sum(1 for e in events if e["label"] == "eviction")
    n_cp = sum(1 for e in events if e["label"] == "compaction")
    ev_disp = [(e["displaced"], e["shortfall"]) for e in events
               if e["label"] == "eviction"
               and isinstance(e["displaced"], (int, float))]
    print(f"reuse-shortfall turns: eviction {n_ev}, compaction {n_cp}, "
          f"ok {len(events) - n_ev - n_cp} (of {len(events)})")
    missed, total_cached, loss_pct = eviction_loss_pct(events)
    if loss_pct is not None:
        print(f"missed tokens due to eviction: {int(missed)} / "
              f"{int(total_cached)} cached ({loss_pct:.2f}%)")
    mm, reusable, mm_pct = prefix_mismatch_pct(events)
    if mm_pct is not None:
        print(f"prefix mismatch rate: {mm_pct:.2f}%  "
              f"(missed {int(mm)} / reusable {int(reusable)} tokens, "
              f"reasoning excluded)")
    if ev_disp:
        r = _pearson([p[0] for p in ev_disp], [p[1] for p in ev_disp])
        tot = sum(p[1] for p in ev_disp)
        print(f"eviction shortfall total {tot} tokens over {len(ev_disp)} "
              f"turns; corr(shortfall, displaced) = "
              f"{r:.3f}" if r is not None else "n/a")

    # session 3-way breakdown of the wall span:
    #   gpu_active (prefill+decode) / queue wait / others (tool+scaffold)
    queue_ms_by_rid: dict[str, float] = {}
    queued_ts_by_rid: dict[str, float] = {}
    sched: dict = {}
    frontend: dict = {}
    if args.frontend is not None and args.frontend.exists():
        arw = _load_arw()
        frontend = arw.parse_frontend(args.frontend)
        print(f"frontend log: {len(frontend)} completed requests parsed")
    if args.logs is not None and args.logs.exists():
        sched = ats.load_sched(args.logs)
        queue_ms_by_rid = {rid: rec.total_queue_ms
                           for rid, rec in sched.items()}
        queued_ts_by_rid = {rid: rec.anchor_ts
                            for rid, rec in sched.items()
                            if rec.anchor_ts is not None}
        n_joined = sum(1 for t in turns
                       if getattr(t, "request_id", None) in queue_ms_by_rid)
        joined_q_s = sum(queue_ms_by_rid[t.request_id] / 1000.0
                         for t in turns
                         if getattr(t, "request_id", None) in queue_ms_by_rid)
        print(f"queue join: {n_joined}/{len(turns)} turns matched a "
              f"SCHED_DELAY record (joined queue total {joined_q_s:.1f}s)")
    # cross-check the SCHED_DELAY join against the engine's own
    # request_queue_time histogram from the scrape (window = profile
    # turns). A large metrics total with a ~0 joined total means the
    # join is broken (request_id mismatch / logs missing SCHED_DELAY),
    # not that there was no queueing.
    if have_metrics:
        ts_win = [t.llm_end_ts for t in turns if t.llm_end_ts is not None] \
            + [t.llm_start_ts for t in turns if t.llm_start_ts is not None]
        q_metrics = counter_delta_total(
            args.metrics, "vllm:request_queue_time_seconds_sum",
            min(ts_win) if ts_win else None,
            max(ts_win) if ts_win else None)
        if q_metrics is not None:
            print(f"engine queue total (vllm request_queue_time metrics): "
                  f"{q_metrics:.1f}s")
    spans = session_spans(turns, queue_ms_by_rid or None,
                          queued_ts_by_rid or None)
    utils = session_utilizations(spans)
    bd = session_breakdown(spans)
    if bd:
        note = "" if queue_ms_by_rid else \
            "  [no --logs: queue counted inside gpu_active]"
        n_el = sum(1 for t in turns
                   if getattr(t, "elapsed_s", None) is not None)
        n_qts = sum(1 for t in turns
                    if getattr(t, "request_id", None) in queued_ts_by_rid)
        n_qseg = sum(len(sp.get("queue_segments", [])) for sp in spans)
        drawn_q = sum(e - s for sp in spans
                      for s, e in sp.get("queue_segments", []))
        print(f"session wall-span breakdown (n={len(bd)} sessions){note}")
        print(f"  [turns with dynamo elapsed_s: {n_el}/{len(turns)}; "
              f"queued_ts anchored: {n_qts}; queue segments drawn: "
              f"{n_qseg} ({drawn_q:.1f}s)]")
        print(f"  {'component':<28} {'mean':>7}")
        for key, label in (("gpu_active", "gpu_active (prefill+decode)"),
                           ("queue", "queue waiting"),
                           ("others", "others (tool+scaffold)")):
            vals = [d[key] for d in bd]
            print(f"  {label:<28} {sum(vals)/len(vals):>7.3f}")

    # fig8/fig9 companion stats: per-request joins by request_id.
    retrieves: dict = {}
    if args.logs is not None and args.logs.exists():
        lmc_stats: dict = {}
        retrieves = parse_lmcache_retrieves(args.logs, lmc_stats)
        print(f"lmcache log scan: {lmc_stats['lookup_lines']} lookup lines, "
              f"{lmc_stats['retrieve_lines']} retrieve lines, "
              f"{lmc_stats['matched']} hit-matched + "
              f"{lmc_stats['fallback_matched']} fallback-matched -> "
              f"{len(retrieves)} requests")
    if sched:
        q_all = [rec.total_queue_ms / 1000.0 for rec in sched.values()
                 if rec.total_queue_ms > 0]
        q_share = [rec.total_queue_ms / frontend[rid].total_ms
                   for rid, rec in sched.items()
                   if rid in frontend and frontend[rid].total_ms > 0]
        print(f"per-request queue (SCHED_DELAY, n={len(q_all)}; "
              f"frontend join {len(q_share)}/{len(sched)}):")
        _stat_line("queue wait (s)", q_all)
        _stat_line("queue / elapsed share", q_share)
    if retrieves:
        r_share = [rec["cost_ms"] / frontend[rid].ttft_ms
                   for rid, rec in retrieves.items()
                   if rid in frontend and frontend[rid].ttft_ms]
        # queue-removed denominator: frontend ttft INCLUDES the engine
        # queue wait, so retrieve/(ttft - queue_ms) is the share of the
        # actual prefill-side wall spent on KV onboarding.
        r_share_net = []
        for rid, rec in retrieves.items():
            fr = frontend.get(rid)
            if fr is None or not fr.ttft_ms or rid not in sched:
                continue
            q = sched[rid].total_queue_ms
            if q is not None and 0 < q < fr.ttft_ms:
                r_share_net.append(rec["cost_ms"] / (fr.ttft_ms - q))
        print(f"lmcache retrieve joins: {len(retrieves)} transfers, "
              f"frontend ttft join {len(r_share)}")
        _stat_line("retrieve cost (ms)",
                   [rec["cost_ms"] for rec in retrieves.values()],
                   "{:.1f}")
        _stat_line("retrieve / TTFT share", r_share)
        _stat_line("retrieve / (TTFT-queue) share", r_share_net)
        _stat_line("WAITING lookups per transfer",
                   [float(rec["waits"]) for rec in retrieves.values()],
                   "{:.1f}")
        # pre-transfer overhead: first lookup -> Retrieved minus the
        # transfer cost = block allocation / eviction + scheduler retries
        # while the request sat in WAITING.
        _stat_line("pre-transfer wait (ms)",
                   [rec["pre_wait_ms"] for rec in retrieves.values()
                    if rec.get("pre_wait_ms") is not None],
                   "{:.1f}")

    if not args.no_figures:
        try:
            fig_turn_llm_time(e0, grouped,
                              out_dir / "fig1_turn_llm_time.pdf",
                              grouped_bounds)
            e0.fig_llm_time_cdf(turns, out_dir / "fig2_llm_time_cdf.pdf")
            fig_hit_vs_kv(e0, ordered, kv, out_dir / "fig3_hit_vs_kv.pdf",
                          grouped_bounds, samples_abs,
                          queue_len=queue_len, prefix_hit=prefix_hit,
                          events=events, turns=turns)
            e0.fig_gap_vs_hit(reuse, out_dir / "fig4_gap_vs_hit.pdf")
            fig_eviction_vs_displacement(
                events, out_dir / "fig6_eviction_vs_displacement.pdf", e0)
            if sched:
                fig_queue_share(e0, sched, frontend,
                                out_dir / "fig8_queue_share.pdf")
            if retrieves:
                fig_retrieve_ttft(e0, retrieves, frontend,
                                  out_dir / "fig9_retrieve_ttft.pdf",
                                  sched=sched)
            if lmc:
                prof_end = max((t.llm_end_ts for t in turns
                                if t.llm_end_ts is not None), default=None)
                fig_lmcache(e0, lmc, kv, out_dir / "fig7_lmcache.pdf",
                            samples_abs, args.cpu_cache_gb, args.disk_cache_gb,
                            profile_end=prof_end)
        except ImportError:
            print("matplotlib not available (no figures written)",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
