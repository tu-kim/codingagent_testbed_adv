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
fig8 is a per-session Gantt (first turn start -> last turn end) showing how
much of each session's wall time is turn-gap delay vs LLM-active.

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
        prev_cached = pe + (prev.output_tokens or 0)
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


def session_spans(turns: list) -> list[dict]:
    """Per session: {session_id, start, end, segments} ordered by start.
    start = first turn's llm_start, end = latest turn's llm_end, segments =
    each turn's (start, end) LLM-active interval. The (end - start) span is
    the session's total wall time; the light area between segments is the
    turn-gap delay (tool + scaffold + queue) we want to visualize."""
    by: dict[str, list[tuple[float, float]]] = {}
    for t in turns:
        s, e = t.llm_start_ts, t.llm_end_ts
        if s is None or e is None:
            continue
        by.setdefault(t.session_id, []).append((s, e))
    out: list[dict] = []
    for sid, segs in by.items():
        segs = sorted(segs)
        out.append({"session_id": sid, "start": segs[0][0],
                    "end": max(e for _, e in segs), "segments": segs})
    out.sort(key=lambda d: d["start"])
    return out


def session_utilizations(spans: list[dict]) -> list[float]:
    """Per-session utilization = LLM-active time / total span, i.e. the
    fraction of the session's wall time actually spent in the LLM (the
    complement, 1 - util, is the turn-gap share). Sessions with a
    non-positive span are skipped."""
    out: list[float] = []
    for sp in spans:
        span = sp["end"] - sp["start"]
        if span <= 0:
            continue
        active = sum(e - s for s, e in sp["segments"])
        out.append(active / span)
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


def fig_session_span(e0, spans: list[dict], path: Path) -> None:
    """Gantt of session wall time: one row per session (first-started at
    top), x = time from the earliest session start. Each row shows the full
    span (first turn start -> last turn end) as a light bar and the
    LLM-active turns as dark segments on top; the light gaps between dark
    segments are the turn-gap delays (tool + scaffold + queue wait) that
    stretch the session's end-to-end time."""
    plt = e0._mpl()
    n = len(spans)
    fig, ax = plt.subplots(figsize=(18, max(4.0, 0.32 * n)))
    if not spans:
        ax.text(0.5, 0.5, "no sessions", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    t0 = spans[0]["start"]
    for i, sp in enumerate(spans):
        y = n - 1 - i                      # first-started session at the top
        ax.broken_barh([(sp["start"] - t0, max(sp["end"] - sp["start"], 1e-9))],
                       (y - 0.4, 0.8), facecolors="tab:blue", alpha=0.2,
                       zorder=1)
        segs = [(s - t0, max(e - s, 1e-3)) for s, e in sp["segments"]]
        ax.broken_barh(segs, (y - 0.4, 0.8), facecolors="tab:blue",
                       alpha=0.9, zorder=2)
    ax.set_ylim(-1, n)
    ax.set_yticks([n - 1 - i for i in range(n)])
    ax.set_yticklabels([sp["session_id"][-8:] for sp in spans],
                       fontsize=max(4, min(8, int(400 / max(n, 1)))))
    ax.set_xlim(left=0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("session (first-started at top)")
    ax.set_title("Session span: session start -> end")
    ax.plot([], [], color="tab:blue", lw=6, alpha=0.9, label="LLM active")
    ax.plot([], [], color="tab:blue", lw=6, alpha=0.2, label="turn-gap delay")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.7)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    """E0's three stacked fig1 panels, but with boundary lines at the
    given ordinals (each trace session's first turn) instead of E0's
    overlap-filtered set."""
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


def fig_hit_vs_kv(e0, ordered: list, kv: list, path: Path,
                  sample_ordinals: list[int],
                  sample_times: list[float],
                  prefix_hit: list[tuple[float, float]] | None = None,
                  events: list[dict] | None = None,
                  turns: list | None = None) -> None:
    """Two stacked fig3 panels for a CONCURRENT run. Top panel (cached-vs-
    reused per turn, sessions in start order as consecutive blocks): the
    KV each turn HAD cached (cached tokens) vs what it reused (hit tokens),
    with the gap on eviction turns marked red = tokens missed due to
    eviction. Bottom panel: GPU KV-cache usage (left y) + the vLLM
    prefix-cache hit rate (right y, `prefix_hit`) on the same scrape time
    axis."""
    plt = e0._mpl()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(18, 10))

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
    ax_top.plot([], [], color="tab:red", lw=1.0, label="miss due to eviction")
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
        kv = e0.trim_to_window(kv, lo, hi)
        t0 = min([lo] + [p[0] for p in kv])
    else:
        t0 = min((p[0] for p in kv), default=0.0)
    x_right = 0.0
    for b in sample_times:
        ax_bot.axvline(b - t0, color="crimson", linewidth=0.5, alpha=0.7,
                       zorder=1)
    if kv:
        kx = [t - t0 for t, _ in kv]
        ax_bot.plot(kx, [v * 100.0 for _, v in kv], color="tab:green",
                    lw=0.1, zorder=2, label="GPU KV usage %")
        x_right = kx[-1]
    if ts_all:
        x_right = max(x_right, hi - t0)
    ax_bot.set_xlim(0, x_right if x_right > 0 else 1)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("GPU KV-cache usage (%)")
    ax_bot.set_title("GPU KV-cache Usage vs time")

    # prefix-cache hit rate on the right y-axis (same scrape clock) —
    # the former fig3-1 curve, folded into fig3.
    ph = prefix_hit or []
    ph = [(t, r) for t, r in ph if t >= t0]
    if ph:
        ax_ph = ax_bot.twinx()
        px = [t - t0 for t, _ in ph]
        ax_ph.plot(px, [r * 100.0 for _, r in ph], color="tab:orange",
                   lw=0.9, zorder=3, label="prefix-cache hit rate %")
        ax_ph.set_ylim(0, 105)
        ax_ph.set_ylabel("prefix-cache hit rate (%)", color="tab:orange")
        h1, l1 = ax_bot.get_legend_handles_labels()
        h2, l2 = ax_ph.get_legend_handles_labels()
        ax_bot.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right",
                      framealpha=0.7)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_lmcache(e0, lmcache: dict[str, list[dict]], gpu_kv: list,
                path: Path, sample_times: list[float],
                cpu_cache_gb: float | None = None,
                disk_cache_gb: float | None = None) -> None:
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
        ax_hr.plot([x for x, _ in pts], [y for _, y in pts], color="tab:olive",
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
                         "panel (fig3) and the LMCache panels (fig7)")
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

    ordered = e0.order_turns(turns)
    sample_ords = session_first_ordinals(ordered)
    samples_abs = session_first_times_abs(ordered)
    reuse = e0.gap_reuse_pairs(turns)
    events = eviction_events(turns, args.compaction_drop_ratio,
                             args.min_shortfall)
    have_metrics = args.metrics is not None and args.metrics.exists()
    kv = e0.kv_usage_series(args.metrics) if have_metrics else []
    prefix_hit = prefix_hit_rate_series(args.metrics) if have_metrics else []
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
    if ev_disp:
        r = _pearson([p[0] for p in ev_disp], [p[1] for p in ev_disp])
        tot = sum(p[1] for p in ev_disp)
        print(f"eviction shortfall total {tot} tokens over {len(ev_disp)} "
              f"turns; corr(shortfall, displaced) = "
              f"{r:.3f}" if r is not None else "n/a")

    # session utilization = LLM-active / total span (1 - util = turn-gap share)
    spans = session_spans(turns)
    utils = session_utilizations(spans)
    if utils:
        mean = sum(utils) / len(utils)
        print(f"session utilization (LLM-active / span): mean {mean:.3f}, "
              f"p90 {_percentile(utils, 90):.3f}, "
              f"p99 {_percentile(utils, 99):.3f} (n={len(utils)})")

    if not args.no_figures:
        try:
            fig_turn_llm_time(e0, ordered, out_dir / "fig1_turn_llm_time.pdf",
                              sample_ords)
            e0.fig_llm_time_cdf(turns, out_dir / "fig2_llm_time_cdf.pdf")
            fig_hit_vs_kv(e0, ordered, kv, out_dir / "fig3_hit_vs_kv.pdf",
                          sample_ords, samples_abs, prefix_hit=prefix_hit,
                          events=events, turns=turns)
            e0.fig_gap_vs_hit(reuse, out_dir / "fig4_gap_vs_hit.pdf")
            fig_eviction_vs_displacement(
                events, out_dir / "fig6_eviction_vs_displacement.pdf", e0)
            fig_session_span(e0, spans, out_dir / "fig8_session_span.pdf")
            if lmc:
                fig_lmcache(e0, lmc, kv, out_dir / "fig7_lmcache.pdf",
                            samples_abs, args.cpu_cache_gb, args.disk_cache_gb)
        except ImportError:
            print("matplotlib not available (no figures written)",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
