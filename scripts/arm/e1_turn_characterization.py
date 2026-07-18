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

Everything else (fig1 panels, fig2 CDF, fig3 bottom KV-usage, fig3-1
worker-log view, fig4 reuse-vs-gap) reuses the E0 implementations.

Usage:
  scripts/arm/e1_turn_characterization.py \\
      --profiles <workspace_root>/profiles \\
      --trace results/<run>/trace.jsonl \\
      [--metrics logs/vllm_metrics.ndjson] \\
      [--cpu-cache-gb N] [--disk-cache-gb N] \\
      [--worker-log logs/vllm-a0.log] [--out <dir>] [--no-figures]

When the scrape NDJSON carries lmcache:* metrics (an LMCache offload run):
  fig7_lmcache — CPU/disk tier occupancy (lmcache:local_cache_usage /
    local_storage_usage bytes; % of --cpu-cache-gb/--disk-cache-gb when
    given) on the GPU-KV-usage time axis + CPU-tier eviction rate, plus a
    host<->device transfer-speed panel (retrieve = onboard, store =
    offload; window-avg tokens/sec from the speed histograms).
  fig8_lmcache_transfer_time — tokens transferred per window (num_hit_tokens
    / num_stored_tokens) and the seconds it took to move them (tokens /
    speed): a time spike without a matching token spike = slow transfer.
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
                "evict_keys": _sum_series(row, "lmcache:local_cpu_evict_keys_count"),
                "evict_failed": _sum_series(row, "lmcache:local_cpu_evict_failed_count"),
                # transferred-token counters (pairs with the speed
                # histograms to derive transfer TIME): hit = tokens
                # retrieved host->device, stored = tokens offloaded
                # device->host.
                "hit_tokens": _sum_series(row, "lmcache:num_hit_tokens"),
                "stored_tokens": _sum_series(row, "lmcache:num_stored_tokens"),
            })
    for recs in out.values():
        recs.sort(key=lambda r: r["ts"])
    return out


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


def fig_reuse_shortfall(events: list[dict], turns: list, path: Path,
                        e0) -> None:
    """Per session block (sessions in start order): prev_cached (what was
    cached) vs cache_read (what was reused). The gap between them = KV the
    session failed to reuse; eviction turns (prompt did not shrink) are
    marked red, compaction turns (prompt shrank) grey. Shows the reuse
    collapse happens WHILE the cached total keeps growing = eviction, not
    compaction."""
    plt = e0._mpl()
    fig, ax = plt.subplots(figsize=(18, 6))
    # session start order
    starts: dict[str, float] = {}
    for t in turns:
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if st is not None and t.session_id not in starts:
            starts[t.session_id] = st
    ev_by_sess: dict[str, list[dict]] = {}
    for e in events:
        ev_by_sess.setdefault(e["session_id"], []).append(e)
    offset = 0
    first = True
    for sid in sorted(ev_by_sess, key=lambda s: starts.get(s, 0.0)):
        evs = sorted(ev_by_sess[sid], key=lambda e: e["step"])
        xs = [offset + j for j in range(len(evs))]
        ax.axvline(offset, color="crimson", linewidth=0.4, alpha=0.5, zorder=1)
        ax.plot(xs, [e["prev_cached"] for e in evs], color="tab:orange",
                lw=0.8, marker=".", ms=2, zorder=2,
                label="prev_cached (was cached)" if first else None)
        ax.plot(xs, [e["cache_read"] for e in evs], color="tab:blue",
                lw=0.8, marker=".", ms=2, zorder=2,
                label="cache_read (reused)" if first else None)
        # mark reuse collapses
        for x, e in zip(xs, evs):
            if e["label"] == "eviction":
                ax.plot([x, x], [e["cache_read"], e["prev_cached"]],
                        color="red", lw=0.8, alpha=0.6, zorder=3)
            elif e["label"] == "compaction":
                ax.plot([x, x], [e["cache_read"], e["prev_cached"]],
                        color="grey", lw=0.8, alpha=0.5, zorder=3)
        first = False
        offset += len(evs)
    ax.plot([], [], color="red", lw=1.0, label="eviction gap (prompt grew)")
    ax.plot([], [], color="grey", lw=1.0, label="compaction gap (prompt shrank)")
    ax.set_xlim(0, max(offset - 1, 1))
    ax.set_ylim(bottom=0)
    ax.set_xlabel("turn (sessions in start order, one block each)")
    ax.set_ylabel("tokens")
    ax.set_title("Cached-vs-reused per turn: gap = KV not reused "
                 "(red = eviction, grey = compaction)")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
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
    axes[0].set_ylabel("LLM time / turn (s)")
    axes[0].set_title("Per-turn LLM Time vs turn")

    tpos = [v for v in tool if v == v and v > 0]
    axes[1].set_yscale("log")
    axes[1].vlines(xs, min(tpos) if tpos else 1e-3, tool, color=tool_colors,
                   linewidth=0.7)
    if tpos:
        axes[1].set_ylim(min(tpos) * 0.8, max(tpos) * 1.2)
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
    axes[2].set_ylabel("log2(LLM / tool)  +up LLM-bound / -down tool-bound")
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


def fig_hit_vs_kv(e0, ordered: list, kv: list, path: Path,
                  sample_ordinals: list[int],
                  sample_times: list[float]) -> None:
    """E0's two stacked fig3 panels for a CONCURRENT run. Top panel:
    sessions are laid out as CONSECUTIVE x-axis blocks, ordered by session
    start time — each block holds exactly ONE session's turns (in its own
    chronological order), so lines never overlap or zigzag across each
    other. Red lines mark the block boundaries. The bottom time panel is
    unchanged (real time, sample-start lines)."""
    plt = e0._mpl()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(18, 10))

    by_sess: dict[str, list[tuple[float, float]]] = {}
    for t in ordered:
        h = e0._hit_tokens(t)
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if h is not None and st is not None:
            by_sess.setdefault(t.session_id, []).append((st, h))
    # blocks in session start order
    order = sorted(by_sess, key=lambda sid: min(ts for ts, _ in by_sess[sid]))
    offset = 0
    for sid in order:
        pts = sorted(by_sess[sid])      # session-local chronological order
        xs = [offset + j for j in range(len(pts))]
        ax_top.axvline(offset, color="crimson", linewidth=0.5, alpha=0.7,
                       zorder=1)
        ax_top.plot(xs, [h for _, h in pts],
                    color="tab:blue", marker=".", ms=3, lw=0.8, zorder=2)
        offset += len(pts)
    ax_top.set_xlim(0, max(offset - 1, 1))
    ax_top.set_ylim(bottom=0)
    ax_top.set_xlabel("turn (sessions in start order, one block each)")
    ax_top.set_ylabel("prefix-cache hit tokens / turn")
    ax_top.set_title("Prefix-cache Hit Tokens vs turn")

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
                    lw=0.1, zorder=2)
        x_right = kx[-1]
    if ts_all:
        x_right = max(x_right, hi - t0)
    ax_bot.set_xlim(0, x_right if x_right > 0 else 1)
    ax_bot.set_ylim(bottom=0)
    ax_bot.set_xlabel("time (s)")
    ax_bot.set_ylabel("GPU KV-cache usage (%)")
    ax_bot.set_title("GPU KV-cache Usage vs time")

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_lmcache(e0, lmcache: dict[str, list[dict]], gpu_kv: list,
                path: Path, sample_times: list[float],
                cpu_cache_gb: float | None = None,
                disk_cache_gb: float | None = None) -> None:
    """LMCache CPU/disk tier occupancy + transfer speed vs time — the
    fig3 analogue for a KVBM-alternative run (LMCache exposes a REAL
    occupancy gauge, unlike KVBM).

    Panel 1: CPU tier usage (lmcache:local_cache_usage bytes; as % of
        --cpu-cache-gb when given, else GB) + disk tier usage, on the SAME
        time axis as the GPU KV-cache usage curve (twin %-axis) so the
        host tier filling as the GPU stays pinned is directly visible;
        plus CPU-tier eviction rate (lmcache:local_cpu_evict_keys_count
        delta, chunks/sec, third axis) so the host-tier-full -> LRU-drop
        onset lines up with the occupancy curve hitting capacity.
    Panel 2: transfer speed (tokens/sec, window-avg from the speed
        histograms): retrieve = host->device onboard, store = device->host
        offload. Same time axis."""
    plt = e0._mpl()
    fig, (ax_use, ax_spd) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    # Anchor t0 to the EARLIEST scrape tick overall (GPU KV appears from
    # worker startup) so the GPU KV-usage curve shows its full 0->full
    # warmup ramp from x=0. The lmcache:* metrics only register after
    # LMCache's first transfer, so the tier-occupancy + speed curves
    # legitimately start LATER (at the real host-tier onset) — that lag is
    # the signal: the host tier engages once the GPU is under pressure.
    # (Both curves cannot start at x=0 on a shared real-time axis without
    # hiding the GPU ramp, so we keep true-zero and let occupancy lag.)
    all_ts = [r["ts"] for recs in lmcache.values() for r in recs]
    all_ts += [t for t, _ in gpu_kv]
    t0 = min(all_ts) if all_ts else 0.0
    GB = float(1 << 30)
    x_right = 0.0

    for b in sample_times:
        ax_use.axvline(b - t0, color="crimson", lw=0.5, alpha=0.7, zorder=1)
        ax_spd.axvline(b - t0, color="crimson", lw=0.5, alpha=0.7, zorder=1)

    # --- panel 1: tier occupancy (+ GPU KV usage on twin axis) ---
    ax_gpu = ax_use.twinx()
    # third axis: CPU-tier eviction rate (chunks/sec), the "host tier full,
    # LRU dropping" signal — offset spine so it doesn't overlap ax_gpu.
    ax_ev = ax_use.twinx()
    ax_ev.spines["right"].set_position(("axes", 1.06))
    if gpu_kv:
        gx = [t - t0 for t, _ in gpu_kv]
        ax_gpu.plot(gx, [v * 100.0 for _, v in gpu_kv], color="tab:green",
                    lw=0.1, zorder=1, label="GPU KV usage %")
        x_right = max(x_right, gx[-1])
    ax_gpu.set_ylim(0, 105)
    ax_gpu.set_ylabel("GPU KV-cache usage (%)")

    as_pct = cpu_cache_gb and cpu_cache_gb > 0
    first = True
    for worker in sorted(lmcache):
        recs = lmcache[worker]
        ux = [r["ts"] - t0 for r in recs]
        cpu = [r["local_usage_bytes"] for r in recs]
        disk = [r["storage_usage_bytes"] for r in recs]
        if as_pct:
            cap = cpu_cache_gb * GB
            ax_use.plot(ux, [b / cap * 100.0 if b is not None else float("nan")
                             for b in cpu],
                        color="tab:purple", lw=0.9, zorder=2,
                        label="CPU tier usage %" if first else None)
        else:
            ax_use.plot(ux, [b / GB if b is not None else float("nan")
                             for b in cpu],
                        color="tab:purple", lw=0.9, zorder=2,
                        label="CPU tier usage (GB)" if first else None)
        if any(d is not None for d in disk):
            if disk_cache_gb and disk_cache_gb > 0 and as_pct:
                dcap = disk_cache_gb * GB
                ax_use.plot(ux, [b / dcap * 100.0 if b is not None
                                 else float("nan") for b in disk],
                            color="tab:blue", lw=0.9, ls="--", zorder=2,
                            label="disk tier usage %" if first else None)
            else:
                ax_use.plot(ux, [b / GB if b is not None else float("nan")
                                 for b in disk],
                            color="tab:blue", lw=0.9, ls="--", zorder=2,
                            label="disk tier usage (GB)" if first else None)
        # CPU-tier eviction rate (chunks/sec): fires once the host tier
        # fills and LRU starts dropping chunks. evict_failed = no evictable
        # candidate found (pure pressure, chunks pinned in-flight).
        ev = counter_rate(recs, "evict_keys")
        evf = counter_rate(recs, "evict_failed")
        if ev:
            ex = [t - t0 for t, _ in ev]
            ax_ev.plot(ex, [v for _, v in ev], color="tab:red", lw=0.9,
                       zorder=3,
                       label="CPU evict rate (chunks/s)" if first else None)
            x_right = max(x_right, ex[-1])
        if any(v > 0 for _, v in evf):
            efx = [t - t0 for t, _ in evf]
            ax_ev.plot(efx, [v for _, v in evf], color="tab:red", lw=0.7,
                       ls=":", alpha=0.8, zorder=3,
                       label="CPU evict-FAILED rate (1/s)" if first else None)
        if ux:
            x_right = max(x_right, ux[-1])
        first = False

    ax_use.set_ylim(bottom=0)
    ax_ev.set_ylim(bottom=0)
    ax_use.set_ylabel("LMCache tier usage "
                      + ("(% of capacity)" if as_pct else "(GB)"))
    ax_ev.set_ylabel("CPU-tier eviction rate (chunks/s)", color="tab:red")
    ax_use.set_title("LMCache Tier Occupancy + Eviction vs time "
                     "(host tier fills -> LRU eviction kicks in)")
    h1, l1 = ax_use.get_legend_handles_labels()
    h2, l2 = ax_gpu.get_legend_handles_labels()
    h3, l3 = ax_ev.get_legend_handles_labels()
    ax_use.legend(h1 + h2 + h3, l1 + l2 + l3, fontsize=8, loc="upper left",
                  framealpha=0.7)

    # --- panel 2: transfer speed (tokens/sec, window-avg) ---
    first = True
    for worker in sorted(lmcache):
        recs = lmcache[worker]
        ret = window_avg_speed(recs, "retrieve_sum", "retrieve_count")
        sto = window_avg_speed(recs, "store_sum", "store_count")
        if ret:
            rx = [t - t0 for t, _ in ret]
            ax_spd.plot(rx, [v for _, v in ret], color="tab:blue", lw=0.9,
                        marker=".", ms=3, zorder=2,
                        label="retrieve  host->device (onboard)"
                        if first else None)
            x_right = max(x_right, rx[-1])
        if sto:
            sx = [t - t0 for t, _ in sto]
            ax_spd.plot(sx, [v for _, v in sto], color="tab:red", lw=0.9,
                        marker=".", ms=3, zorder=2,
                        label="store  device->host (offload)"
                        if first else None)
            x_right = max(x_right, sx[-1])
        first = False
    ax_spd.set_ylim(bottom=0)
    ax_spd.set_xlim(0, x_right if x_right > 0 else 1)
    ax_spd.set_xlabel("time (s)")
    ax_spd.set_ylabel("transfer speed (tokens/sec)")
    ax_spd.set_title("LMCache Transfer Speed vs time "
                     "(window-avg from speed histograms)")
    ax_spd.legend(fontsize=8, loc="upper right", framealpha=0.7)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_lmcache_transfer_time(e0, lmcache: dict[str, list[dict]],
                              path: Path, sample_times: list[float]) -> None:
    """How long the transferred tokens took to move, per window.

    Pairs the transferred-token counters (lmcache:num_hit_tokens for
    retrieve = host->device, num_stored_tokens for store = device->host)
    with the matching speed histogram: seconds = tokens / (delta(_sum)/
    delta(_count)). Two stacked panels on a shared time axis:
      top    — tokens transferred per window (the actual token volume)
      bottom — seconds it took to transfer THOSE tokens (tokens / speed)
    so a spike in bottom-panel time that is NOT explained by a top-panel
    token spike is a slow-transfer (contention) window."""
    plt = e0._mpl()
    fig, (ax_tok, ax_t) = plt.subplots(2, 1, figsize=(18, 10), sharex=True)

    all_ts = [r["ts"] for recs in lmcache.values() for r in recs]
    t0 = min(all_ts) if all_ts else 0.0
    x_right = 0.0

    for b in sample_times:
        ax_tok.axvline(b - t0, color="crimson", lw=0.5, alpha=0.7, zorder=1)
        ax_t.axvline(b - t0, color="crimson", lw=0.5, alpha=0.7, zorder=1)

    specs = [("hit_tokens", "retrieve_sum", "retrieve_count", "tab:blue",
              "retrieve  host->device"),
             ("stored_tokens", "store_sum", "store_count", "tab:red",
              "store  device->host")]
    first = True
    for worker in sorted(lmcache):
        recs = lmcache[worker]
        for tok_key, sk, ck, color, label in specs:
            batches = transfer_batches(recs, tok_key, sk, ck)
            if not batches:
                continue
            bx = [t - t0 for t, _, _ in batches]
            ax_tok.plot(bx, [tok for _, tok, _ in batches], color=color,
                        lw=0.9, marker=".", ms=3, zorder=2,
                        label=label if first else None)
            ax_t.plot(bx, [sec for _, _, sec in batches], color=color,
                      lw=0.9, marker=".", ms=3, zorder=2,
                      label=label if first else None)
            x_right = max(x_right, bx[-1])
        first = False

    ax_tok.set_ylim(bottom=0)
    ax_tok.set_ylabel("tokens transferred / window")
    ax_tok.set_title("LMCache Tokens Transferred per window")
    ax_tok.legend(fontsize=8, loc="upper right", framealpha=0.7)

    ax_t.set_ylim(bottom=0)
    ax_t.set_xlim(0, x_right if x_right > 0 else 1)
    ax_t.set_xlabel("time (s)")
    ax_t.set_ylabel("seconds to transfer those tokens")
    ax_t.set_title("LMCache Transfer Time per window "
                   "(tokens / speed; spike w/o token spike = slow transfer)")
    ax_t.legend(fontsize=8, loc="upper right", framealpha=0.7)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--metrics", type=Path, default=None,
                    help="vLLM scrape NDJSON for the KV-usage panel")
    ap.add_argument("--worker-log", type=Path, default=None,
                    help="vLLM worker log for fig3-1")
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
    lmc = lmcache_series(args.metrics) if have_metrics else {}

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

    if not args.no_figures:
        try:
            fig_turn_llm_time(e0, ordered, out_dir / "fig1_turn_llm_time.pdf",
                              sample_ords)
            e0.fig_llm_time_cdf(turns, out_dir / "fig2_llm_time_cdf.pdf")
            fig_hit_vs_kv(e0, ordered, kv, out_dir / "fig3_hit_vs_kv.pdf",
                          sample_ords, samples_abs)
            e0.fig_gap_vs_hit(reuse, out_dir / "fig4_gap_vs_hit.pdf")
            fig_reuse_shortfall(events, turns,
                                out_dir / "fig5_reuse_shortfall.pdf", e0)
            fig_eviction_vs_displacement(
                events, out_dir / "fig6_eviction_vs_displacement.pdf", e0)
            if args.worker_log and args.worker_log.exists():
                wl_hits, wl_kv = e0.worker_log_series(args.worker_log)
                if wl_hits or wl_kv:
                    e0.fig_worker_hit_kv(wl_hits, wl_kv,
                                         out_dir / "fig3-1_worker_hit_kv.pdf")
            if lmc:
                fig_lmcache(e0, lmc, kv, out_dir / "fig7_lmcache.pdf",
                            samples_abs, args.cpu_cache_gb, args.disk_cache_gb)
                fig_lmcache_transfer_time(
                    e0, lmc, out_dir / "fig8_lmcache_transfer_time.pdf",
                    samples_abs)
        except ImportError:
            print("matplotlib not available (no figures written)",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
