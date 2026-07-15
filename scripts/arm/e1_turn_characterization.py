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
      [--worker-log logs/vllm-a0.log] [--out <dir>] [--no-figures]
"""

from __future__ import annotations

import argparse
import importlib.util
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
        except ImportError:
            print("matplotlib not available (no figures written)",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
