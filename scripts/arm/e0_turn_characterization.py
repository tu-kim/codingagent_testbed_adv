#!/usr/bin/env python3
"""E0 turn characterization: LLM-time variation + tool composition of the
fast (small-LLM) turns + KV/hit dynamics over time.

E0 is the `--sequential` characterization run (one request in flight, no
eviction interference) whose purpose is to (a) see how LLM time varies
turn-to-turn, (b) identify which (previous tool, current tool) combos
produce the SMALL-LLM turns that are CPU-offload candidates, and (c)
establish the low-load baseline for prefix-cache hit vs GPU KV size.

Four views (each -> CSV always, PDF when matplotlib is present):

1. fig1_turn_llm_time      per-turn LLM time (llm.end duration_s) as bars on
                           a TIME axis (x = s from run start, x starts at 0),
                           y clipped to --ymax (default 30s) to drop the tail;
                           red lines at SAMPLE starts (sessions with
                           >= --min-turns-boundary turns, so single-turn title
                           / task-subagent sessions don't clutter it).
2. bottom_pct_tools.csv    for the bottom 10/20/30/40% of the LLM-time
                           distribution (the fastest/smallest turns = CPU-
                           offload candidates), the distribution and % of
                           previous-tool and current-tool.
3. fig3_hit_vs_kv          shared time axis (x from 0) with prefix-cache hit
                           ratio (left y, drawn PER SESSION so the line never
                           jumps across a session's 0-hit first turn; helper
                           single-turn sessions dropped) and GPU KV-cache
                           usage (right y, from vllm_metrics.ndjson).
4. fig4_gap_vs_hit         turn gap (llm.start(N) - llm.end(N-1)) vs prefix-
                           cache hit ratio, points COLORED by the preceding
                           event ("compaction?" / prev tool) so the spread at
                           a given gap is explained (also gap_hit.csv).

Turn data is loaded via the sibling scripts/analyze_turn_scheduling.py
(TurnRec: llm_wall_s, llm_start_ts/llm_end_ts, prev_tools, tool_names,
cache_hit_ratio, away_s == turn gap).

Usage:
  scripts/arm/e0_turn_characterization.py \\
      --profiles <workspace_root>/profiles \\
      [--metrics logs/vllm_metrics.ndjson] \\
      [--cutoffs 90,80,70,60] [--out <dir>] [--no-figures]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter
from pathlib import Path

# Reuse the turn parser from the sibling analyzer.
_ATS_PATH = Path(__file__).resolve().parents[1] / "analyze_turn_scheduling.py"


def _load_ats():
    spec = importlib.util.spec_from_file_location("analyze_turn_scheduling", _ATS_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_turn_scheduling"] = mod
    spec.loader.exec_module(mod)
    return mod


def _cur_key(turn) -> str:
    return "+".join(sorted(set(turn.tool_names))) if turn.tool_names else "(none)"


# ---------- ordering + variation ----------


def order_turns(turns: list) -> list:
    """Chronological order by llm_start_ts (fallback llm_end_ts). In E0
    (sequential) this groups turns into contiguous per-session blocks."""
    def key(t):
        return (t.llm_start_ts if t.llm_start_ts is not None
                else t.llm_end_ts if t.llm_end_ts is not None else math.inf)
    return sorted(turns, key=key)


def session_boundaries(ordered: list) -> list[int]:
    """Ordinal indices where the session changes (for vertical lines)."""
    out = []
    for i in range(1, len(ordered)):
        if ordered[i].session_id != ordered[i - 1].session_id:
            out.append(i)
    return out


def sample_start_times(ordered: list, min_turns: int = 2) -> list[float]:
    """Wall-clock start time of each SAMPLE session's first turn.

    A "sample" here is a session with >= min_turns turns. Single-turn
    sessions (opencode's per-session title-generation agent, `task`
    sub-agents) are NOT samples and would otherwise draw a boundary line
    around almost every early bar. Returns times relative to the first
    turn's start (so the axis begins at 0)."""
    if not ordered:
        return []
    t0 = _t0(ordered)
    counts: dict[str, int] = {}
    first: dict[str, float] = {}
    for t in ordered:
        counts[t.session_id] = counts.get(t.session_id, 0) + 1
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if st is not None and t.session_id not in first:
            first[t.session_id] = st
    return sorted(first[s] - t0 for s, c in counts.items()
                  if c >= min_turns and s in first)


def _t0(ordered: list) -> float:
    for t in ordered:
        if t.llm_start_ts is not None:
            return t.llm_start_ts
        if t.llm_end_ts is not None:
            return t.llm_end_ts
    return 0.0


# ---------- bottom-percentile tool composition ----------


def _percentile_value(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return s[idx]


def bottom_pct_tool_dist(turns: list, cutoffs: list[float]) -> list[dict]:
    """For each cutoff q (e.g. 0.9), take turns whose llm_wall_s is in the
    bottom q of the distribution, and report the count + % of each
    previous-tool and current-tool key within that subset."""
    walls = [t.llm_wall_s for t in turns if t.llm_wall_s is not None]
    rows: list[dict] = []
    for q in cutoffs:
        thr = _percentile_value(walls, q)
        subset = [t for t in turns
                  if t.llm_wall_s is not None and t.llm_wall_s <= thr]
        n = len(subset)
        prev_c = Counter(t.prev_key for t in subset)
        cur_c = Counter(_cur_key(t) for t in subset)
        for side, counter in (("prev", prev_c), ("cur", cur_c)):
            for tool, c in counter.most_common():
                rows.append({
                    "cutoff_pct": int(round(q * 100)),
                    "threshold_llm_wall_s": thr,
                    "subset_n": n,
                    "side": side,
                    "tool": tool,
                    "count": c,
                    "pct": (100.0 * c / n) if n else math.nan,
                })
    return rows


# ---------- time series: hit ratio + KV usage ----------


def hit_series(turns: list) -> list[tuple[float, float, str]]:
    """(llm_end_ts, cache_hit_ratio, session_id) for turns with both,
    sorted by ts. Session id is kept so the plot can break the line at
    session boundaries (a 0.9 continuation turn followed by the next
    session's 0 first-turn is NOT a real fluctuation) and drop single-
    turn helper sessions."""
    out = [(t.llm_end_ts, t.cache_hit_ratio, t.session_id) for t in turns
           if t.llm_end_ts is not None and t.cache_hit_ratio is not None]
    return sorted(out)


def kv_usage_series(metrics_path: Path,
                    metric: str = "vllm:kv_cache_usage_perc",
                    ) -> list[tuple[float, float]]:
    """(ts, mean gauge value across ok worker ticks) from the scrape NDJSON."""
    out: list[tuple[float, float]] = []
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
            series = (row.get("metrics") or {}).get(metric)
            if ts is None or not series:
                continue
            vals = [e.get("value") for e in series
                    if isinstance(e.get("value"), (int, float))
                    and math.isfinite(e["value"])]
            if vals:
                out.append((float(ts), sum(vals) / len(vals)))
    return sorted(out)


# ---------- turn gap vs hit ----------


def gap_hit_pairs(turns: list) -> list[tuple[float, float]]:
    """(turn_gap_s, cache_hit_ratio); turn_gap == away_s."""
    return [(t.away_s, t.cache_hit_ratio) for t in turns
            if t.away_s is not None and t.cache_hit_ratio is not None]


def categorize_gap_hit(turns: list, compaction_drop_ratio: float = 0.6
                       ) -> list[tuple[float, float, str, str, int]]:
    """(turn_gap, hit, category, session_id, step) for turns with gap+hit.

    In sequential runs the turn gap is ~the preceding tool's exec time, so
    two turns at the SAME gap that differ in hit ratio differ because the
    prompt was rewritten differently before re-entry. category explains
    the vertical spread:
      "compaction?"  the prompt SHRANK sharply vs the previous turn
                     (effective_input(N) < ratio * effective_input(N-1)) —
                     opencode summarized the history, breaking the prefix
                     (heuristic; opencode emits no explicit compaction
                     event, so this is inferred from the token drop).
      otherwise      the previous turn's tool(s) (prev_key), e.g. "read"
                     (injects large content → lower hit) vs "bash".
    """
    eff = {(t.session_id, t.step): t.effective_input for t in turns}
    out = []
    for t in turns:
        if t.away_s is None or t.cache_hit_ratio is None:
            continue
        prev_eff = eff.get((t.session_id, t.step - 1))
        cur_eff = t.effective_input
        cat = t.prev_key
        if (prev_eff and cur_eff is not None and prev_eff > 0
                and cur_eff < compaction_drop_ratio * prev_eff):
            cat = "compaction?"
        out.append((t.away_s, t.cache_hit_ratio, cat, t.session_id, t.step))
    return out


# ---------- CSV writers ----------


def write_ordered_csv(path: Path, ordered: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ordinal", "session_id", "step", "llm_start_ts",
                    "llm_end_ts", "llm_wall_s", "prev_tools", "cur_tools",
                    "turn_gap_s", "cache_hit_ratio"])
        for i, t in enumerate(ordered):
            w.writerow([i, t.session_id, t.step,
                        t.llm_start_ts if t.llm_start_ts is not None else "",
                        t.llm_end_ts if t.llm_end_ts is not None else "",
                        t.llm_wall_s if t.llm_wall_s is not None else "",
                        t.prev_key, _cur_key(t),
                        t.away_s if t.away_s is not None else "",
                        t.cache_hit_ratio if t.cache_hit_ratio is not None else ""])


def write_bottom_pct_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "cutoff_pct", "threshold_llm_wall_s", "subset_n",
            "side", "tool", "count", "pct"])
        w.writeheader()
        w.writerows(rows)


# ---------- figures (matplotlib, lazy + guarded) ----------


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_turn_llm_time(ordered: list, path: Path, *, ymax: float = 30.0,
                      min_turns_boundary: int = 2) -> None:
    """Each turn as a bar spanning its LLM-busy interval on a TIME axis
    (x = seconds from run start, starting at 0); height = LLM time. Red
    lines mark SAMPLE (>= min_turns_boundary-turn session) starts. y is
    clipped to `ymax` s to drop the long tail."""
    plt = _mpl()
    t0 = _t0(ordered)
    xs, widths, heights = [], [], []
    x_right = 0.0
    for t in ordered:
        wall = t.llm_wall_s if t.llm_wall_s is not None else 0.0
        start = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if start is None:
            continue
        x = start - t0
        xs.append(x)
        widths.append(max(wall, 1e-9))     # span the busy interval
        heights.append(wall)
        x_right = max(x_right, x + wall)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(xs, heights, width=widths, align="edge", linewidth=0)
    for b in sample_start_times(ordered, min_turns_boundary):
        ax.axvline(b, color="crimson", linewidth=0.6, alpha=0.7)
    ax.set_xlim(0, x_right if x_right > 0 else 1)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("LLM time / turn (s)")
    ax.set_title(f"Per-turn LLM time (E0; y clipped at {ymax:g}s, "
                 "red = sample start)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_hit_vs_kv(hits: list[tuple[float, float, str]],
                  kv: list[tuple[float, float]], path: Path,
                  *, min_turns: int = 2) -> None:
    """Prefix-cache hit ratio (left y) vs GPU KV usage (right y) on a
    SHARED time axis. Both series are shifted by the same origin (the
    earliest data point of either), and x starts at 0, so whichever
    stream began first sits at the left edge instead of the hit line
    being pinned to 0 while KV floats. The hit line is drawn PER SESSION
    (no segment crosses a session boundary) and sessions with fewer than
    `min_turns` turns (title / task-subagent helpers, all hit=0) are
    dropped — that is what removes the fake 0<->0.9 sawtooth."""
    plt = _mpl()
    # keep only sessions with >= min_turns hit points
    counts: dict[str, int] = {}
    for _ts, _h, sid in hits:
        counts[sid] = counts.get(sid, 0) + 1
    kept = [(ts, h, sid) for ts, h, sid in hits if counts[sid] >= min_turns]

    origins = [p[0] for p in kept] + [p[0] for p in kv]
    t0 = min(origins) if origins else 0.0
    x_right = 0.0

    fig, ax = plt.subplots(figsize=(9, 3.5))
    # group kept hits by session, draw each as its own line segment
    by_sess: dict[str, list[tuple[float, float]]] = {}
    for ts, h, sid in kept:
        by_sess.setdefault(sid, []).append((ts, h))
    first_label = True
    for sid, pts in by_sess.items():
        pts.sort()
        xs = [ts - t0 for ts, _ in pts]
        ys = [h for _, h in pts]
        ax.plot(xs, ys, color="tab:blue", marker=".", ms=3, lw=0.8,
                label="prefix hit" if first_label else None)
        first_label = False
        if xs:
            x_right = max(x_right, xs[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("prefix-cache hit ratio", color="tab:blue")
    ax.set_ylim(0, 1)
    if kv:
        ax2 = ax.twinx()
        kx = [t - t0 for t, _ in kv]
        ax2.plot(kx, [v for _, v in kv], color="tab:orange", lw=1.0,
                 label="KV usage")
        ax2.set_ylabel("GPU KV-cache usage", color="tab:orange")
        if kx:
            x_right = max(x_right, kx[-1])
    ax.set_xlim(0, x_right if x_right > 0 else 1)
    ax.set_title("Prefix-cache hit ratio vs GPU KV size (E0)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_gap_vs_hit(cats: list[tuple[float, float, str, str, int]],
                   path: Path) -> None:
    """Scatter of turn gap vs hit, COLORED by the preceding-event category
    (compaction? / prev tool) so the vertical spread at a given gap is
    explained."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    groups: dict[str, list[tuple[float, float]]] = {}
    for g, h, c, _sid, _step in cats:
        groups.setdefault(c, []).append((g, h))
    for c in sorted(groups):
        pts = groups[c]
        gs = [p[0] for p in pts]
        hs = [p[1] for p in pts]
        if c.startswith("compaction"):
            ax.scatter(gs, hs, s=36, marker="x", color="crimson",
                       label=f"{c} ({len(pts)})", zorder=3)
        else:
            ax.scatter(gs, hs, s=14, alpha=0.6, label=f"{c} ({len(pts)})")
    ax.set_xlabel("turn gap (s)")
    ax.set_ylabel("prefix-cache hit ratio")
    ax.set_ylim(0, 1)
    if groups:
        ax.legend(fontsize=7, loc="best", framealpha=0.7)
    ax.set_title("Turn gap vs prefix-cache hit, by preceding event (E0)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------


def _print_bottom(rows: list[dict]) -> None:
    by_cut: dict[int, list[dict]] = {}
    for r in rows:
        by_cut.setdefault(r["cutoff_pct"], []).append(r)
    for cut in sorted(by_cut):
        sub = by_cut[cut]
        thr = sub[0]["threshold_llm_wall_s"]
        n = sub[0]["subset_n"]
        print(f"\nbottom {cut}% of LLM time (<= {thr:.3f}s, n={n}):")
        for side in ("prev", "cur"):
            items = [r for r in sub if r["side"] == side]
            top = ", ".join(f"{r['tool']} {r['pct']:.0f}%" for r in items[:5])
            print(f"  {side}_tool: {top}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--metrics", type=Path, default=None,
                    help="vLLM scrape NDJSON for the KV-usage right axis")
    ap.add_argument("--cutoffs", default="10,20,30,40",
                    help="bottom-percentile cutoffs (comma list of %); the "
                         "fastest/smallest-LLM-time turns = CPU-offload "
                         "candidates. Default bottom 10,20,30,40%")
    ap.add_argument("--compaction-drop-ratio", type=float, default=0.6,
                    help="fig4: a turn whose effective_input < ratio * the "
                         "previous turn's is flagged 'compaction?'. Default 0.6")
    ap.add_argument("--ymax", type=float, default=30.0,
                    help="fig1 y-axis (LLM time) clip in seconds; drops the "
                         "long tail. Default 30.")
    ap.add_argument("--min-turns-boundary", type=int, default=2,
                    help="fig1: only sessions with >= this many turns get a "
                         "sample-boundary line (filters single-turn title / "
                         "task-subagent sessions). Default 2.")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    if not args.profiles.is_dir():
        print(f"error: profiles dir not found: {args.profiles}", file=sys.stderr)
        return 2
    cutoffs = [float(c) / 100.0 for c in args.cutoffs.split(",") if c.strip()]

    ats = _load_ats()
    turns = ats.load_turns(args.profiles)
    if not turns:
        print("error: no turns parsed from profiles", file=sys.stderr)
        return 2

    ordered = order_turns(turns)
    samples = sample_start_times(ordered, args.min_turns_boundary)
    bottom_rows = bottom_pct_tool_dist(turns, cutoffs)
    hits = hit_series(turns)
    cats = categorize_gap_hit(turns, args.compaction_drop_ratio)
    kv = kv_usage_series(args.metrics) if args.metrics and args.metrics.exists() else []

    out_dir = args.out or (args.profiles / "e0")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ordered_csv(out_dir / "turns_ordered.csv", ordered)
    write_bottom_pct_csv(out_dir / "bottom_pct_tools.csv", bottom_rows)
    with (out_dir / "gap_hit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session_id", "step", "turn_gap_s", "cache_hit_ratio", "category"])
        for g, h, c, sid, step in cats:
            w.writerow([sid, step, g, h, c])

    n_sessions = len({t.session_id for t in turns})
    print(f"turns: {len(turns)} across {n_sessions} sessions "
          f"({len(samples)} samples with >= {args.min_turns_boundary} turns)")
    _print_bottom(bottom_rows)
    if args.metrics and not kv:
        print("warning: no KV-usage series parsed from --metrics", file=sys.stderr)

    if not args.no_figures:
        try:
            fig_turn_llm_time(ordered, out_dir / "fig1_turn_llm_time.pdf",
                              ymax=args.ymax,
                              min_turns_boundary=args.min_turns_boundary)
            fig_hit_vs_kv(hits, kv, out_dir / "fig3_hit_vs_kv.pdf",
                          min_turns=args.min_turns_boundary)
            fig_gap_vs_hit(cats, out_dir / "fig4_gap_vs_hit.pdf")
            print(f"\nwrote figures under {out_dir}")
        except ImportError:
            print("matplotlib not available; wrote CSVs only "
                  "(--no-figures to silence)", file=sys.stderr)

    print(f"wrote {out_dir / 'turns_ordered.csv'}")
    print(f"wrote {out_dir / 'bottom_pct_tools.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
