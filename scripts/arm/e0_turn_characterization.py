#!/usr/bin/env python3
"""E0 turn characterization: LLM-time variation + tool composition of the
fast (small-LLM) turns + KV/hit dynamics over time.

E0 is the `--sequential` characterization run (one request in flight, no
eviction interference) whose purpose is to (a) see how LLM time varies
turn-to-turn, (b) identify which (previous tool, current tool) combos
produce the SMALL-LLM turns that are CPU-offload candidates, and (c)
establish the low-load baseline for prefix-cache hit vs GPU KV size.

Four views (each -> CSV always, PDF when matplotlib is present):

1. fig1_turn_llm_time      per-turn LLM time (llm.end duration_s) as a bar
                           chart along a turn ordinal axis, with vertical
                           lines at SAMPLE (session) boundaries -> turn-to-
                           turn LLM-time variation.
2. bottom_pct_tools.csv    for the bottom 90/80/70/60% of the LLM-time
                           distribution (the fast turns), the distribution
                           and % of previous-tool and current-tool.
3. fig3_hit_vs_kv          time axis (wall clock) with prefix-cache hit
                           ratio (left y) and GPU KV-cache usage (right y,
                           from vllm_metrics.ndjson).
4. fig4_gap_vs_hit         turn gap (llm.start(N) - llm.end(N-1)) vs prefix-
                           cache hit ratio.

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


def hit_series(turns: list) -> list[tuple[float, float]]:
    """(llm_end_ts, cache_hit_ratio) for turns that have both."""
    out = [(t.llm_end_ts, t.cache_hit_ratio) for t in turns
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


def fig_turn_llm_time(ordered: list, boundaries: list[int], path: Path) -> None:
    plt = _mpl()
    walls = [t.llm_wall_s if t.llm_wall_s is not None else 0.0 for t in ordered]
    fig, ax = plt.subplots(figsize=(max(6, len(ordered) * 0.06), 3.5))
    ax.bar(range(len(walls)), walls, width=1.0, linewidth=0)
    for b in boundaries:
        ax.axvline(b - 0.5, color="crimson", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("turn ordinal (time order; red = sample boundary)")
    ax.set_ylabel("LLM time / turn (s)")
    ax.set_title("Per-turn LLM time (E0)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_hit_vs_kv(hits: list[tuple[float, float]],
                  kv: list[tuple[float, float]], path: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if hits:
        t0 = hits[0][0]
        ax.plot([t - t0 for t, _ in hits], [h for _, h in hits],
                color="tab:blue", marker=".", ms=3, lw=0.8, label="prefix hit")
    ax.set_xlabel("wall clock (s from first turn)")
    ax.set_ylabel("prefix-cache hit ratio", color="tab:blue")
    ax.set_ylim(0, 1)
    if kv:
        t0 = hits[0][0] if hits else kv[0][0]
        ax2 = ax.twinx()
        ax2.plot([t - t0 for t, _ in kv], [v for _, v in kv],
                 color="tab:orange", lw=1.0, label="KV usage")
        ax2.set_ylabel("GPU KV-cache usage", color="tab:orange")
    ax.set_title("Prefix-cache hit ratio vs GPU KV size (E0)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_gap_vs_hit(pairs: list[tuple[float, float]], path: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(5, 4))
    if pairs:
        ax.scatter([g for g, _ in pairs], [h for _, h in pairs], s=10, alpha=0.6)
    ax.set_xlabel("turn gap (s) = llm.start(N) - llm.end(N-1)")
    ax.set_ylabel("prefix-cache hit ratio")
    ax.set_ylim(0, 1)
    ax.set_title("Turn gap vs prefix-cache hit (E0)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------


def _print_bottom(rows: list[dict]) -> None:
    by_cut: dict[int, list[dict]] = {}
    for r in rows:
        by_cut.setdefault(r["cutoff_pct"], []).append(r)
    for cut in sorted(by_cut, reverse=True):
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
    ap.add_argument("--cutoffs", default="90,80,70,60",
                    help="bottom-percentile cutoffs (comma list of %)")
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
    boundaries = session_boundaries(ordered)
    bottom_rows = bottom_pct_tool_dist(turns, cutoffs)
    hits = hit_series(turns)
    pairs = gap_hit_pairs(turns)
    kv = kv_usage_series(args.metrics) if args.metrics and args.metrics.exists() else []

    out_dir = args.out or (args.profiles / "e0")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ordered_csv(out_dir / "turns_ordered.csv", ordered)
    write_bottom_pct_csv(out_dir / "bottom_pct_tools.csv", bottom_rows)

    n_sessions = len({t.session_id for t in turns})
    print(f"turns: {len(turns)} across {n_sessions} sessions "
          f"({len(boundaries)} sample boundaries)")
    _print_bottom(bottom_rows)
    if args.metrics and not kv:
        print("warning: no KV-usage series parsed from --metrics", file=sys.stderr)

    if not args.no_figures:
        try:
            fig_turn_llm_time(ordered, boundaries, out_dir / "fig1_turn_llm_time.pdf")
            fig_hit_vs_kv(hits, kv, out_dir / "fig3_hit_vs_kv.pdf")
            fig_gap_vs_hit(pairs, out_dir / "fig4_gap_vs_hit.pdf")
            print(f"\nwrote figures under {out_dir}")
        except ImportError:
            print("matplotlib not available; wrote CSVs only "
                  "(--no-figures to silence)", file=sys.stderr)

    print(f"wrote {out_dir / 'turns_ordered.csv'}")
    print(f"wrote {out_dir / 'bottom_pct_tools.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
