#!/usr/bin/env python3
"""Per-request latency composition analysis (pandas).

Consumes a LONG-FORM CSV where each row is one request (in this testbed:
one OpenCode turn) with its latency broken into additive components. The
canonical input is the per-turn dump emitted by
`scripts/analyze_profiles.py`:

    <fig-out>/fig6_turn_decomposition_per_turn.csv
      columns: session_id, step, duration_s, llm_wall_s, tool_wall_s, others_s

but any CSV with a total column + component columns works (see --components
/ --total). task-tool turns are already excluded upstream by fig6.

Two ways to weight a "share", and this script reports BOTH because they
answer different questions:
  * pooled share  = Σ component / Σ total  (time-weighted: where the
                    wall-clock actually goes; long requests dominate).
  * per-request   = mean/percentiles of (component / total) per row
                    (each request weighted equally: the typical request's
                    shape). These do NOT sum to 1 across components at a
                    given percentile -- each percentile is drawn from a
                    different set of requests.

Outputs (into --output):
  latency_pooled_share.csv          — (1) pooled/time-weighted share per component
  latency_per_request_share.csv     — (2) per-request share distribution
                                          (mean/p50/p90/p99 + p25/p75/min/max)
  latency_conditional_by_bucket.csv — (3) mean per-request share per component,
                                          conditioned on total-latency bucket
                                          split at the p50/p90/p99 of total
  latency_share_violin.png          — per-component per-request share distribution
  latency_sorted_stacked_bar.png    — requests sorted by total latency, stacked
                                          ABSOLUTE component seconds (height=total)
  latency_bucket_stacked_bar.png    — mean share per component per total-latency
                                          bucket (each bar sums to ~1)

Usage:
  scripts/analyze_latency_breakdown.py \\
      --input  results/run1/figures/fig6_turn_decomposition_per_turn.csv \\
      --output results/run1/latency_breakdown
  # custom columns:
  scripts/analyze_latency_breakdown.py --input x.csv --output out \\
      --components llm_wall_s tool_wall_s others_s --total duration_s
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on the GPU host
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


DEFAULT_COMPONENTS = ["llm_wall_s", "tool_wall_s", "others_s"]
DEFAULT_TOTAL = "duration_s"

# Stable component -> color so all three figures agree.
_COLOR_CYCLE = ["C0", "C2", "0.6", "C1", "C3", "C4", "C5", "C6"]

# Total-latency buckets, split at the p50/p90/p99 of the total column.
# Comparison is `<=` on the lower edge so ties land in the lower bucket
# (robust to duplicate thresholds, unlike pd.cut with non-unique bins).
BUCKET_ORDER = ["<=p50", "p50-p90", "p90-p99", ">p99"]


def _load(input_path: Path, components: list[str],
          total: str | None) -> tuple[pd.DataFrame, str]:
    """Read the long-form CSV, coerce numerics, drop unusable rows.

    `comment="#"` lets this tolerate the annotated stats CSVs too, though the
    per-turn dump is already clean."""
    df = pd.read_csv(input_path, comment="#", skip_blank_lines=True)

    missing = [c for c in components if c not in df.columns]
    if missing:
        sys.exit(
            f"error: component column(s) {missing} not in {input_path} "
            f"(have: {list(df.columns)})"
        )

    for c in components:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if total and total in df.columns:
        df[total] = pd.to_numeric(df[total], errors="coerce")
        total_col = total
    else:
        # No total column (or the named one is absent): reconstruct it as the
        # sum of components. Shares then sum to exactly 1 by construction.
        if total and total not in df.columns:
            print(
                f"note: total column '{total}' absent; using sum of components",
                file=sys.stderr,
            )
        total_col = "_total_s"
        df[total_col] = df[components].sum(axis=1)

    n0 = len(df)
    df = df.dropna(subset=components + [total_col])
    df = df[df[total_col] > 0].reset_index(drop=True)
    dropped = n0 - len(df)
    if dropped:
        print(f"note: dropped {dropped}/{n0} rows (NaN component or total<=0)",
              file=sys.stderr)
    if df.empty:
        sys.exit("error: no usable rows after cleaning")
    return df, total_col


def _shares(df: pd.DataFrame, components: list[str], total_col: str) -> pd.DataFrame:
    """Per-request share of each component (component / total), one col each."""
    out = pd.DataFrame(index=df.index)
    for c in components:
        out[c] = df[c] / df[total_col]
    return out


def _pooled_share(df, components, total_col) -> pd.DataFrame:
    """(1) Time-weighted share = Σ component / Σ total."""
    grand = float(df[total_col].sum())
    rows = []
    for c in components:
        s = float(df[c].sum())
        rows.append({
            "component": c,
            "total_seconds": s,
            "pooled_share": (s / grand) if grand > 0 else float("nan"),
        })
    rows.append({
        "component": "TOTAL",
        "total_seconds": grand,
        "pooled_share": 1.0 if grand > 0 else float("nan"),
    })
    return pd.DataFrame(rows)


def _per_request_share_table(shares: pd.DataFrame, components: list[str]) -> pd.DataFrame:
    """(2) Distribution of per-request shares, each request weighted equally."""
    rows = []
    for c in components:
        v = shares[c].to_numpy(dtype=float)
        rows.append({
            "component": c,
            "n_requests": v.size,
            "mean": float(np.mean(v)),
            "p50": float(np.percentile(v, 50)),
            "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)),
            "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
        })
    return pd.DataFrame(rows)


def _assign_bucket(total: pd.Series) -> tuple[pd.Series, dict]:
    """Bucket each request by its total latency, split at p50/p90/p99."""
    p50, p90, p99 = (float(total.quantile(q)) for q in (0.5, 0.9, 0.99))

    def which(t: float) -> str:
        if t <= p50:
            return "<=p50"
        if t <= p90:
            return "p50-p90"
        if t <= p99:
            return "p90-p99"
        return ">p99"

    cat = pd.Categorical(total.map(which), categories=BUCKET_ORDER, ordered=True)
    return pd.Series(cat, index=total.index), {"p50": p50, "p90": p90, "p99": p99}


def _conditional_by_bucket(df, shares, components, total_col) -> pd.DataFrame:
    """(3) Mean per-request share per component, conditioned on total bucket."""
    bucket, thr = _assign_bucket(df[total_col])
    work = shares.copy()
    work["_bucket"] = bucket
    work["_total"] = df[total_col].to_numpy()

    rows = []
    for b in BUCKET_ORDER:
        sub = work[work["_bucket"] == b]
        row = {
            "bucket": b,
            "n_requests": len(sub),
            "mean_total_s": float(sub["_total"].mean()) if len(sub) else float("nan"),
        }
        for c in components:
            row[f"{c}_mean_share"] = float(sub[c].mean()) if len(sub) else float("nan")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["thresholds"] = thr
    return out


# ---------- figures ----------

def _colors(components: list[str]) -> dict:
    return {c: _COLOR_CYCLE[i % len(_COLOR_CYCLE)] for i, c in enumerate(components)}


def _fig_violin(shares, components, colors, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(components), 3.2))
    for i, c in enumerate(components):
        pos = i + 1
        v = shares[c].to_numpy(dtype=float)
        color = colors[c]
        # gaussian_kde (inside violinplot) is singular when the sample has
        # zero variance (e.g. a component that is always 0). Fall back to a
        # median marker + point cloud for degenerate/tiny samples.
        if v.size >= 2 and float(np.var(v)) > 1e-12:
            parts = ax.violinplot([v], positions=[pos],
                                  showmedians=True, showextrema=True)
            parts["bodies"][0].set_facecolor(color)
            parts["bodies"][0].set_alpha(0.7)
            parts["bodies"][0].set_edgecolor("black")
            parts["bodies"][0].set_linewidth(0.5)
            for key in ("cbars", "cmins", "cmaxes", "cmedians"):
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(0.8)
        else:
            med = float(np.median(v)) if v.size else 0.0
            ax.scatter([pos], [med], color=color, edgecolor="black",
                       zorder=3, s=30)
            ax.hlines(med, pos - 0.25, pos + 0.25, color="black", linewidth=0.8)
    ax.set_xticks(range(1, len(components) + 1))
    ax.set_xticklabels(components, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("per-request share of total latency")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-request latency-component share", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _fig_sorted_stacked(df, components, colors, total_col, out_path: Path) -> None:
    order = df[total_col].to_numpy(dtype=float).argsort()
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for c in components:
        vals = df[c].to_numpy(dtype=float)[order]
        ax.bar(x, vals, bottom=bottom, width=1.0, linewidth=0,
               color=colors[c], label=c)
        bottom += vals
    ax.set_xlabel("request (sorted by total latency, ascending)")
    ax.set_ylabel("latency (s)")
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_title("Latency composition vs magnitude", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=len(components), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _fig_bucket_stacked(cond, components, colors, out_path: Path) -> None:
    present = cond[cond["n_requests"] > 0]
    labels = present["bucket"].tolist()
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    fig, ax = plt.subplots(figsize=(1.8 + 1.0 * len(labels), 3.4))
    for c in components:
        vals = present[f"{c}_mean_share"].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.7, color=colors[c],
               edgecolor="black", linewidth=0.4, label=c)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{b}\n(n={int(n)})" for b, n in zip(labels, present["n_requests"])],
        fontsize=8,
    )
    ax.set_ylabel("mean per-request share")
    ax.set_ylim(0, 1.02)
    ax.set_title("Composition conditioned on total-latency bucket", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=len(components),
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, type=Path,
                    help="long-form per-request CSV (one row per request/turn)")
    ap.add_argument("--output", required=True, type=Path,
                    help="output directory for CSVs + PNGs")
    ap.add_argument("--components", nargs="+", default=DEFAULT_COMPONENTS,
                    help=f"component columns (default: {DEFAULT_COMPONENTS})")
    ap.add_argument("--total", default=DEFAULT_TOTAL,
                    help=f"total-latency column (default: {DEFAULT_TOTAL}); "
                         "if absent, computed as the sum of components")
    args = ap.parse_args(argv)

    if not args.input.exists():
        sys.exit(f"error: input not found: {args.input}")
    args.output.mkdir(parents=True, exist_ok=True)

    df, total_col = _load(args.input, args.components, args.total)
    shares = _shares(df, args.components, total_col)
    colors = _colors(args.components)

    # ----- (1) pooled -----
    pooled = _pooled_share(df, args.components, total_col)
    pooled.to_csv(args.output / "latency_pooled_share.csv", index=False)

    # ----- (2) per-request distribution -----
    per_req = _per_request_share_table(shares, args.components)
    per_req.to_csv(args.output / "latency_per_request_share.csv", index=False)

    # ----- (3) conditional by total-latency bucket -----
    cond = _conditional_by_bucket(df, shares, args.components, total_col)
    cond.to_csv(args.output / "latency_conditional_by_bucket.csv", index=False)

    # ----- figures -----
    _fig_violin(shares, args.components, colors,
                args.output / "latency_share_violin.png")
    _fig_sorted_stacked(df, args.components, colors, total_col,
                        args.output / "latency_sorted_stacked_bar.png")
    _fig_bucket_stacked(cond, args.components, colors,
                        args.output / "latency_bucket_stacked_bar.png")

    # ----- stdout summary -----
    thr = cond.attrs.get("thresholds", {})
    print(f"n_requests = {len(df)}   (total col: {total_col})")
    print(f"total-latency thresholds: p50={thr.get('p50', float('nan')):.3f}s  "
          f"p90={thr.get('p90', float('nan')):.3f}s  "
          f"p99={thr.get('p99', float('nan')):.3f}s")
    print("\n(1) pooled / time-weighted share:")
    for _, r in pooled.iterrows():
        if r["component"] == "TOTAL":
            continue
        print(f"  {r['component']:<14} {r['pooled_share']:>7.2%}  "
              f"({r['total_seconds']:.1f}s)")
    print("\n(2) per-request share (each request weighted equally):")
    print(f"  {'component':<14} {'mean':>7} {'p50':>7} {'p90':>7} {'p99':>7}")
    for _, r in per_req.iterrows():
        print(f"  {r['component']:<14} {r['mean']:>7.1%} {r['p50']:>7.1%} "
              f"{r['p90']:>7.1%} {r['p99']:>7.1%}")
    print("\n(3) mean share conditioned on total-latency bucket:")
    hdr = "  " + f"{'bucket':<10} {'n':>5} {'mean_tot_s':>11}  " + \
          " ".join(f"{c:>14}" for c in args.components)
    print(hdr)
    for _, r in cond.iterrows():
        if r["n_requests"] == 0:
            continue
        shares_str = " ".join(f"{r[f'{c}_mean_share']:>14.1%}" for c in args.components)
        print(f"  {r['bucket']:<10} {int(r['n_requests']):>5} "
              f"{r['mean_total_s']:>11.2f}  {shares_str}")
    print(f"\nwrote CSVs + PNGs to {args.output}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
