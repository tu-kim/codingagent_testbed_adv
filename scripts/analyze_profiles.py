#!/usr/bin/env python3
"""Paper-style analysis plots for aggregated OpenCode profile NDJSON.

Consumes either a single aggregated NDJSON file (output of
`scripts/aggregate_profiles.sh`) or a directory of per-session
`<sessionID>.jsonl` files. Produces PDF figures suitable for inclusion
in academic papers (NeurIPS/ICML-style: serif, single-column width,
300 DPI, tight bounding box, spines on left/bottom only).

Figures emitted:
  fig1_session_e2e_latency.pdf   — per-session E2E wall (query.end.duration_s),
                                   sorted bar chart.
  fig2_tool_exec_time.pdf        — per-tool execution time mean ± std bar chart.
  fig3_tool_llm_ratio_turn.pdf   — distribution of (tool_wall / llm_wall) per turn.
  fig4_tool_llm_ratio_tool.pdf   — per-tool (tool_duration / turn_llm_duration)
                                   distribution (boxplot).
  fig5_tool_tokens.pdf           — for each tool call: (left) turn output tokens
                                   that produced the call, (right) input-token
                                   delta added to the next turn by tool results.

Usage:
  scripts/analyze_profiles.py \\
      --input results/run1/profiles.jsonl \\
      --output results/run1/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------- paper style ----------


PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3.3, 2.3),       # single-column width
    "figure.dpi": 150,                   # display
    "savefig.dpi": 300,                  # print
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,                  # embed TrueType so editors can edit text
    "ps.fonttype": 42,
}


# ---------- data model ----------


@dataclass
class ToolCall:
    name: str
    step: int
    duration_s: float
    ok: bool
    output_chars: int | None = None


@dataclass
class Turn:
    step: int
    turn_start_ts: float | None = None
    turn_end_ts: float | None = None
    turn_duration_s: float | None = None
    llm_wall_s: float | None = None           # from turn.end
    tool_wall_s: float | None = None           # from turn.end
    llm_input_tokens: int | None = None        # from llm.end.tokens
    llm_output_tokens: int | None = None
    llm_cache_read: int = 0
    llm_step_duration_s: float | None = None
    llm_stream_end_s: float | None = None
    tools: list[ToolCall] = field(default_factory=list)

    @property
    def llm_effective_input(self) -> int | None:
        """Dynamo-side ISL = adjusted input + cache.read."""
        if self.llm_input_tokens is None:
            return None
        return self.llm_input_tokens + self.llm_cache_read


@dataclass
class Session:
    session_id: str
    query_start_ts: float | None = None
    query_end_ts: float | None = None
    e2e_duration_s: float | None = None
    turns: dict[int, Turn] = field(default_factory=dict)
    aborted: bool = False


# ---------- ingest ----------


def _iter_event_files(path: Path):
    if path.is_file():
        yield path
        return
    for f in sorted(path.glob("*.jsonl")):
        yield f


def load_sessions(path: Path) -> dict[str, Session]:
    sessions: dict[str, Session] = {}

    def ensure_session(sid: str) -> Session:
        s = sessions.get(sid)
        if s is None:
            s = Session(session_id=sid)
            sessions[sid] = s
        return s

    def ensure_turn(s: Session, step: int) -> Turn:
        t = s.turns.get(step)
        if t is None:
            t = Turn(step=step)
            s.turns[step] = t
        return t

    for f in _iter_event_files(path):
        with f.open(encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ev_type = ev.get("ev")
                sid = ev.get("sessionID")
                if not ev_type or not sid:
                    continue
                sess = ensure_session(sid)
                ts = ev.get("ts")

                if ev_type == "query.start":
                    sess.query_start_ts = ts
                elif ev_type == "query.end":
                    sess.query_end_ts = ts
                    if ev.get("duration_s") is not None:
                        sess.e2e_duration_s = ev["duration_s"]
                    sess.aborted = bool(ev.get("aborted"))

                elif ev_type == "turn.start":
                    step = ev.get("step")
                    if step is None:
                        continue
                    t = ensure_turn(sess, step)
                    t.turn_start_ts = ts
                elif ev_type == "turn.end":
                    step = ev.get("step")
                    if step is None:
                        continue
                    t = ensure_turn(sess, step)
                    t.turn_end_ts = ts
                    t.turn_duration_s = ev.get("duration_s")
                    t.llm_wall_s = ev.get("llm_wall_s")
                    t.tool_wall_s = ev.get("tool_wall_s")

                elif ev_type == "llm.end":
                    step = ev.get("step")
                    if step is None:
                        continue
                    t = ensure_turn(sess, step)
                    t.llm_step_duration_s = ev.get("step_duration_s")
                    t.llm_stream_end_s = ev.get("stream_end_s") or ev.get("duration_s")
                    tokens = ev.get("tokens") or {}
                    # AI SDK v5/v6 shape; fall back to classic
                    inp = tokens.get("input")
                    out = tokens.get("output")
                    cache = tokens.get("cache") or {}
                    if inp is None:
                        inp = tokens.get("prompt_tokens")
                        out = tokens.get("completion_tokens")
                    t.llm_input_tokens = inp
                    t.llm_output_tokens = out
                    t.llm_cache_read = (cache.get("read") if isinstance(cache, dict) else 0) or 0

                elif ev_type == "tool.end":
                    step = ev.get("step")
                    name = ev.get("name") or "?"
                    dur = ev.get("duration_s")
                    if step is None or dur is None:
                        continue
                    t = ensure_turn(sess, step)
                    t.tools.append(
                        ToolCall(
                            name=name,
                            step=step,
                            duration_s=float(dur),
                            ok=bool(ev.get("ok", True)),
                            output_chars=ev.get("output_chars"),
                        )
                    )

    # Backfill e2e_duration_s from ts if duration_s was missing
    for s in sessions.values():
        if s.e2e_duration_s is None and s.query_start_ts and s.query_end_ts:
            s.e2e_duration_s = s.query_end_ts - s.query_start_ts

    return sessions


# ---------- metric helpers ----------


def per_tool_duration_stats(sessions: dict[str, Session]):
    """{tool_name: (mean_s, std_s, n)}"""
    bucket: dict[str, list[float]] = defaultdict(list)
    for s in sessions.values():
        for t in s.turns.values():
            for tc in t.tools:
                if tc.ok:
                    bucket[tc.name].append(tc.duration_s)
    return {
        name: (float(np.mean(v)), float(np.std(v)), len(v))
        for name, v in sorted(bucket.items(), key=lambda kv: -np.mean(kv[1]))
    }


def turn_ratio_distribution(sessions: dict[str, Session]) -> list[float]:
    """tool_wall_s / llm_wall_s per turn; skip turns missing either."""
    out: list[float] = []
    for s in sessions.values():
        for t in s.turns.values():
            if t.tool_wall_s is None or t.llm_wall_s is None or t.llm_wall_s <= 0:
                continue
            out.append(t.tool_wall_s / t.llm_wall_s)
    return out


def per_tool_ratio_distribution(sessions: dict[str, Session]) -> dict[str, list[float]]:
    """For each tool call: tool.duration_s / corresponding turn's llm wall.
    Grouped by tool name."""
    bucket: dict[str, list[float]] = defaultdict(list)
    for s in sessions.values():
        for t in s.turns.values():
            llm = t.llm_stream_end_s or t.llm_wall_s
            if llm is None or llm <= 0:
                continue
            for tc in t.tools:
                if tc.ok:
                    bucket[tc.name].append(tc.duration_s / llm)
    return bucket


def tool_token_pairs(sessions: dict[str, Session]):
    """Yields (tool_name, turn_output_tokens, next_turn_input_added).

    `next_turn_input_added` = next_turn's effective input − this_turn's
    effective input − this_turn's output. This equals the total
    "tool result" payload appended between turns. When a turn has
    multiple tool calls we attribute the SAME delta to each (caller
    decides how to handle).
    """
    out = []
    for s in sessions.values():
        sorted_steps = sorted(s.turns.keys())
        for i, step in enumerate(sorted_steps):
            t = s.turns[step]
            if t.llm_output_tokens is None:
                continue
            next_step = sorted_steps[i + 1] if i + 1 < len(sorted_steps) else None
            next_t = s.turns.get(next_step) if next_step is not None else None
            if next_t is None or next_t.llm_effective_input is None or t.llm_effective_input is None:
                added = None
            else:
                added = next_t.llm_effective_input - t.llm_effective_input - t.llm_output_tokens
                added = max(0, added)  # clamp (next turn may have compacted)
            for tc in t.tools:
                if tc.ok:
                    out.append((tc.name, t.llm_output_tokens, added))
    return out


# ---------- plotting ----------


def _stat_lines(
    ax,
    values: list[float],
    *,
    orient: str = "h",   # "h" = horizontal axhlines (for value distributions on y), "v" = axvlines
    stats: tuple[str, ...] = ("mean", "p90", "p99"),
    fmt: str = ".2f",
    unit: str = "",
) -> None:
    """Overlay mean/p90/p99 (or median/p90/p99) reference lines with
    inline value labels in a small font. Used by fig1 and fig3."""
    if not values:
        return
    arr = np.asarray(values, dtype=float)
    if "median" in stats and "mean" in stats:
        # rare; if both wanted, plot both
        pass
    spec: list[tuple[str, float, str]] = []
    for name in stats:
        if name == "mean":
            spec.append(("mean", float(np.mean(arr)), "C0"))
        elif name == "median":
            spec.append(("median", float(np.median(arr)), "C0"))
        elif name == "p90":
            spec.append(("p90", float(np.percentile(arr, 90)), "C2"))
        elif name == "p99":
            spec.append(("p99", float(np.percentile(arr, 99)), "C3"))
    # stagger label y-positions when lines are dense to reduce overlap
    label_axis_pos = [0.97, 0.85, 0.73, 0.61]
    for i, (label, value, color) in enumerate(spec):
        if orient == "h":
            ax.axhline(value, color=color, linestyle="--", linewidth=0.7, alpha=0.85)
            trans = ax.get_yaxis_transform()
            ax.text(
                0.99, value, f" {label}={value:{fmt}}{unit}",
                transform=trans, va="center", ha="right", fontsize=6.0,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
            )
        else:  # vertical lines on a histogram-like x-axis
            ax.axvline(value, color=color, linestyle="--", linewidth=0.7, alpha=0.85)
            trans = ax.get_xaxis_transform()
            y_axis_pos = label_axis_pos[i % len(label_axis_pos)]
            ax.text(
                value, y_axis_pos, f"{label}={value:{fmt}}{unit}",
                transform=trans, va="top", ha="left", fontsize=6.0,
                color=color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
            )


def plot_session_e2e(sessions: dict[str, Session], out: Path) -> Path:
    durs = sorted(
        (s.e2e_duration_s for s in sessions.values() if s.e2e_duration_s is not None),
        reverse=False,
    )
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    if not durs:
        ax.text(0.5, 0.5, "no sessions with e2e duration",
                ha="center", va="center", transform=ax.transAxes)
    else:
        x = np.arange(len(durs))
        ax.bar(x, durs, color="0.4", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("Session (sorted by latency)")
        ax.set_ylabel("E2E latency (s)")
        _stat_lines(ax, durs, orient="h",
                    stats=("mean", "p90", "p99"), fmt=".1f", unit="s")
        ax.set_xlim(-0.5, len(durs) - 0.5)
    fig.tight_layout()
    path = out / "fig1_session_e2e_latency.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_tool_exec_time(sessions: dict[str, Session], out: Path) -> Path:
    stats = per_tool_duration_stats(sessions)
    fig, ax = plt.subplots(figsize=(3.7, 2.5))   # slightly wider for inline values
    if not stats:
        ax.text(0.5, 0.5, "no successful tool calls",
                ha="center", va="center", transform=ax.transAxes)
    else:
        names = list(stats.keys())
        means = np.array([stats[n][0] for n in names])
        stds = np.array([stats[n][1] for n in names])
        ns = [stats[n][2] for n in names]
        y = np.arange(len(names))
        # Log x-axis handles wide range (e.g. `task` sub-agent ~30s
        # vs `read` ~0.05s) so the smaller tools stay visible.
        eps = 1e-3
        means_plot = np.maximum(means, eps)
        # Asymmetric error bars: clip lower end at eps so log scale
        # doesn't hit zero or negative values (mean - std can be < 0).
        lower = np.minimum(stds, means_plot - eps)
        upper = stds
        ax.barh(y, means_plot, xerr=[lower, upper], color="0.4",
                edgecolor="black", linewidth=0.5,
                error_kw={"linewidth": 0.6, "capsize": 1.5})
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Execution time (s, log scale)")
        ax.set_xscale("log")
        ax.set_xlim(left=eps)
        # Annotate each bar with mean ± std (n) in small font next to
        # the bar's RIGHT edge -- bypasses log-scale visual asymmetry.
        for i, (m, s, c) in enumerate(zip(means, stds, ns)):
            ax.text(
                max(m + s, eps) * 1.4, i,
                f"{m:.2f}±{s:.2f} (n={c})",
                va="center", ha="left", fontsize=6.0, color="0.2",
            )
    fig.tight_layout()
    path = out / "fig2_tool_exec_time.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ratio_per_turn(sessions: dict[str, Session], out: Path) -> Path:
    ratios = turn_ratio_distribution(sessions)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    if not ratios:
        ax.text(0.5, 0.5, "no turns with both tool and llm wall",
                ha="center", va="center", transform=ax.transAxes)
    else:
        arr = np.array(ratios)
        # Log x-axis spreads the dense 0-1 region while still
        # showing the long tail. Clip exact zeros to a small eps for
        # the log axis; the underlying stats are computed on raw arr.
        eps = max(1e-3, float(arr[arr > 0].min()) if (arr > 0).any() else 1e-3)
        arr_pos = np.where(arr <= 0, eps, arr)
        upper = float(arr_pos.max() * 1.1)
        bins = np.logspace(np.log10(eps), np.log10(upper), 40)
        ax.hist(arr_pos, bins=bins, color="0.5", edgecolor="black", linewidth=0.4)
        ax.set_xscale("log")
        ax.set_xlabel(r"Tool wall / LLM wall per turn (log scale)")
        ax.set_ylabel("Turn count")
        # Stats: median / p90 / p99 vertical lines + small inline values
        _stat_lines(ax, ratios, orient="v",
                    stats=("median", "p90", "p99"), fmt=".2f")
        ax.axvline(1.0, color="0.7", linestyle=":", linewidth=0.5)
    fig.tight_layout()
    path = out / "fig3_tool_llm_ratio_turn.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ratio_per_tool(sessions: dict[str, Session], out: Path) -> Path:
    by_tool = per_tool_ratio_distribution(sessions)
    # Wider figure + room on left for the y-axis label that was being
    # cropped at default size.
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    if not by_tool:
        ax.text(0.5, 0.5, "no tool/llm pairs",
                ha="center", va="center", transform=ax.transAxes)
    else:
        items = sorted(by_tool.items(), key=lambda kv: -np.median(kv[1]))
        names = [k for k, _ in items]
        data = [v for _, v in items]
        # Log y-axis: `task` ratios can be 100x larger than `read`,
        # without log the boxes for small tools collapse to a flat line.
        eps = 1e-3
        data_log = [np.maximum(np.asarray(d, dtype=float), eps) for d in data]
        ax.boxplot(
            data_log,
            widths=0.55,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "0.85", "linewidth": 0.6},
            medianprops={"color": "C3", "linewidth": 1.0},
            whiskerprops={"linewidth": 0.5},
            capprops={"linewidth": 0.5},
        )
        ax.set_xticks(range(1, len(names) + 1))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Tool duration / Turn LLM wall\n(log scale)")
        ax.set_yscale("log")
        ax.axhline(1.0, color="0.7", linestyle=":", linewidth=0.6)
    # Reserve extra left margin so the multi-line y-label isn't clipped.
    fig.tight_layout()
    fig.subplots_adjust(left=0.20)
    path = out / "fig4_tool_llm_ratio_tool.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_tool_tokens(sessions: dict[str, Session], out: Path) -> Path:
    pairs = tool_token_pairs(sessions)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), sharey=False)
    ax_out, ax_in = axes
    by_tool_out: dict[str, list[int]] = defaultdict(list)
    by_tool_in: dict[str, list[int]] = defaultdict(list)
    if not pairs:
        for ax in axes:
            ax.text(0.5, 0.5, "no tool calls with token data",
                    ha="center", va="center", transform=ax.transAxes)
    else:
        for name, out_tok, in_added in pairs:
            by_tool_out[name].append(out_tok)
            if in_added is not None:
                by_tool_in[name].append(in_added)

        names = sorted(by_tool_out.keys(), key=lambda n: -float(np.mean(by_tool_out[n])))

        def _box(ax, bucket: dict[str, list[int]], ylabel: str):
            data = [bucket.get(n, []) for n in names]
            if not any(data):
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes)
                return
            ax.boxplot(
                data,
                widths=0.55,
                showfliers=False,
                patch_artist=True,
                boxprops={"facecolor": "0.85", "linewidth": 0.6},
                medianprops={"color": "C3", "linewidth": 1.0},
                whiskerprops={"linewidth": 0.5},
                capprops={"linewidth": 0.5},
            )
            ax.set_xticks(range(1, len(names) + 1))
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_ylabel(ylabel)

        _box(ax_out, by_tool_out, "Turn output tokens")
        _box(ax_in, by_tool_in, "Next-turn input added (tokens)")
        ax_out.set_title("(a) Generating turn", fontsize=9)
        ax_in.set_title("(b) Tool-result payload", fontsize=9)

    fig.tight_layout()
    path = out / "fig5_tool_tokens.pdf"
    fig.savefig(path)
    plt.close(fig)

    # ----- companion mean/std table for fig5 -----
    _write_tool_tokens_table(by_tool_out, by_tool_in, out)
    return path


def _write_tool_tokens_table(
    by_tool_out: dict[str, list[int]],
    by_tool_in: dict[str, list[int]],
    out: Path,
) -> Path:
    """Write fig5's underlying mean/std numbers as a CSV + print to stdout.

    Columns: tool, n_calls, turn_out_mean, turn_out_std,
             next_in_added_mean, next_in_added_std.
    """
    import csv

    csv_path = out / "fig5_tool_tokens_stats.csv"
    names = sorted(
        by_tool_out.keys(),
        key=lambda n: -float(np.mean(by_tool_out[n])) if by_tool_out[n] else 0.0,
    )

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tool", "n_calls",
            "turn_out_mean", "turn_out_std",
            "next_in_added_mean", "next_in_added_std",
        ])
        for name in names:
            out_arr = np.asarray(by_tool_out[name], dtype=float)
            in_arr = np.asarray(by_tool_in.get(name, []), dtype=float)
            w.writerow([
                name, len(out_arr),
                f"{out_arr.mean():.1f}" if out_arr.size else "",
                f"{out_arr.std():.1f}" if out_arr.size else "",
                f"{in_arr.mean():.1f}" if in_arr.size else "",
                f"{in_arr.std():.1f}" if in_arr.size else "",
            ])

    # Pretty-print to stdout as well
    print()
    print("fig5 mean/std table (tokens per tool call):")
    header = f"{'tool':<12} {'n':>5} {'out_μ':>9} {'out_σ':>9} {'added_μ':>10} {'added_σ':>10}"
    print(header)
    print("-" * len(header))
    for name in names:
        out_arr = np.asarray(by_tool_out[name], dtype=float)
        in_arr = np.asarray(by_tool_in.get(name, []), dtype=float)
        in_m = f"{in_arr.mean():>10.1f}" if in_arr.size else f"{'-':>10}"
        in_s = f"{in_arr.std():>10.1f}" if in_arr.size else f"{'-':>10}"
        print(
            f"{name:<12} {len(out_arr):>5} {out_arr.mean():>9.1f} "
            f"{out_arr.std():>9.1f} {in_m} {in_s}"
        )
    print(f"  (csv: {csv_path})")
    return csv_path


# ---------- entry point ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input", required=True, type=Path,
        help="Aggregated NDJSON file OR directory containing <session>.jsonl files",
    )
    ap.add_argument(
        "--output", required=True, type=Path,
        help="Directory to write fig{1..5}_*.pdf into (created if missing)",
    )
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions(args.input)
    if not sessions:
        print("no sessions parsed from input", file=sys.stderr)
        return 1

    n_turns = sum(len(s.turns) for s in sessions.values())
    n_tools = sum(len(t.tools) for s in sessions.values() for t in s.turns.values())
    print(f"loaded {len(sessions)} sessions, {n_turns} turns, {n_tools} tool calls")

    plt.rcParams.update(PAPER_STYLE)

    for fn in (
        plot_session_e2e,
        plot_tool_exec_time,
        plot_ratio_per_turn,
        plot_ratio_per_tool,
        plot_tool_tokens,
    ):
        path = fn(sessions, args.output)
        print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
