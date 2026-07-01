#!/usr/bin/env python3
"""Paper-style analysis plots for aggregated OpenCode profile NDJSON.

Consumes either a single aggregated NDJSON file (output of
`scripts/aggregate_profiles.sh`) or a directory of per-session
`<sessionID>.jsonl` files. Produces PDF figures suitable for inclusion
in academic papers (NeurIPS/ICML-style: serif, single-column width,
300 DPI, tight bounding box, spines on left/bottom only).

Failure / abort policy (intentional, do not "clean up" silently):
  * Failed tool calls (tool.end.ok == false) are EXCLUDED from per-tool
    stats (fig2 / fig4 / fig5) so a tool's mean isn't poisoned by
    error-path durations.
  * Aborted sessions (query.end.aborted == true) are INCLUDED in
    session-level metrics (fig1) -- this is real-world latency the
    system produced.
  * Errored LLM steps (llm.end.finish == "error" etc.) are INCLUDED in
    turn-level stats (fig3 / fig6) -- same rationale.
  * Tasks the runner never started (clone / session-create failure
    before opencode is reached) leave no NDJSON, so they're naturally
    absent.
Tweak per-call by editing the `if tc.ok` filters and adding an
`s.aborted` skip if a different policy is needed.

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
  fig6_turn_decomposition.pdf    — mean turn duration broken into LLM /
                                   tool / post-overhead segments. Turns that
                                   fired the `task` tool are EXCLUDED wholesale:
                                   `task` runs a nested agent session (its own
                                   LLM + tools) whose wall time lands in this
                                   turn's tool_wall, so it's downstream LLM
                                   work, not the plain "one LLM step + local
                                   tools" shape. Task turns are analyzed
                                   separately. Companion stats CSV
                                   (mean/median/p90/p99 per component + average
                                   per-turn ratio).
  fig6b_turn_share_distribution.pdf — empirical CDF of each component's
                                   (llm/tool/others) per-turn share of duration,
                                   overlaid in one graph.
  fig7_post_overhead_breakdown.pdf — post_overhead (others) split into
                                   snapshot+DB vs the rest (see analyze_post_overhead).
  latency_share_violin.pdf,        — per-request latency-composition views on the
  latency_sorted_stacked.pdf,        WALL-CLOCK-anchored components (pooled share,
  latency_bucket_stacked.pdf         per-request share distribution, and
  + latency_*.csv                    conditional-by-total-latency-bucket breakdown).
                                   Merged in from the former standalone
                                   analyze_latency_breakdown.py.

Usage:
  scripts/analyze_profiles.py \\
      --input results/run1/profiles.jsonl \\
      --output results/run1/figures \\
      [--exclude-tools task]   # drop tools from per-tool plots (fig2/4/5)
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


# OpenCode's `task` tool spawns a nested agent session (its own LLM loop +
# tools). Its wall time is downstream LLM time, NOT ordinary tool execution,
# so the turn decomposition splits it into a separate segment. Analyzed in
# detail elsewhere.
TASK_TOOL_NAME = "task"


@dataclass
class ToolCall:
    name: str
    step: int
    duration_s: float
    ok: bool
    output_chars: int | None = None
    args_head: str | None = None       # tool.start.args_head (≤200-char JSON preview)


@dataclass
class Turn:
    step: int
    turn_start_ts: float | None = None
    turn_end_ts: float | None = None
    turn_duration_s: float | None = None
    llm_wall_s: float | None = None           # from turn.end
    tool_wall_s: float | None = None           # from turn.end
    post_overhead_s: float | None = None       # from turn.end (= duration - llm - tool, clamped)
    llm_input_tokens: int | None = None        # from llm.end.tokens
    llm_output_tokens: int | None = None
    llm_cache_read: int = 0
    llm_step_duration_s: float | None = None
    llm_stream_end_s: float | None = None
    llm_end_ts: float | None = None            # wall-clock ts of the llm.end event
    # from llm.end.post_stream_overhead_s = max(0, finish-step - streamFinish):
    # the per-step framework finalization slice (snapshot.track + snapshot.patch
    # + session.updateMessage DB write + EventV2 dual-write; processor.ts:453-497).
    # This is the measurable component INSIDE turn.post_overhead_s. None when the
    # AI SDK "finish" event didn't fire for the step (streamFinish missing) -- in
    # that case llm_wall_s fell back to firstTool/lastText and the LLM stream
    # TAIL leaked into post_overhead instead.
    post_stream_overhead_s: float | None = None
    stream_finish_fired: bool | None = None    # llm.end carried a non-null post_stream_overhead_s
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
    # tool.start carries args_head (the command preview); tool.end carries the
    # duration. Stash args_head by (sid, callID) on start, attach it on end.
    pending_tool_args: dict[tuple[str, str], str] = {}

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
                    t.post_overhead_s = ev.get("post_overhead_s")

                elif ev_type == "llm.end":
                    step = ev.get("step")
                    if step is None:
                        continue
                    t = ensure_turn(sess, step)
                    t.llm_end_ts = ts
                    t.llm_step_duration_s = ev.get("step_duration_s")
                    t.llm_stream_end_s = ev.get("stream_end_s") or ev.get("duration_s")
                    # post_stream_overhead_s is emitted (possibly null) by the
                    # current profile patch. Key-absent => older patch, no data;
                    # key-present-but-null => streamFinish didn't fire this step.
                    if "post_stream_overhead_s" in ev:
                        pso = ev.get("post_stream_overhead_s")
                        t.post_stream_overhead_s = pso
                        t.stream_finish_fired = pso is not None
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

                elif ev_type == "tool.start":
                    cid = ev.get("callID")
                    ah = ev.get("args_head")
                    if cid is not None and ah is not None:
                        pending_tool_args[(sid, cid)] = ah
                elif ev_type == "tool.end":
                    step = ev.get("step")
                    name = ev.get("name") or "?"
                    dur = ev.get("duration_s")
                    if step is None or dur is None:
                        continue
                    t = ensure_turn(sess, step)
                    cid = ev.get("callID")
                    ah = pending_tool_args.pop((sid, cid), None) if cid is not None else None
                    t.tools.append(
                        ToolCall(
                            name=name,
                            step=step,
                            duration_s=float(dur),
                            ok=bool(ev.get("ok", True)),
                            output_chars=ev.get("output_chars"),
                            args_head=ah,
                        )
                    )

    # Backfill e2e_duration_s from ts if duration_s was missing
    for s in sessions.values():
        if s.e2e_duration_s is None and s.query_start_ts and s.query_end_ts:
            s.e2e_duration_s = s.query_end_ts - s.query_start_ts

    return sessions


# ---------- metric helpers ----------


def per_tool_duration_stats(sessions: dict[str, Session],
                             exclude: set[str] | None = None):
    """{tool_name: (mean_s, std_s, n)}"""
    excl = exclude or set()
    bucket: dict[str, list[float]] = defaultdict(list)
    for s in sessions.values():
        for t in s.turns.values():
            for tc in t.tools:
                if tc.ok and tc.name not in excl:
                    bucket[tc.name].append(tc.duration_s)
    return {
        name: (float(np.mean(v)), float(np.std(v)), len(v))
        for name, v in sorted(bucket.items(), key=lambda kv: -np.mean(kv[1]))
    }


def turn_ratio_distribution(sessions: dict[str, Session]) -> list[float]:
    """tool_wall_s / llm_wall_s per turn; skip turns missing either."""
    return [r for _, _, _, _, r in turn_ratio_rows(sessions)]


def turn_ratio_rows(sessions: dict[str, Session]):
    """Per-turn rows for CSV: (session_id, step, tool_wall_s, llm_wall_s, ratio)."""
    out = []
    for sid, s in sorted(sessions.items()):
        for step, t in sorted(s.turns.items()):
            if t.tool_wall_s is None or t.llm_wall_s is None or t.llm_wall_s <= 0:
                continue
            out.append((sid, step, t.tool_wall_s, t.llm_wall_s,
                        t.tool_wall_s / t.llm_wall_s))
    return out


def per_tool_ratio_rows(sessions: dict[str, Session],
                          exclude: set[str] | None = None):
    """Per-tool-call rows for fig4 CSV: (session_id, step, tool, ratio)."""
    excl = exclude or set()
    out = []
    for sid, s in sorted(sessions.items()):
        for step, t in sorted(s.turns.items()):
            llm = t.llm_stream_end_s or t.llm_wall_s
            if llm is None or llm <= 0:
                continue
            for tc in t.tools:
                if tc.ok and tc.name not in excl:
                    out.append((sid, step, tc.name, tc.duration_s / llm))
    return out


def per_tool_ratio_distribution(sessions: dict[str, Session],
                                 exclude: set[str] | None = None) -> dict[str, list[float]]:
    """For each tool call: tool.duration_s / corresponding turn's llm wall.
    Grouped by tool name."""
    excl = exclude or set()
    bucket: dict[str, list[float]] = defaultdict(list)
    for s in sessions.values():
        for t in s.turns.values():
            llm = t.llm_stream_end_s or t.llm_wall_s
            if llm is None or llm <= 0:
                continue
            for tc in t.tools:
                if tc.ok and tc.name not in excl:
                    bucket[tc.name].append(tc.duration_s / llm)
    return bucket


def tool_token_pairs(sessions: dict[str, Session],
                       exclude: set[str] | None = None):
    """Yields (tool_name, turn_output_tokens, next_turn_input_added).

    `next_turn_input_added` = next_turn's effective input − this_turn's
    effective input − this_turn's output. This equals the total
    "tool result" payload appended between turns. When a turn has
    multiple tool calls we attribute the SAME delta to each (caller
    decides how to handle).
    """
    excl = exclude or set()
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
                if tc.ok and tc.name not in excl:
                    out.append((tc.name, t.llm_output_tokens, added))
    return out


# ---------- plotting ----------


def _annotate_exclusion(ax, exclude_tools) -> None:
    """Print a tiny `(excl: <names>)` note in the upper-right corner of
    a plot when the underlying data filtered some tools out. Keeps the
    reader from mistaking a reduced bar set for missing data."""
    if not exclude_tools:
        return
    label = "(excl: " + ", ".join(sorted(exclude_tools)) + ")"
    ax.text(
        0.99, 0.99, label,
        transform=ax.transAxes, ha="right", va="top", fontsize=6.0,
        color="0.35",
    )


def _summary_stats(values) -> dict[str, float]:
    """Return mean / median / p90 / p99 for a numeric sequence."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def _write_csv_with_stats(
    path: Path,
    header: list[str],
    rows: list[tuple],
    *,
    stats_values: list[float] | None = None,
    stats_label: str = "value",
) -> None:
    """Write `rows` under `header`, then append a `# stats:` section with
    mean / median / p90 / p99 of `stats_values` (defaults to None = skip)."""
    import csv

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)
        if stats_values is not None and len(stats_values) > 0:
            stats = _summary_stats(stats_values)
            f.write("\n# summary stats for " + stats_label + "\n")
            sw = csv.writer(f)
            sw.writerow(["stat", stats_label])
            for name in ("mean", "median", "p90", "p99"):
                sw.writerow([name, f"{stats[name]:.4f}"])


def _detect_outlier_index(values, threshold: float = 3.0) -> int | None:
    """Return index of a single dominant outlier whose value exceeds
    `threshold × the second-largest`, else None. Used to decide whether
    to break the axis at the gap between the outlier and everyone else."""
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return None
    order = np.argsort(-arr)
    top, second = arr[order[0]], arr[order[1]]
    if second <= 0:
        return int(order[0]) if top > 0 else None
    if top > threshold * second:
        return int(order[0])
    return None


def _draw_break_marks(ax_lo, ax_hi, orient: str = "x") -> None:
    """Diagonal `//` break marks at the seam between two axes.

    orient="x": ax_lo is LEFT, ax_hi is RIGHT.
    orient="y": ax_lo is BOTTOM, ax_hi is TOP.
    """
    d = 0.018  # half-size in axes-fraction
    common = dict(color="k", clip_on=False, linewidth=0.7)
    if orient == "x":
        ax_lo.spines["right"].set_visible(False)
        ax_hi.spines["left"].set_visible(False)
        kw = dict(common, transform=ax_lo.transAxes)
        ax_lo.plot([1 - d, 1 + d], [-d, +d], **kw)
        ax_lo.plot([1 - d, 1 + d], [1 - d, 1 + d], **kw)
        kw = dict(common, transform=ax_hi.transAxes)
        ax_hi.plot([-d, +d], [-d, +d], **kw)
        ax_hi.plot([-d, +d], [1 - d, 1 + d], **kw)
    elif orient == "y":
        ax_lo.spines["top"].set_visible(False)
        ax_hi.spines["bottom"].set_visible(False)
        kw = dict(common, transform=ax_lo.transAxes)
        ax_lo.plot([-d, +d], [1 - d, 1 + d], **kw)
        ax_lo.plot([1 - d, 1 + d], [1 - d, 1 + d], **kw)
        kw = dict(common, transform=ax_hi.transAxes)
        ax_hi.plot([-d, +d], [-d, +d], **kw)
        ax_hi.plot([1 - d, 1 + d], [-d, +d], **kw)


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
                    stats=("median", "p90", "p99"), fmt=".1f", unit="s")
        ax.set_xlim(-0.5, len(durs) - 0.5)
    fig.tight_layout()
    path = out / "fig1_session_e2e_latency.pdf"
    fig.savefig(path)
    plt.close(fig)
    # Companion CSV: per-session values + summary stats.
    _write_csv_with_stats(
        out / "fig1_session_e2e_latency.csv",
        header=["session_id", "e2e_duration_s"],
        rows=[
            (sid, f"{s.e2e_duration_s:.3f}")
            for sid, s in sorted(sessions.items())
            if s.e2e_duration_s is not None
        ],
        stats_values=durs,
        stats_label="e2e_duration_s",
    )
    return path


def plot_tool_exec_time(sessions: dict[str, Session], out: Path,
                          exclude_tools: set[str] | None = None) -> Path:
    stats = per_tool_duration_stats(sessions, exclude=exclude_tools)
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    if not stats:
        ax.text(0.5, 0.5, "no successful tool calls",
                ha="center", va="center", transform=ax.transAxes)
    else:
        names = list(stats.keys())
        means = np.array([stats[n][0] for n in names])
        stds = np.array([stats[n][1] for n in names])
        ns = [stats[n][2] for n in names]
        # Companion CSV carries the exact values; the plot stays
        # visual-only (no inline labels, no axis break) so it reads
        # cleanly even when one tool like `task` dominates.
        _write_csv_with_stats(
            out / "fig2_tool_exec_time.csv",
            header=["tool", "n_calls", "mean_s", "std_s"],
            rows=[
                (n, ns[i], f"{means[i]:.4f}", f"{stds[i]:.4f}")
                for i, n in enumerate(names)
            ],
            stats_values=None,
        )
        y = np.arange(len(names))
        ax.barh(y, means, xerr=stds, color="0.4", edgecolor="black",
                linewidth=0.5, error_kw={"linewidth": 0.6, "capsize": 1.5})
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("Execution time (s)")
        ax.set_xlim(left=0)
        _annotate_exclusion(ax, exclude_tools)
    fig.tight_layout()
    path = out / "fig2_tool_exec_time.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_ratio_per_turn(sessions: dict[str, Session], out: Path) -> Path:
    ratios = turn_ratio_distribution(sessions)
    if not ratios:
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        ax.text(0.5, 0.5, "no turns with both tool and llm wall",
                ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        path = out / "fig3_tool_llm_ratio_turn.pdf"
        fig.savefig(path)
        plt.close(fig)
        return path

    arr = np.array(ratios, dtype=float)
    # Break the x-axis between p90 (or whatever cap covers the bulk)
    # and the tail's min, when the tail is significantly far from the
    # bulk. Otherwise plot a single linear axis.
    p50 = float(np.median(arr))
    p90 = float(np.percentile(arr, 90))
    p99 = float(np.percentile(arr, 99))
    bulk_cap = p90 * 1.10
    tail = arr[arr > bulk_cap]
    do_break = tail.size > 0 and (tail.min() > 2.0 * bulk_cap)

    if not do_break:
        fig, ax = plt.subplots(figsize=(3.5, 2.4))
        nbins = 40
        # Linear bins from 0 to max+10%
        bins = np.linspace(0.0, max(arr.max(), 1e-6) * 1.1, nbins)
        ax.hist(arr, bins=bins, color="0.5", edgecolor="black", linewidth=0.4)
        ax.set_xlabel(r"Tool wall / LLM wall per turn")
        ax.set_ylabel("Turn count")
        _stat_lines(ax, ratios, orient="v",
                    stats=("median", "p90", "p99"), fmt=".2f")
        ax.axvline(1.0, color="0.7", linestyle=":", linewidth=0.5)
        fig.tight_layout()
    else:
        # Broken x-axis: linear scales on both halves with `//` marks.
        fig, (ax_lo, ax_hi) = plt.subplots(
            1, 2, figsize=(4.0, 2.5), sharey=True,
            gridspec_kw={"width_ratios": [3, 1]},
        )
        bins_lo = np.linspace(0.0, bulk_cap, 35)
        # Span hi: from tail.min to arr.max, padded a bit.
        hi_lo = float(tail.min()) * 0.95
        hi_hi = float(arr.max()) * 1.05
        bins_hi = np.linspace(hi_lo, hi_hi, 12)
        ax_lo.hist(arr, bins=bins_lo, color="0.5", edgecolor="black", linewidth=0.4)
        ax_hi.hist(arr, bins=bins_hi, color="0.5", edgecolor="black", linewidth=0.4)
        ax_lo.set_xlim(0.0, bulk_cap)
        ax_hi.set_xlim(hi_lo, hi_hi)
        ax_lo.set_ylabel("Turn count")
        fig.text(0.5, 0.02, "Tool wall / LLM wall per turn", ha="center", fontsize=9)
        _draw_break_marks(ax_lo, ax_hi, orient="x")
        # Stat lines: place each in the subplot whose x-range contains it.
        trans_lo = ax_lo.get_xaxis_transform()
        trans_hi = ax_hi.get_xaxis_transform()
        for label, value, color, ypos in [
            ("median", p50, "C0", 0.97),
            ("p90", p90, "C2", 0.85),
            ("p99", p99, "C3", 0.73),
        ]:
            target = ax_lo if value <= bulk_cap else ax_hi
            trans = trans_lo if target is ax_lo else trans_hi
            target.axvline(value, color=color, linestyle="--",
                           linewidth=0.7, alpha=0.85)
            target.text(value, ypos, f"{label}={value:.2f}",
                        transform=trans, va="top", ha="left", fontsize=6.0,
                        color=color,
                        bbox=dict(facecolor="white", edgecolor="none",
                                   alpha=0.7, pad=0.5))
        # 1.0 reference if it falls in the lo region.
        if 1.0 <= bulk_cap:
            ax_lo.axvline(1.0, color="0.7", linestyle=":", linewidth=0.5)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = out / "fig3_tool_llm_ratio_turn.pdf"
    fig.savefig(path)
    plt.close(fig)
    # Companion CSV: per-turn values + summary stats.
    rows = turn_ratio_rows(sessions)
    _write_csv_with_stats(
        out / "fig3_tool_llm_ratio_turn.csv",
        header=["session_id", "step", "tool_wall_s", "llm_wall_s", "ratio"],
        rows=[(sid, step, f"{tw:.4f}", f"{lw:.4f}", f"{r:.6f}")
              for sid, step, tw, lw, r in rows],
        stats_values=[r for *_, r in rows],
        stats_label="ratio",
    )
    return path


def plot_ratio_per_tool(sessions: dict[str, Session], out: Path,
                          exclude_tools: set[str] | None = None) -> Path:
    by_tool = per_tool_ratio_distribution(sessions, exclude=exclude_tools)
    if not by_tool:
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.text(0.5, 0.5, "no tool/llm pairs",
                ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        path = out / "fig4_tool_llm_ratio_tool.pdf"
        fig.savefig(path)
        plt.close(fig)
        return path

    items = sorted(by_tool.items(), key=lambda kv: -np.median(kv[1]))
    names = [k for k, _ in items]
    data = [np.asarray(v, dtype=float) for _, v in items]
    medians = np.array([float(np.median(d)) for d in data])
    outlier = _detect_outlier_index(medians, threshold=3.0)

    def _draw_boxes(ax):
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

    if outlier is None:
        fig, ax = plt.subplots(figsize=(3.8, 2.7))
        _draw_boxes(ax)
        ax.set_ylabel("Tool duration / Turn LLM wall")
        ax.axhline(1.0, color="0.7", linestyle=":", linewidth=0.6)
        ax.set_ylim(bottom=0)
        _annotate_exclusion(ax, exclude_tools)
        fig.tight_layout()
        fig.subplots_adjust(left=0.18)
    else:
        # Broken y-axis: linear on both halves, `//` marks at the seam.
        # Top covers the outlier's whisker range, bottom covers everyone
        # else (with a comfy headroom).
        outlier_data = data[outlier]
        q1, q3 = np.percentile(outlier_data, [25, 75])
        iqr = q3 - q1
        # whisker bounds (matplotlib default = 1.5*IQR, clipped to actual data)
        whisker_lo = max(float(outlier_data.min()),
                         float(q1 - 1.5 * iqr))
        whisker_hi = min(float(outlier_data.max()),
                         float(q3 + 1.5 * iqr))
        non_outlier_data = np.concatenate(
            [d for j, d in enumerate(data) if j != outlier]
        ) if any(d.size for j, d in enumerate(data) if j != outlier) else np.array([0.0])
        bot_hi = float(np.percentile(non_outlier_data, 99)) * 1.15
        if whisker_lo <= bot_hi:
            # outlier's lower whisker overlaps the bulk; no break needed
            fig, ax = plt.subplots(figsize=(3.8, 2.7))
            _draw_boxes(ax)
            ax.set_ylabel("Tool duration / Turn LLM wall")
            ax.axhline(1.0, color="0.7", linestyle=":", linewidth=0.6)
            ax.set_ylim(bottom=0)
            _annotate_exclusion(ax, exclude_tools)
            fig.tight_layout()
            fig.subplots_adjust(left=0.18)
        else:
            fig, (ax_hi, ax_lo) = plt.subplots(
                2, 1, figsize=(3.8, 3.1), sharex=True,
                gridspec_kw={"height_ratios": [1, 3]},
            )
            _draw_boxes(ax_hi)
            _draw_boxes(ax_lo)
            ax_hi.set_ylim(whisker_lo * 0.95, whisker_hi * 1.05)
            ax_lo.set_ylim(0, bot_hi)
            ax_hi.tick_params(axis="x", bottom=False, labelbottom=False)
            ax_lo.axhline(1.0, color="0.7", linestyle=":", linewidth=0.6)
            _draw_break_marks(ax_lo, ax_hi, orient="y")
            _annotate_exclusion(ax_hi, exclude_tools)
            # Shared y-label centered across both subplots.
            fig.text(0.02, 0.5, "Tool duration / Turn LLM wall",
                     ha="center", va="center", rotation="vertical", fontsize=9)
            fig.tight_layout(rect=(0.05, 0, 1, 1))
            fig.subplots_adjust(hspace=0.08, left=0.18)
    path = out / "fig4_tool_llm_ratio_tool.pdf"
    fig.savefig(path)
    plt.close(fig)
    # Companion CSV: per-tool-call ratios (no aggregated stats here --
    # the boxplot already shows per-tool distribution shape).
    rows = per_tool_ratio_rows(sessions, exclude=exclude_tools)
    _write_csv_with_stats(
        out / "fig4_tool_llm_ratio_tool.csv",
        header=["session_id", "step", "tool", "ratio"],
        rows=[(sid, step, name, f"{r:.6f}")
              for sid, step, name, r in rows],
        stats_values=None,
    )
    return path


def plot_tool_tokens(sessions: dict[str, Session], out: Path,
                       exclude_tools: set[str] | None = None) -> Path:
    pairs = tool_token_pairs(sessions, exclude=exclude_tools)
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
        _annotate_exclusion(ax_out, exclude_tools)

    fig.tight_layout()
    path = out / "fig5_tool_tokens.pdf"
    fig.savefig(path)
    plt.close(fig)

    # ----- companion mean/std table for fig5 -----
    _write_tool_tokens_table(by_tool_out, by_tool_in, out)
    return path


def _collect_turn_decomposition(sessions: dict[str, Session]):
    """Per-turn (session_id, step, duration_s, llm_wall_s, tool_wall_s,
    post_overhead_s, llm_wall_true_s, others_true_s).

    Turns that fired the `task` tool are EXCLUDED wholesale: `task` spawns
    a nested agent session (its own LLM loop + tools) whose wall time lands
    inside this turn's tool_wall_s, so such a turn is really downstream LLM
    work, not the ordinary "one LLM step + local tools" shape this
    decomposition characterizes. Task turns are analyzed separately.

    The last two fields are the WALL-CLOCK-ANCHORED correction to the
    stream-event llm_wall_s. The profiler's llm_wall_s = start-step -> first
    tool/last text, but `start-step` fires only when opencode begins
    CONSUMING the response, which for a large buffered tool-call arrives at
    the END of generation -- so the real LLM time lands in the
    turn.start -> llm.start gap and gets misattributed to post_overhead
    ("others"). llm_wall_true_s = (llm.end.ts - turn.start.ts) - tool_wall
    recovers the true client-observed LLM wall (request send -> response
    fully received, incl. any server queue/prefill), and others_true_s =
    duration - llm_true - tool_wall is then just the finish-step ->
    turn.end framework tail. Both are None when turn.start/llm.end
    timestamps are missing (fall back to the stream-based split).

    Skips turns where the three turn.end component fields aren't all
    present (post_overhead_s is reconstructed from the others when only
    it is missing -- older patches didn't emit it on turn.end)."""
    rows = []
    for sid, s in sorted(sessions.items()):
        for step, t in sorted(s.turns.items()):
            if any(tc.name == TASK_TOOL_NAME for tc in t.tools):
                continue
            d, lw, tw = t.turn_duration_s, t.llm_wall_s, t.tool_wall_s
            if d is None or lw is None or tw is None:
                continue
            po = t.post_overhead_s
            if po is None:
                po = max(0.0, d - lw - tw)
            llm_true = others_true = None
            if t.turn_start_ts is not None and t.llm_end_ts is not None:
                llm_true = max(0.0, t.llm_end_ts - t.turn_start_ts - tw)
                others_true = max(0.0, d - llm_true - tw)
            rows.append((sid, step, d, lw, tw, po, llm_true, others_true))
    return rows


_SHARE_COLORS = {"llm_wall_s": "C0", "tool_wall_s": "C2", "others": "0.5"}


def _smooth_cdf(xs, xmax: float):
    """Monotone-smooth empirical CDF for a sorted share sample. Returns (x, y) on
    a dense grid via a shape-preserving PCHIP spline (no overshoot past [0,1],
    stays non-decreasing); falls back to the piecewise-linear CDF when scipy is
    absent or there are too few points. Anchored at (0,0) and extended flat at
    y=1.0 out to xmax so a curve that reached 1.0 runs to the right edge."""
    n = xs.size
    ux = np.unique(xs)                                   # strictly increasing knots
    uy = np.searchsorted(xs, ux, side="right") / n       # true CDF value at each knot
    x_pts, y_pts = ux.tolist(), uy.tolist()
    if x_pts[0] > 0.0:
        x_pts, y_pts = [0.0] + x_pts, [0.0] + y_pts
    if x_pts[-1] < xmax:
        x_pts, y_pts = x_pts + [xmax], y_pts + [1.0]
    x_pts, y_pts = np.asarray(x_pts, dtype=float), np.asarray(y_pts, dtype=float)
    if x_pts.size >= 3:
        try:
            from scipy.interpolate import PchipInterpolator
            f = PchipInterpolator(x_pts, y_pts)
            gx = np.linspace(float(x_pts[0]), float(x_pts[-1]), 400)
            return gx, np.clip(f(gx), 0.0, 1.0)
        except Exception:
            pass
    return x_pts, y_pts


def _fig_turn_share_dist(share_arrays: dict, share_components, path: Path) -> Path:
    """Empirical CDF of each component's per-turn share of duration (llm / tool /
    others), overlaid in one graph. y = fraction of turns with share <= x, so a
    curve hugging the right edge means that component dominates most turns.
    Shares are in [0,1] for the anchored split; parallel-tool edge cases can push
    tool share past 1, so the x-axis extends to the observed max."""
    # pre-pass: collect sorted shares + the global x-extent so every curve can be
    # drawn flat to the right edge once it reaches 1.0.
    series = []
    xmax = 1.0
    for name in share_components:
        v = share_arrays.get(name)
        if v is None or v.size == 0:
            continue
        xs = np.sort(np.asarray(v, dtype=float))
        xmax = max(xmax, float(xs[-1]))
        series.append((name, xs))
    if not series:
        return path

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    for name, xs in series:
        gx, gy = _smooth_cdf(xs, xmax)
        ax.plot(gx, gy, color=_SHARE_COLORS.get(name, "C3"), linewidth=1.6,
                label=name, solid_joinstyle="round", solid_capstyle="round",
                antialiased=True)
    ax.axhline(0.5, color="0.7", linewidth=0.6, linestyle=":")  # median reference
    ax.set_xlabel("per-turn share of duration")
    ax.set_ylabel("cumulative fraction of turns")
    ax.set_xlim(0.0, xmax)
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Per-turn share CDF (llm / tool / others)", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_turn_decomposition(sessions: dict[str, Session], out: Path) -> Path | None:
    """Mean/median/p90/p99 stats for per-turn duration / llm_wall /
    tool_wall / others, plus the per-turn share distribution. The LLM
    component is WALL-CLOCK-ANCHORED (llm.end - turn.start - tool_wall),
    which recovers the real LLM time the stream-event llm_wall_s misses on
    buffered large tool-call turns; turns lacking timestamps fall back to
    the stream-based split. The legacy stream-based numbers are printed
    below for reference, and the per-turn CSV carries both
    (llm_wall_s/others_s = stream, llm_wall_true_s/others_true_s = anchored).
    Emits a horizontal stacked bar figure + companion CSVs + stdout tables."""
    rows = _collect_turn_decomposition(sessions)
    if not rows:
        print("\nturn decomposition: no turns with full timing data")
        return None

    arr = np.array([(r[2], r[3], r[4], r[5]) for r in rows], dtype=float)
    dur, llm, tool, post = arr.T

    # Wall-clock-anchored correction (turn.start -> llm.end). Present only on
    # turns that carry both timestamps; NaN elsewhere. See
    # _collect_turn_decomposition for why the stream-based llm_wall_s
    # under-measures buffered large tool-call turns.
    corr = np.array(
        [(r[6] if r[6] is not None else np.nan,
          r[7] if r[7] is not None else np.nan) for r in rows],
        dtype=float,
    )
    llm_true, others_true = corr.T
    n_corr = int(np.count_nonzero(~np.isnan(llm_true)))

    # Promote the wall-clock-anchored values (llm.end - turn.start - tool) to
    # PRIMARY: the stream-event llm_wall_s under-measures buffered large
    # tool-call turns (start-step fires only when opencode starts consuming the
    # response). Fall back to the stream-based split only where turn.start /
    # llm.end timestamps are missing. Both variants satisfy
    # llm + tool + others == duration, so the fallback is seamless and the
    # tool_wall_s / duration_s columns are untouched.
    llm_stream, post_stream = llm.copy(), post.copy()
    llm = np.where(np.isnan(llm_true), llm_stream, llm_true)
    post = np.where(np.isnan(others_true), post_stream, others_true)

    components = [
        ("duration_s", dur),
        ("llm_wall_s", llm),      # wall-clock-anchored (llm.end - turn.start - tool)
        ("tool_wall_s", tool),
        ("others", post),         # residual = finish-step -> turn.end framework tail
    ]
    stats = {name: _summary_stats(vals) for name, vals in components}
    share_components = ("llm_wall_s", "tool_wall_s", "others")

    # Per-turn share distribution (each turn weighted equally -- this is the
    # mean-OF-ratios family, NOT ratio-of-means, so long turns don't dominate).
    # Skip turns with zero duration to avoid div-by-zero.
    safe = dur > 0
    by_name = {"llm_wall_s": llm, "tool_wall_s": tool, "others": post}
    share_arrays = {
        name: (by_name[name][safe] / dur[safe]) if safe.any() else np.array([])
        for name in share_components
    }
    share_stats = {name: _summary_stats(share_arrays[name]) for name in share_components}
    # empty-stats guard so downstream formatting never KeyErrors
    for name in share_components:
        if not share_stats[name]:
            share_stats[name] = {"mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}

    # ----- stdout pretty table -----
    print()
    print(f"Per-turn duration decomposition (n={len(rows)} turns, "
          f"task-tool turns excluded; LLM wall-clock-anchored):")
    hdr = f"{'component':<18} {'mean':>9} {'median':>9} {'p90':>9} {'p99':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, _ in components:
        s = stats[name]
        print(f"{name:<18} {s['mean']:>9.3f} {s['median']:>9.3f} "
              f"{s['p90']:>9.3f} {s['p99']:>9.3f}")
    print()
    print("Per-turn share of duration (per-turn ratios, task-tool turns excluded):")
    shdr = f"{'component':<18} {'mean':>9} {'median':>9} {'p90':>9} {'p99':>9}"
    print(shdr)
    print("-" * len(shdr))
    for name in share_components:
        s = share_stats[name]
        print(f"{name:<18} {s['mean']:>9.1%} {s['median']:>9.1%} "
              f"{s['p90']:>9.1%} {s['p99']:>9.1%}")

    # ----- legacy stream-event split (start-step anchor), for reference -----
    # The primary tables above are now wall-clock-anchored; show the old
    # stream-based llm/others alongside so the correction's effect is visible.
    if n_corr < len(rows):
        print(f"\n  (wall-clock-anchored on {n_corr}/{len(rows)} turns; the rest "
              f"fell back to the stream-based split)")
    if n_corr:
        mask = ~np.isnan(llm_true)
        moved = float(np.mean(llm_true[mask] - llm_stream[mask]))
        print()
        print("Legacy stream-event decomposition (start-step anchor, reference):")
        lhdr = f"{'component':<18} {'mean':>9} {'median':>9} {'p90':>9} {'p99':>9}"
        print(lhdr)
        print("-" * len(lhdr))
        for name, vals in (("llm_wall_s(stream)", llm_stream),
                           ("others(stream)", post_stream)):
            s = _summary_stats(vals)
            print(f"{name:<18} {s['mean']:>9.3f} {s['median']:>9.3f} "
                  f"{s['p90']:>9.3f} {s['p99']:>9.3f}")
        print(f"  mean LLM time the stream split misattributed to 'others': "
              f"{moved:+.3f}s/turn (now folded back into llm_wall_s)")

    # ----- CSV -----
    import csv
    csv_path = out / "fig6_turn_decomposition_stats.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "n_turns", "mean_s", "median_s", "p90_s", "p99_s"])
        for name, _ in components:
            s = stats[name]
            w.writerow([
                name, len(rows),
                f"{s['mean']:.4f}", f"{s['median']:.4f}",
                f"{s['p90']:.4f}", f"{s['p99']:.4f}",
            ])
        f.write("\n# per-turn share of duration (distribution of per-turn ratios,"
                " each turn weighted equally; turns that fired the task tool"
                " are excluded)\n")
        w.writerow(["component", "n_turns",
                    "mean_ratio", "median_ratio", "p90_ratio", "p99_ratio"])
        for name in share_components:
            s = share_stats[name]
            w.writerow([
                name, len(rows),
                f"{s['mean']:.4f}", f"{s['median']:.4f}",
                f"{s['p90']:.4f}", f"{s['p99']:.4f}",
            ])

    # ----- per-turn RAW rows CSV (one row per turn) -----
    # Long-form per-turn dump (task-tool turns already excluded). The
    # latency-composition views are computed in-process by
    # analyze_latency_composition(); this CSV is the portable long-form export.
    per_turn_path = out / "fig6_turn_decomposition_per_turn.csv"
    with per_turn_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "step", "duration_s",
                    "llm_wall_s", "tool_wall_s", "others_s",
                    "llm_wall_true_s", "others_true_s"])
        for sid, step, d, lw, tw, po, llm_t, oth_t in rows:
            w.writerow([sid, step,
                        f"{d:.6f}", f"{lw:.6f}", f"{tw:.6f}", f"{po:.6f}",
                        "" if llm_t is None else f"{llm_t:.6f}",
                        "" if oth_t is None else f"{oth_t:.6f}"])

    # ----- figure: horizontal stacked single bar showing mean composition -----
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    pieces = [
        ("llm_wall",       float(llm.mean()),  "C0"),
        ("tool_wall",      float(tool.mean()), "C2"),
        ("others",         float(post.mean()), "0.6"),
    ]
    mean_dur = float(dur.mean()) or 1.0
    left = 0.0
    for label, value, color in pieces:
        ax.barh(0, value, left=left, color=color, edgecolor="black", linewidth=0.5,
                label=f"{label}: {value:.2f}s ({value/mean_dur:.0%})")
        left += value
    ax.set_yticks([])
    ax.set_xlabel("Mean turn duration (s)")
    ax.set_xlim(left=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5),
              ncol=3, frameon=False, fontsize=7, handlelength=1.0,
              columnspacing=0.8, handletextpad=0.3)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.45)
    path = out / "fig6_turn_decomposition.pdf"
    fig.savefig(path)
    plt.close(fig)

    # companion: the per-turn share of each component overlaid in one graph
    _fig_turn_share_dist(share_arrays, share_components,
                         out / "fig6b_turn_share_distribution.pdf")
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


def _load_trace_repo_map(trace_path: Path):
    """Parse a run's trace.jsonl into session_id -> dict(instance_id, repo,
    arrival_offset_s, rtt_s). repo is derived from the instance_id
    (`org__repo-<num>` -> `org/repo`) as a workspace-SIZE proxy -- the snapshot
    cost driver. Returns {} on any failure (the decomposition still runs without
    repo/concurrency breakdowns)."""
    out: dict[str, dict] = {}
    try:
        with trace_path.open(encoding="utf-8", errors="replace") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = rec.get("session_id")
                iid = rec.get("instance_id")
                if not sid:
                    continue
                repo = None
                if iid and "-" in iid:
                    repo = iid.rsplit("-", 1)[0].replace("__", "/")
                out[sid] = {
                    "instance_id": iid,
                    "repo": repo,
                    "arrival_offset_s": rec.get("arrival_offset_s"),
                    "rtt_s": rec.get("rtt_s"),
                }
    except OSError:
        return {}
    return out


def _session_concurrency(trace_map: dict[str, dict]) -> dict[str, int]:
    """For each session, how many OTHER sessions were in flight at the same
    time (overlap of [arrival, arrival+rtt] windows). A proxy for disk-IO
    contention during the per-step git snapshot. Sessions missing arrival/rtt
    are skipped."""
    windows = []
    for sid, m in trace_map.items():
        a, r = m.get("arrival_offset_s"), m.get("rtt_s")
        if a is None or r is None:
            continue
        windows.append((sid, float(a), float(a) + float(r)))
    conc: dict[str, int] = {}
    for sid, a0, b0 in windows:
        n = 0
        for sid2, a1, b1 in windows:
            if sid2 == sid:
                continue
            if a1 < b0 and a0 < b1:  # overlap
                n += 1
        conc[sid] = n
    return conc


def analyze_post_overhead(sessions: dict[str, Session], out: Path,
                          trace_path: Path | None = None,
                          tail_quantile: float | None = None,
                          min_duration_s: float | None = None) -> Path | None:
    """Drill INTO turn.post_overhead_s (the duration residual = duration -
    llm_wall - tool_wall, labelled "others" in fig6). Splits it per turn into:
      explained   = post_stream_overhead_s  (snapshot.track+patch + DB write)
      unexplained = post_overhead_s - explained  (pre-turn setup + inter-step
                    gaps + LLM-tail leak when streamFinish didn't fire)
    and reports the streamFinish-miss rate (turns where the LLM stream tail
    leaked into overhead). With --trace, also breaks down by repo (size proxy)
    and by concurrent in-flight sessions (IO-contention proxy).

    task-tool turns are EXCLUDED (same rationale as fig6: a `task` turn's wall
    is a nested agent loop, not this turn's own LLM+finalization shape), so the
    turn population here matches fig6's per-turn decomposition.

    tail_quantile / min_duration_s restrict the drill-down to the LONGEST turns
    (turn_duration_s strictly above the threshold). Pass tail_quantile=0.99 to
    interrogate exactly the >p99 bucket -- i.e. to test whether the tail's
    inflated "others" is real finalization overhead (large explained slice) or
    leaked LLM stream tail (high streamFinish-miss rate + unexplained slice)."""

    def _is_task_turn(t: Turn) -> bool:
        return any(tc.name == TASK_TOOL_NAME for tc in t.tools)

    # Threshold from the SAME (task-excluded) turn population so tail_quantile
    # lines up with fig6's >pQ bucket. Uses linear-interpolation percentile
    # (numpy default), matching analyze_latency_composition's np.quantile buckets.
    dur_floor = min_duration_s
    if tail_quantile is not None:
        durs = [t.turn_duration_s for s in sessions.values()
                for t in s.turns.values()
                if t.turn_duration_s is not None and not _is_task_turn(t)]
        if durs:
            q_thr = float(np.percentile(np.array(durs, dtype=float),
                                        tail_quantile * 100.0))
            dur_floor = q_thr if dur_floor is None else max(dur_floor, q_thr)

    rows = []  # (sid, step, post, explained_or_None, finish_fired_or_None)
    for sid, s in sorted(sessions.items()):
        for step, t in sorted(s.turns.items()):
            if _is_task_turn(t):
                continue
            if dur_floor is not None:
                if t.turn_duration_s is None or t.turn_duration_s <= dur_floor:
                    continue
            po = t.post_overhead_s
            if po is None:
                d, lw, tw = t.turn_duration_s, t.llm_wall_s, t.tool_wall_s
                if d is None or lw is None or tw is None:
                    continue
                po = max(0.0, d - lw - tw)
            rows.append((sid, step, float(po), t.post_stream_overhead_s,
                         t.stream_finish_fired))
    subset = "" if dur_floor is None else f" with duration > {dur_floor:.2f}s"
    if not rows:
        print(f"\npost_overhead decomposition: no turns with timing data{subset}")
        return None

    po_all = np.array([r[2] for r in rows], dtype=float)
    # Turns that DO have a post_stream measurement (current patch + streamFinish fired).
    paired = [(po, ps) for _, _, po, ps, _ in rows if ps is not None]
    have_instr = any(r[3] is not None or r[4] is not None for r in rows)

    print()
    print(f"post_overhead decomposition (n={len(rows)} turns{subset}):")
    print(f"  post_overhead_s        mean={po_all.mean():.3f}  "
          f"median={np.median(po_all):.3f}  p90={np.percentile(po_all,90):.3f}  "
          f"p99={np.percentile(po_all,99):.3f}")

    explained_arr = unexplained_arr = None
    if paired:
        po_p = np.array([p for p, _ in paired], dtype=float)
        ps_p = np.array([s for _, s in paired], dtype=float)
        explained_arr = ps_p
        unexplained_arr = np.maximum(0.0, po_p - ps_p)
        # share of post_overhead that the snapshot+DB slice accounts for,
        # averaged per-turn (skip po==0 to avoid div-by-zero).
        safe = po_p > 1e-9
        share = float((ps_p[safe] / po_p[safe]).mean()) if safe.any() else 0.0
        print(f"  explained (snap+DB)    mean={ps_p.mean():.3f}  "
              f"median={np.median(ps_p):.3f}  p90={np.percentile(ps_p,90):.3f}  "
              f"p99={np.percentile(ps_p,99):.3f}   (n={len(paired)} paired turns)")
        print(f"  unexplained (rest)     mean={unexplained_arr.mean():.3f}  "
              f"median={np.median(unexplained_arr):.3f}  "
              f"p90={np.percentile(unexplained_arr,90):.3f}  "
              f"p99={np.percentile(unexplained_arr,99):.3f}")
        print(f"  -> snapshot+DB explains {share:.0%} of post_overhead "
              f"(mean per-turn share); the rest is pre-turn setup / step gaps / "
              f"LLM-tail leak")
    elif not have_instr:
        print("  (post_stream_overhead_s absent -> profile NDJSON predates the "
              "patch field; rerun with the current opencode-profile.patch to "
              "split snapshot+DB from the rest)")

    # streamFinish-miss diagnostic: turns where the AI SDK finish event didn't
    # fire => post_stream is null AND the LLM tail leaked into post_overhead.
    fired = [r for r in rows if r[4] is True]
    missed = [r for r in rows if r[4] is False]
    if fired or missed:
        nf, nm = len(fired), len(missed)
        tot = nf + nm
        print()
        print(f"streamFinish fired: {nf}/{tot} turns "
              f"({nf/tot:.0%}); missed: {nm} ({nm/tot:.0%})")
        if missed:
            po_missed = np.array([r[2] for r in missed], dtype=float)
            po_fired = np.array([r[2] for r in fired], dtype=float) if fired else np.array([0.0])
            print(f"  post_overhead on MISSED turns  median={np.median(po_missed):.3f} "
                  f"(these include leaked LLM stream tail)")
            print(f"  post_overhead on FIRED turns   median={np.median(po_fired):.3f}")

    # ----- per-turn CSV dump (for external correlation) -----
    import csv
    trace_map = _load_trace_repo_map(trace_path) if trace_path else {}
    conc = _session_concurrency(trace_map) if trace_map else {}
    per_turn_csv = out / ("post_overhead_per_turn.csv" if dur_floor is None
                          else "post_overhead_per_turn_tail.csv")
    with per_turn_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "step", "instance_id", "repo",
                    "in_flight_concurrency", "post_overhead_s",
                    "post_stream_overhead_s", "unexplained_s", "stream_finish_fired"])
        for sid, step, po, ps, ff in rows:
            meta = trace_map.get(sid, {})
            unexp = max(0.0, po - ps) if ps is not None else ""
            w.writerow([sid, step, meta.get("instance_id", ""),
                        meta.get("repo", ""), conc.get(sid, ""),
                        f"{po:.4f}", "" if ps is None else f"{ps:.4f}",
                        "" if unexp == "" else f"{unexp:.4f}",
                        "" if ff is None else int(ff)])

    # ----- per-repo breakdown (size proxy) -----
    if trace_map:
        by_repo: dict[str, list[float]] = {}
        for sid, step, po, ps, ff in rows:
            repo = trace_map.get(sid, {}).get("repo")
            if repo:
                by_repo.setdefault(repo, []).append(po)
        if by_repo:
            print()
            print("post_overhead by repo (size proxy; sorted by median desc):")
            print(f"  {'repo':<28} {'n':>5} {'median':>8} {'p90':>8}")
            for repo, vals in sorted(by_repo.items(),
                                     key=lambda kv: -float(np.median(kv[1]))):
                a = np.array(vals, dtype=float)
                print(f"  {repo:<28} {len(a):>5} {np.median(a):>8.3f} "
                      f"{np.percentile(a,90):>8.3f}")

        # ----- concurrency correlation (IO-contention proxy) -----
        if conc:
            xs, ys = [], []
            # per-session mean post_overhead vs that session's in-flight count
            sess_po: dict[str, list[float]] = {}
            for sid, step, po, ps, ff in rows:
                sess_po.setdefault(sid, []).append(po)
            for sid, vals in sess_po.items():
                if sid in conc:
                    xs.append(conc[sid])
                    ys.append(float(np.mean(vals)))
            if len(xs) >= 3 and len(set(xs)) > 1:
                r = float(np.corrcoef(xs, ys)[0, 1])
                print()
                print(f"concurrency vs post_overhead: Pearson r = {r:+.3f} "
                      f"(n={len(xs)} sessions; +ve => IO contention inflates overhead)")

    print(f"  (per-turn csv: {per_turn_csv})")

    # ----- figure: post_overhead = explained + unexplained stacked bar -----
    if explained_arr is not None:
        fig, ax = plt.subplots(figsize=(3.5, 1.5))
        pieces = [
            ("snapshot+DB", float(explained_arr.mean()), "C1"),
            ("rest (setup/gaps/tail)", float(unexplained_arr.mean()), "0.6"),
        ]
        total = float(po_all.mean()) or 1.0
        left = 0.0
        for label, value, color in pieces:
            ax.barh(0, value, left=left, color=color, edgecolor="black",
                    linewidth=0.5, label=f"{label}: {value:.2f}s ({value/total:.0%})")
            left += value
        ax.set_yticks([])
        ax.set_xlabel("Mean post_overhead (s)")
        ax.set_xlim(left=0)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=1,
                  frameon=False, fontsize=7, handlelength=1.0,
                  columnspacing=0.8, handletextpad=0.3)
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.55)
        path = out / ("fig7_post_overhead_breakdown.pdf" if dur_floor is None
                      else "fig7_post_overhead_breakdown_tail.pdf")
        fig.savefig(path)
        plt.close(fig)
        return path
    return None


# ---------- entry point ----------


# Total-latency buckets, split at the p50/p90/p99 of the per-turn duration.
# Comparison is `<=` on the lower edge so ties fall in the lower bucket.
_LAT_BUCKETS = ["<=p50", "p50-p90", "p90-p99", ">p99"]


def _lat_fig_violin(per_req: dict, comps, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * len(comps), 3.2))
    for i, (name, _vals, color) in enumerate(comps):
        pos = i + 1
        v = per_req[name]
        # gaussian_kde (inside violinplot) is singular on zero-variance samples
        # (e.g. tool share ~0 everywhere); fall back to a median marker.
        if v.size >= 2 and float(np.var(v)) > 1e-12:
            parts = ax.violinplot([v], positions=[pos], showmedians=True, showextrema=True)
            parts["bodies"][0].set_facecolor(color)
            parts["bodies"][0].set_alpha(0.7)
            parts["bodies"][0].set_edgecolor("black")
            parts["bodies"][0].set_linewidth(0.5)
            for k in ("cbars", "cmins", "cmaxes", "cmedians"):
                if k in parts:
                    parts[k].set_color("black")
                    parts[k].set_linewidth(0.8)
        else:
            med = float(np.median(v)) if v.size else 0.0
            ax.scatter([pos], [med], color=color, edgecolor="black", zorder=3, s=30)
            ax.hlines(med, pos - 0.25, pos + 0.25, color="black", linewidth=0.8)
    ax.set_xticks(range(1, len(comps) + 1))
    ax.set_xticklabels([n for n, _, _ in comps], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("per-request share of total latency")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Per-request latency-component share", fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _lat_fig_sorted_stacked(dur, comps, path: Path) -> Path:
    order = dur.argsort()
    x = np.arange(len(order))
    bottom = np.zeros(len(order))
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for name, vals, color in comps:
        v = vals[order]
        ax.bar(x, v, bottom=bottom, width=1.0, linewidth=0, color=color, label=name)
        bottom += v
    ax.set_xlabel("turn (sorted by total latency, ascending)")
    ax.set_ylabel("latency (s)")
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(bottom=0)
    ax.set_title("Latency composition vs magnitude", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=len(comps), loc="upper left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _lat_fig_bucket_stacked(cond_rows, comps, path: Path) -> Path:
    present = [r for r in cond_rows if r["n"] > 0]
    x = np.arange(len(present))
    bottom = np.zeros(len(present))
    fig, ax = plt.subplots(figsize=(1.8 + 1.0 * len(present), 3.4))
    for name, _vals, color in comps:
        vals = np.array([r[name] for r in present], dtype=float)
        ax.bar(x, vals, bottom=bottom, width=0.7, color=color,
               edgecolor="black", linewidth=0.4, label=name)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['bucket']}\n(n={r['n']})" for r in present], fontsize=8)
    ax.set_ylabel("mean per-request share")
    ax.set_ylim(0, 1.02)
    ax.set_title("Composition by total-latency bucket", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=len(comps),
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def analyze_latency_composition(sessions: dict[str, Session], out: Path) -> Path | None:
    """Per-request (per-turn) latency-composition analysis on the WALL-CLOCK-
    anchored components (llm = llm.end - turn.start - tool, with the stream-based
    split as fallback where timestamps are missing; task-tool turns excluded via
    _collect_turn_decomposition). Merged in from the former standalone
    analyze_latency_breakdown.py so all profile analysis lives in one script and
    shares the corrected LLM timing.

    Emits into `out`:
      latency_pooled_share.csv          -- (1) pooled/time-weighted share (Σcomp/Σdur)
      latency_per_request_share.csv     -- (2) per-request share distribution
                                               (mean/p50/p90/p99/p25/p75/min/max)
      latency_conditional_by_bucket.csv -- (3) mean per-request share per component,
                                               conditioned on the total-latency bucket
                                               split at the p50/p90/p99 of duration
      latency_share_violin.pdf          -- per-component per-request share distribution
      latency_sorted_stacked.pdf        -- turns sorted by total latency, stacked seconds
      latency_bucket_stacked.pdf        -- mean share per component per bucket
    Returns `out` (or None when there are no turns)."""
    rows = _collect_turn_decomposition(sessions)
    if not rows:
        print("\nlatency composition: no turns with full timing data")
        return None

    dur = np.array([r[2] for r in rows], dtype=float)
    llm_s = np.array([r[3] for r in rows], dtype=float)
    tool = np.array([r[4] for r in rows], dtype=float)
    post_s = np.array([r[5] for r in rows], dtype=float)
    llm_t = np.array([r[6] if r[6] is not None else np.nan for r in rows], dtype=float)
    oth_t = np.array([r[7] if r[7] is not None else np.nan for r in rows], dtype=float)
    llm = np.where(np.isnan(llm_t), llm_s, llm_t)       # anchored, stream fallback
    others = np.where(np.isnan(oth_t), post_s, oth_t)

    comps = [("llm_wall_s", llm, "C0"), ("tool_wall_s", tool, "C2"),
             ("others", others, "0.6")]
    safe = dur > 0

    def _pct(v, q):
        return float(np.percentile(v, q)) if v.size else float("nan")

    import csv
    # ----- (1) pooled / time-weighted share -----
    grand = float(dur.sum())
    pooled = {}
    with (out / "latency_pooled_share.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "total_seconds", "pooled_share"])
        for name, vals, _ in comps:
            s = float(vals.sum())
            pooled[name] = (s / grand) if grand > 0 else float("nan")
            w.writerow([name, f"{s:.4f}", f"{pooled[name]:.4f}"])
        w.writerow(["TOTAL", f"{grand:.4f}", "1.0000" if grand > 0 else ""])

    # ----- (2) per-request share distribution (each turn weighted equally) -----
    per_req = {}
    with (out / "latency_per_request_share.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["component", "n_requests", "mean", "p50", "p90", "p99",
                    "p25", "p75", "min", "max"])
        for name, vals, _ in comps:
            sh = vals[safe] / dur[safe]
            per_req[name] = sh
            w.writerow([
                name, sh.size,
                f"{float(np.mean(sh)):.4f}" if sh.size else "",
                f"{_pct(sh, 50):.4f}", f"{_pct(sh, 90):.4f}", f"{_pct(sh, 99):.4f}",
                f"{_pct(sh, 25):.4f}", f"{_pct(sh, 75):.4f}",
                f"{float(sh.min()):.4f}" if sh.size else "",
                f"{float(sh.max()):.4f}" if sh.size else "",
            ])

    # ----- (3) conditional by total-latency bucket -----
    p50, p90, p99 = (float(np.quantile(dur, q)) for q in (0.5, 0.9, 0.99))

    def _bucket(t):
        if t <= p50:
            return 0
        if t <= p90:
            return 1
        if t <= p99:
            return 2
        return 3

    bidx = np.array([_bucket(t) for t in dur])
    cond_rows = []
    with (out / "latency_conditional_by_bucket.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bucket", "n_requests", "mean_total_s"]
                   + [f"{n}_mean_share" for n, _, _ in comps])
        for bi, blabel in enumerate(_LAT_BUCKETS):
            m = (bidx == bi) & safe
            n = int(m.sum())
            row = {"bucket": blabel, "n": n,
                   "mean_total": float(dur[m].mean()) if n else float("nan")}
            shares = []
            for name, vals, _ in comps:
                sh = float((vals[m] / dur[m]).mean()) if n else float("nan")
                shares.append(sh)
                row[name] = sh
            cond_rows.append(row)
            w.writerow([blabel, n, f"{row['mean_total']:.4f}"]
                       + [f"{x:.4f}" for x in shares])

    # ----- stdout summary -----
    print()
    print(f"Latency composition (n={dur.size} turns, wall-clock-anchored, "
          f"task-tool turns excluded):")
    print(f"  total-latency thresholds: p50={p50:.3f}s  p90={p90:.3f}s  p99={p99:.3f}s")
    print("  (1) pooled / time-weighted share:")
    for name, vals, _ in comps:
        print(f"      {name:<12} {pooled[name]:>7.2%}  ({float(vals.sum()):.1f}s)")
    print("  (2) per-request share (each turn weighted equally):")
    print(f"      {'component':<12} {'mean':>7} {'p50':>7} {'p90':>7} {'p99':>7}")
    for name, _vals, _ in comps:
        sh = per_req[name]
        print(f"      {name:<12} {float(np.mean(sh)) if sh.size else float('nan'):>7.1%} "
              f"{_pct(sh, 50):>7.1%} {_pct(sh, 90):>7.1%} {_pct(sh, 99):>7.1%}")
    print("  (3) mean share by total-latency bucket:")
    print("      " + f"{'bucket':<10} {'n':>5} {'mean_tot':>9}  "
          + " ".join(f"{n:>12}" for n, _, _ in comps))
    for r in cond_rows:
        if r["n"] == 0:
            continue
        print("      " + f"{r['bucket']:<10} {r['n']:>5} {r['mean_total']:>9.2f}  "
              + " ".join(f"{r[n]:>12.1%}" for n, _, _ in comps))

    # ----- figures -----
    _lat_fig_violin(per_req, comps, out / "latency_share_violin.pdf")
    _lat_fig_sorted_stacked(dur, comps, out / "latency_sorted_stacked.pdf")
    _lat_fig_bucket_stacked(cond_rows, comps, out / "latency_bucket_stacked.pdf")
    return out


def _fmt_tok(n) -> str:
    return "-" if n is None else str(int(n))


def _extract_cmd(args_head: str | None) -> str:
    """Pull the human-meaningful command out of a tool.start args_head preview
    (JSON.stringify(args) truncated to 200 chars). Returns the bash command /
    file path / pattern when the preview parses; else the raw (truncated)
    preview so the start of a long command is still visible."""
    if not args_head:
        return ""
    try:
        d = json.loads(args_head)
        if isinstance(d, dict):
            for k in ("command", "cmd", "filePath", "path", "pattern", "query"):
                v = d.get(k)
                if isinstance(v, str) and v:
                    return v
    except Exception:
        pass
    return args_head


def analyze_tool_dominated_turns(sessions: dict[str, Session], out: Path,
                                 top_n: int = 20,
                                 trace_path: Path | None = None) -> Path | None:
    """Find turns where tool_wall_s dominates the turn wall (largest
    tool_wall/duration) and dump their LLM token usage + tool-type breakdown.
    task-tool turns are excluded. tool_share can exceed 1.0 when a step ran tools
    in PARALLEL (tool_wall is the SUM of per-tool durations, which can beat the
    wall span). Writes tool_dominated_turns.csv (all turns w/ tools, sorted by
    share) + a stdout table of the top N + a dominant-tool rollup."""
    trace_map = _load_trace_repo_map(trace_path) if trace_path else {}
    recs = []
    for sid, s in sorted(sessions.items()):
        for step, t in sorted(s.turns.items()):
            if any(tc.name == TASK_TOOL_NAME for tc in t.tools):
                continue
            d, tw = t.turn_duration_s, t.tool_wall_s
            if d is None or tw is None or d <= 0 or not t.tools:
                continue
            by_tool: dict[str, float] = {}
            out_chars = 0
            for tc in t.tools:
                by_tool[tc.name] = by_tool.get(tc.name, 0.0) + (tc.duration_s or 0.0)
                out_chars += tc.output_chars or 0
            dom_tool, dom_s = max(by_tool.items(), key=lambda kv: kv[1])
            # command of the LONGEST single call of the dominant tool
            dom_longest = max((tc for tc in t.tools if tc.name == dom_tool),
                              key=lambda tc: tc.duration_s or 0.0)
            dom_cmd = _extract_cmd(dom_longest.args_head)
            meta = trace_map.get(sid, {})
            recs.append({
                "sid": sid, "step": step,
                "instance_id": meta.get("instance_id") or "",
                "repo": meta.get("repo") or "",
                "duration_s": d, "tool_wall_s": tw, "tool_share": tw / d,
                "in_tok": t.llm_input_tokens, "out_tok": t.llm_output_tokens,
                "cache_read": t.llm_cache_read, "n_tools": len(t.tools),
                "out_chars": out_chars, "dom_tool": dom_tool, "dom_tool_s": dom_s,
                "dom_cmd": dom_cmd, "by_tool": by_tool,
            })
    if not recs:
        print("\ntool-dominated turns: no turns with tools")
        return None
    recs.sort(key=lambda r: -r["tool_share"])
    top = recs[:top_n]

    print()
    print(f"Top {len(top)} tool-dominated turns (by tool_wall/duration; task "
          f"excluded; share>100% => parallel tools):")
    hdr = (f"{'instance':<24} {'step':>4} {'dur_s':>8} {'tool_s':>8} {'share':>6} "
           f"{'dom_tool':>10} {'dom_s':>7} {'in_tok':>8} {'out_tok':>8} {'cache_r':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in top:
        inst = (r["instance_id"] or r["sid"])[:24]
        print(f"{inst:<24} {r['step']:>4} {r['duration_s']:>8.2f} "
              f"{r['tool_wall_s']:>8.2f} {r['tool_share']:>6.0%} "
              f"{r['dom_tool']:>10} {r['dom_tool_s']:>7.2f} "
              f"{_fmt_tok(r['in_tok']):>8} {_fmt_tok(r['out_tok']):>8} "
              f"{_fmt_tok(r['cache_read']):>8}")
        if r["dom_cmd"]:
            cmd = r["dom_cmd"].replace("\n", "\\n")
            print(f"      └ {cmd[:160]}")

    dom_counts: dict[str, int] = {}
    for r in top:
        dom_counts[r["dom_tool"]] = dom_counts.get(r["dom_tool"], 0) + 1
    roll = ", ".join(f"{k}×{v}" for k, v in
                     sorted(dom_counts.items(), key=lambda kv: -kv[1]))
    print(f"  dominant-tool rollup (top {len(top)}): {roll}")

    import csv
    csv_path = out / "tool_dominated_turns.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "step", "instance_id", "repo", "duration_s",
                    "tool_wall_s", "tool_share", "input_tokens", "output_tokens",
                    "cache_read", "n_tools", "tool_output_chars",
                    "dominant_tool", "dominant_tool_s", "dominant_tool_cmd", "tools"])
        for r in recs:
            tools_str = ";".join(f"{k}:{v:.3f}" for k, v in
                                 sorted(r["by_tool"].items(), key=lambda kv: -kv[1]))
            w.writerow([
                r["sid"], r["step"], r["instance_id"], r["repo"],
                f"{r['duration_s']:.4f}", f"{r['tool_wall_s']:.4f}",
                f"{r['tool_share']:.4f}",
                "" if r["in_tok"] is None else r["in_tok"],
                "" if r["out_tok"] is None else r["out_tok"],
                r["cache_read"], r["n_tools"], r["out_chars"],
                r["dom_tool"], f"{r['dom_tool_s']:.4f}", r["dom_cmd"], tools_str,
            ])
    print(f"  (csv: {csv_path})")
    return csv_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--input", required=True, type=Path,
        help="Aggregated NDJSON file OR directory containing <session>.jsonl files",
    )
    ap.add_argument(
        "--output", required=True, type=Path,
        help="Directory to write fig{1..6}_*.pdf into (created if missing)",
    )
    ap.add_argument(
        "--exclude-tools", nargs="*", default=[], metavar="NAME",
        help="Exclude these tool names from per-tool analyses (fig2 / fig4 / fig5). "
             "Common use: `--exclude-tools task` to drop the sub-agent tool whose "
             "duration is a full nested agent loop, not a leaf-tool call. Note: "
             "fig3 / fig6 use turn.end's pre-aggregated tool_wall_s so they are "
             "unaffected by this flag.",
    )
    ap.add_argument(
        "--trace", type=Path, default=None, metavar="trace.jsonl",
        help="Optional run trace.jsonl. Enables the post_overhead breakdown to "
             "join sessions to instance_id/repo (workspace-size proxy) and to "
             "compute per-session in-flight concurrency (IO-contention proxy).",
    )
    ap.add_argument(
        "--post-overhead-tail-quantile", type=float, default=None, metavar="Q",
        help="Restrict the post_overhead drill-down to turns whose duration "
             "exceeds the Q-quantile of the (task-excluded) turn population, "
             "e.g. 0.99 -> the >p99 bucket. Use to test whether the tail's "
             "inflated 'others' is real finalization overhead or leaked LLM "
             "stream tail (streamFinish miss). Writes *_tail.{csv,pdf}.",
    )
    ap.add_argument(
        "--post-overhead-min-duration", type=float, default=None, metavar="SEC",
        help="Restrict the post_overhead drill-down to turns longer than SEC "
             "seconds (combined with --post-overhead-tail-quantile via max).",
    )
    ap.add_argument(
        "--tool-dominated-top-n", type=int, default=20, metavar="N",
        help="How many top tool-dominated turns (by tool_wall/duration) to print "
             "with their LLM token usage + tool breakdown (default 20).",
    )
    args = ap.parse_args(argv)
    exclude_tools = set(args.exclude_tools)

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

    # Plots that accept the per-tool exclusion vs those that don't.
    per_tool_plots = (plot_tool_exec_time, plot_ratio_per_tool, plot_tool_tokens)
    other_plots = (plot_session_e2e, plot_ratio_per_turn, plot_turn_decomposition)
    for fn in (*other_plots, *per_tool_plots):
        if fn in per_tool_plots:
            path = fn(sessions, args.output, exclude_tools=exclude_tools)
        else:
            path = fn(sessions, args.output)
        if path is not None:
            print(f"  wrote {path}")

    # post_overhead drill-down (step 1): split turn.post_overhead_s into the
    # snapshot+DB finalization slice vs the rest, + streamFinish-miss + repo /
    # concurrency breakdowns when --trace is supplied.
    path = analyze_post_overhead(
        sessions, args.output, trace_path=args.trace,
        tail_quantile=args.post_overhead_tail_quantile,
        min_duration_s=args.post_overhead_min_duration,
    )
    if path is not None:
        print(f"  wrote {path}")

    # latency composition (merged from the former analyze_latency_breakdown.py):
    # per-request pooled / percentile / bucketed share on the wall-clock-anchored
    # components, plus violin / sorted-stacked / bucket-stacked figures.
    path = analyze_latency_composition(sessions, args.output)
    if path is not None:
        print(f"  wrote latency_* CSVs + figures to {path}")

    # tool-dominated turns: where tool_wall_s eats the turn wall -- with LLM
    # token usage + tool-type breakdown to see what drove them.
    path = analyze_tool_dominated_turns(
        sessions, args.output, top_n=args.tool_dominated_top_n,
        trace_path=args.trace,
    )
    if path is not None:
        print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
