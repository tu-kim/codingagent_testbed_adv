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
    post_overhead_s).

    Turns that fired the `task` tool are EXCLUDED wholesale: `task` spawns
    a nested agent session (its own LLM loop + tools) whose wall time lands
    inside this turn's tool_wall_s, so such a turn is really downstream LLM
    work, not the ordinary "one LLM step + local tools" shape this
    decomposition characterizes. Task turns are analyzed separately.

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
            rows.append((sid, step, d, lw, tw, po))
    return rows


def plot_turn_decomposition(sessions: dict[str, Session], out: Path) -> Path | None:
    """Mean/median/p90/p99 stats for turn.end's duration / llm_wall /
    tool_wall / post_overhead, plus the average per-turn share of duration
    each component occupies. Emits a small horizontal stacked bar figure
    + companion CSV + stdout table."""
    rows = _collect_turn_decomposition(sessions)
    if not rows:
        print("\nturn decomposition: no turns with full timing data")
        return None

    arr = np.array([(d, lw, tw, po) for *_, d, lw, tw, po in rows], dtype=float)
    dur, llm, tool, post = arr.T

    components = [
        ("duration_s", dur),
        ("llm_wall_s", llm),
        ("tool_wall_s", tool),
        ("post_overhead_s", post),
    ]
    stats = {name: _summary_stats(vals) for name, vals in components}
    share_components = ("llm_wall_s", "tool_wall_s", "post_overhead_s")

    # Average per-turn share (NOT total ratio -- that would weight long
    # turns more). Skip turns with zero duration to avoid div-by-zero.
    safe = dur > 0
    if safe.any():
        by_name = {"llm_wall_s": llm, "tool_wall_s": tool, "post_overhead_s": post}
        ratio_mean = {
            name: float((by_name[name][safe] / dur[safe]).mean())
            for name in share_components
        }
    else:
        ratio_mean = {name: 0.0 for name in share_components}

    # ----- stdout pretty table -----
    print()
    print(f"Per-turn duration decomposition (n={len(rows)} turns, "
          f"task-tool turns excluded):")
    hdr = f"{'component':<18} {'mean':>9} {'median':>9} {'p90':>9} {'p99':>9}"
    print(hdr)
    print("-" * len(hdr))
    for name, _ in components:
        s = stats[name]
        print(f"{name:<18} {s['mean']:>9.3f} {s['median']:>9.3f} "
              f"{s['p90']:>9.3f} {s['p99']:>9.3f}")
    print()
    print("Average per-turn share of duration (task-tool turns excluded):")
    for name in share_components:
        print(f"  {name:<18} {ratio_mean[name]:>7.2%}")

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
        f.write("\n# average per-turn share of duration (mean of per-turn ratios;"
                " turns that fired the task tool are excluded)\n")
        w.writerow(["component", "mean_ratio"])
        for name in share_components:
            w.writerow([name, f"{ratio_mean[name]:.4f}"])

    # ----- figure: horizontal stacked single bar showing mean composition -----
    fig, ax = plt.subplots(figsize=(3.5, 1.5))
    pieces = [
        ("llm_wall",       float(llm.mean()),  "C0"),
        ("tool_wall",      float(tool.mean()), "C2"),
        ("post_overhead",  float(post.mean()), "0.6"),
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
                          trace_path: Path | None = None) -> Path | None:
    """Drill INTO turn.post_overhead_s (the duration residual = duration -
    llm_wall - tool_wall). Splits it per turn into:
      explained   = post_stream_overhead_s  (snapshot.track+patch + DB write)
      unexplained = post_overhead_s - explained  (pre-turn setup + inter-step
                    gaps + LLM-tail leak when streamFinish didn't fire)
    and reports the streamFinish-miss rate (turns where the LLM stream tail
    leaked into overhead). With --trace, also breaks down by repo (size proxy)
    and by concurrent in-flight sessions (IO-contention proxy)."""
    rows = []  # (sid, step, post, explained_or_None, finish_fired_or_None)
    for sid, s in sorted(sessions.items()):
        for step, t in sorted(s.turns.items()):
            po = t.post_overhead_s
            if po is None:
                d, lw, tw = t.turn_duration_s, t.llm_wall_s, t.tool_wall_s
                if d is None or lw is None or tw is None:
                    continue
                po = max(0.0, d - lw - tw)
            rows.append((sid, step, float(po), t.post_stream_overhead_s,
                         t.stream_finish_fired))
    if not rows:
        print("\npost_overhead decomposition: no turns with timing data")
        return None

    po_all = np.array([r[2] for r in rows], dtype=float)
    # Turns that DO have a post_stream measurement (current patch + streamFinish fired).
    paired = [(po, ps) for _, _, po, ps, _ in rows if ps is not None]
    have_instr = any(r[3] is not None or r[4] is not None for r in rows)

    print()
    print(f"post_overhead decomposition (n={len(rows)} turns):")
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
    per_turn_csv = out / "post_overhead_per_turn.csv"
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
        path = out / "fig7_post_overhead_breakdown.pdf"
        fig.savefig(path)
        plt.close(fig)
        return path
    return None


# ---------- entry point ----------


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
    path = analyze_post_overhead(sessions, args.output, trace_path=args.trace)
    if path is not None:
        print(f"  wrote {path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
