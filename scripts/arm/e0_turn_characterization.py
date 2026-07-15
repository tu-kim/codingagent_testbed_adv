#!/usr/bin/env python3
"""E0 turn characterization: LLM-time variation + tool composition of the
fast (small-LLM) turns + KV/hit dynamics over time.

E0 is the `--sequential` characterization run (one request in flight, no
eviction interference) whose purpose is to (a) see how LLM time varies
turn-to-turn, (b) identify which (previous tool, current tool) combos
produce the SMALL-LLM turns that are CPU-offload candidates, and (c)
establish the low-load baseline for prefix-cache hit vs GPU KV size.

Views (each figure -> PDF when matplotlib is present; CSVs always):

1. fig1_turn_llm_time      THREE stacked panels on a shared TURN-index axis
                           (thin fixed-width bars, LOG y): (a) per-turn LLM
                           time, (b) per-turn tool execution time, (c) LLM/
                           tool ratio (dashed line at 1.0); red lines at the
                           TOP-LEVEL sample starts (trace-filtered main
                           sessions, not nested inside another's window).
2. fig2_llm_time_cdf       CDF of per-turn LLM time (log x) — the fast/small
                           turns (CPU-offload candidates) and the long tail.
   bottom_pct_tools.csv    for the bottom 10/20/30/40% of the LLM-time
                           distribution, the distribution and % of previous-
                           tool and current-tool.
3. fig3_hit_vs_kv          TWO stacked panels: (top) prefix-cache HIT TOKENS
                           per turn (tokens.cache.read) on a TURN-index axis,
                           per-session line segments; (bottom) GPU KV-cache
                           usage (%) on a TIME axis (scrape series trimmed to
                           the run window so an early-started scraper can't
                           shift the origin); red sample-start lines on both.
   fig3-1_worker_hit_kv    (with --worker-log) hit rate + KV usage parsed
                           STRAIGHT from the vLLM worker log's engine-stats
                           line — both off one clock, no alignment needed.
4. fig4_gap_vs_hit         turn gap (LOG x) vs PREVIOUS-TURN KV REUSE ratio
                           = cache_read(N) / (eff_input(N-1)+output(N-1)):
                           of the KV the previous turn left cached, how much
                           this turn reused. Unlike the raw hit ratio it is
                           not deflated by how much NEW content the turn read
                           (single color; also gap_hit.csv).
   zero_turns.csv          turns whose llm.end duration_s is ~0 (buffered-
                           tool-call steps; see near_zero_turns) — proof
                           they still generated output_tokens.

Turn data is loaded via the sibling scripts/analyze_turn_scheduling.py
(TurnRec: llm_wall_s, llm_start_ts/llm_end_ts, prev_tools, tool_names,
cache_hit_ratio, away_s == turn gap).

Usage:
  scripts/arm/e0_turn_characterization.py \\
      --profiles <workspace_root>/profiles \\
      --trace results/<run>/trace.jsonl     # REQUIRED: filter to MAIN sessions \\
      [--metrics logs/vllm_metrics.ndjson] \\
      [--worker-log logs/vllm-a0.log] \\
      [--cutoffs 90,80,70,60] [--out <dir>] [--no-figures]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
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


def near_zero_turns(profiles_dir: Path, keep_ids: set[str] | None,
                    threshold_s: float = 0.01) -> list[dict]:
    """Characterize turns whose llm.end duration_s is ~0 (<= threshold).

    WHY these exist: the profiler's `duration_s` is
    (AI-SDK `finish` ts | first tool.start | last text-end) - `start-step`.
    For a BUFFERED tool-call step the AI SDK fires `start-step` only AFTER
    it has already parsed the tool call out of a stream that finished
    earlier, so `finish - start-step` collapses to ~0 even though the model
    really did generate output_tokens. (The patch documents this and added
    turn.end's wall-clock-anchored `llm_wall_true_s` precisely because
    `duration_s` under-measures here — see opencode-profile.patch:99-101.)
    step_duration_s (finish-step - start-step) is a saner per-step wall.

    This dumps each near-zero turn's finish reason, output_tokens (proof the
    step DID generate), duration_s vs step_duration_s, and tools, so the
    '0.0001s turns' can be confirmed as buffered-tool-call steps rather than
    dropped blindly."""
    rows: list[dict] = []
    for f in sorted(profiles_dir.glob("*.jsonl")):
        sid = f.stem
        if keep_ids is not None and sid not in keep_ids:
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("ev") != "llm.end":
                continue
            dur = ev.get("duration_s")
            if dur is None or dur > threshold_s:
                continue
            tokens = ev.get("tokens") or {}
            out = tokens.get("output")
            if out is None:
                out = tokens.get("completion_tokens")
            rows.append({
                "session_id": sid,
                "step": ev.get("step"),
                "duration_s": dur,
                "step_duration_s": ev.get("step_duration_s"),
                "post_stream_overhead_s": ev.get("post_stream_overhead_s"),
                "finish": ev.get("finish"),
                "output_tokens": out,
                "request_id": ev.get("request_id"),
            })
    return rows


def trace_session_ids(trace_path: Path) -> set[str]:
    """session_id set from a run's trace.jsonl — exactly the MAIN sessions
    (one per sample). profiles/ additionally contains title-generation and
    `task` sub-agent sessions; filtering to this set drops them at the
    source instead of heuristically downstream."""
    ids: set[str] = set()
    with trace_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("session_id")
            if sid:
                ids.add(sid)
    return ids


def _cur_key(turn) -> str:
    return "+".join(sorted(set(turn.tool_names))) if turn.tool_names else "(none)"


def _tool_time(turn) -> float | None:
    """Per-turn tool execution wall via TurnRec.tool_time_s (turn.end
    tool_wall_s > sum of tool.end duration_s); None when no tool ran."""
    return getattr(turn, "tool_time_s", None)


def _hit_ratio(turn) -> float | None:
    """Prefix-cache hit RATIO (cache.read / effective input); falls back
    to computing it from input_tokens/cache_read for stand-ins without
    the TurnRec property."""
    r = getattr(turn, "cache_hit_ratio", None)
    if r is not None:
        return r
    inp = getattr(turn, "input_tokens", None)
    if inp is None:
        return None
    cache = getattr(turn, "cache_read", 0) or 0
    eff = inp + cache
    return (cache / eff) if eff else None


def _llm_time(turn) -> float | None:
    """Per-turn LLM time via TurnRec.llm_time_s (dynamo elapsed_s >
    llm_wall_true_s > duration_s), so buffered tool-call steps whose
    stream-based duration_s collapsed to ~0 get their REAL wall.
    Falls back to llm_wall_s for stand-ins without the property."""
    v = getattr(turn, "llm_time_s", None)
    return v if v is not None else turn.llm_wall_s


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


def _sample_windows(ordered: list, min_turns: int = 1
                    ) -> list[tuple[float, float, str]]:
    """(first_start, last_end, session_id) for each TOP-LEVEL sample: a
    >= min_turns session whose first turn does NOT start inside another
    kept session's window. Nested `task` sub-agent sessions are excluded
    here (they get merged into their parent by sample_assignment)."""
    if not ordered:
        return []
    counts: dict[str, int] = {}
    first: dict[str, float] = {}
    last_end: dict[str, float] = {}
    for t in ordered:
        counts[t.session_id] = counts.get(t.session_id, 0) + 1
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        en = t.llm_end_ts if t.llm_end_ts is not None else st
        if st is not None and t.session_id not in first:
            first[t.session_id] = st
        if en is not None:
            last_end[t.session_id] = max(last_end.get(t.session_id, en), en)
    cands = sorted((first[s], last_end.get(s, first[s]), s)
                   for s, c in counts.items() if c >= min_turns and s in first)
    out: list[tuple[float, float, str]] = []
    win_end = -math.inf
    for st, en, s in cands:
        if st < win_end:            # starts inside a kept session -> nested
            win_end = max(win_end, en)
            continue
        out.append((st, en, s))
        win_end = en
    return out


def sample_sessions(ordered: list, min_turns: int = 1) -> list[str]:
    """Ordered list of top-level SAMPLE session_ids (see _sample_windows)."""
    return [s for _st, _en, s in _sample_windows(ordered, min_turns)]


def sample_assignment(ordered: list, min_turns: int = 1) -> dict:
    """Map EVERY session_id to the sample it belongs to: a nested `task`
    sub-agent session (whose first turn falls inside a sample's window) is
    assigned to that enclosing sample; a top-level session maps to itself.

    This is what keeps a sample's fig3 line continuous: the sub-agent's
    turns are part of the SAME sample's timeline, so they must be drawn on
    the parent's line, not as a separate (overlapping) or dropped series."""
    windows = _sample_windows(ordered, min_turns)
    first: dict[str, float] = {}
    for t in ordered:
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if st is not None and t.session_id not in first:
            first[t.session_id] = st
    mapping: dict[str, str] = {}
    for sid, st in first.items():
        assigned = sid
        for ws, we, ss in windows:
            if ws <= st <= we:
                assigned = ss
                break
        mapping[sid] = assigned
    return mapping


def sample_start_times_abs(ordered: list, min_turns: int = 2) -> list[float]:
    """ABSOLUTE wall-clock start time of each top-level sample session's
    first turn. See sample_sessions for the filter."""
    keep = set(sample_sessions(ordered, min_turns))
    first: dict[str, float] = {}
    for t in ordered:
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if st is not None and t.session_id in keep and t.session_id not in first:
            first[t.session_id] = st
    return sorted(first.values())


def sample_start_times(ordered: list, min_turns: int = 2) -> list[float]:
    """Sample start times relative to run start (fig1's axis origin)."""
    t0 = _t0(ordered)
    return [v - t0 for v in sample_start_times_abs(ordered, min_turns)]


def sample_start_ordinals(ordered: list, min_turns: int = 2) -> list[int]:
    """Ordinal index (turn granularity) of each top-level sample session's
    first turn — the vertical-line positions for the turn-indexed fig1."""
    keep = set(sample_sessions(ordered, min_turns))
    seen: set[str] = set()
    out: list[int] = []
    for i, t in enumerate(ordered):
        if t.session_id in keep and t.session_id not in seen:
            seen.add(t.session_id)
            out.append(i)
    return out


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
    """For each cutoff q (e.g. 0.9), take turns whose LLM time (llm_time_s
    preference chain) is in the bottom q of the distribution, and report
    the count + % of each previous-tool and current-tool key within that
    subset."""
    walls = [w for w in (_llm_time(t) for t in turns) if w is not None]
    rows: list[dict] = []
    for q in cutoffs:
        thr = _percentile_value(walls, q)
        subset = [t for t in turns
                  if _llm_time(t) is not None and _llm_time(t) <= thr]
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


def _hit_tokens(turn):
    """Prefix-cache HIT TOKEN COUNT for this turn (tokens.cache.read), or
    None when the turn carries no usage data. Absolute tokens, not a
    ratio — the actual KV volume reused from cache."""
    if getattr(turn, "input_tokens", None) is None:
        return None
    return getattr(turn, "cache_read", 0) or 0


def hit_series(turns: list) -> list[tuple[float, float, str]]:
    """(llm_end_ts, hit_tokens, session_id) for turns with both, sorted
    by ts. Session id is kept so the plot can break the line at session
    boundaries and drop single-turn helper sessions."""
    out = []
    for t in turns:
        h = _hit_tokens(t)
        if t.llm_end_ts is not None and h is not None:
            out.append((t.llm_end_ts, h, t.session_id))
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


_WORKER_STATS_RE = re.compile(
    r"GPU KV cache usage:\s*(?P<kv>[0-9.]+)\s*%"
    r".*?Prefix cache hit rate:\s*(?P<hit>[0-9.]+)\s*%")
# leading log timestamp: "MM-DD HH:MM:SS" (vLLM/dynamo default) or ISO / bare
# "HH:MM:SS". Year is usually absent, so we work in seconds-of-day and undo
# midnight wraparound to get a monotonic relative time.
_WORKER_TS_RE = re.compile(
    r"(?:(?P<md>\d{2}-\d{2})[ T])?(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})"
    r"(?:[.,](?P<frac>\d+))?")


def _worker_line_seconds(line: str) -> float | None:
    """Seconds-of-day (float, with fractional if present) from a worker log
    line's leading timestamp, or None if no timestamp is found."""
    m = _WORKER_TS_RE.search(line[:40])
    if not m:
        return None
    sec = int(m.group("h")) * 3600 + int(m.group("m")) * 60 + int(m.group("s"))
    if m.group("frac"):
        sec += float("0." + m.group("frac"))
    return float(sec)


def worker_log_series(log_path: Path,
                      ) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Parse a vLLM/dynamo worker log (e.g. logs/vllm-a0.log) for the
    periodic engine stats line and return (hit_series, kv_series), each a
    list of (rel_seconds_from_first_stat, value_fraction).

    vLLM's LoggingStatLogger prints one line per interval carrying BOTH
    'GPU KV cache usage: X%' and 'Prefix cache hit rate: Y%'. Reading them
    from the SAME log (same clock) means the two curves need no cross-
    source alignment. Percentages are converted to 0-1 fractions. Times
    are seconds-of-day made monotonic across a midnight rollover."""
    raw: list[tuple[float, float, float]] = []
    prev = None
    day = 0.0
    with log_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            sm = _WORKER_STATS_RE.search(line)
            if not sm:
                continue
            secs = _worker_line_seconds(line)
            if secs is None:
                continue
            if prev is not None and secs < prev:   # crossed midnight
                day += 86400.0
            prev = secs
            raw.append((secs + day,
                        float(sm.group("hit")) / 100.0,
                        float(sm.group("kv")) / 100.0))
    if not raw:
        return [], []
    t0 = raw[0][0]
    hits = [(t - t0, h) for t, h, _kv in raw]
    kv = [(t - t0, k) for t, _h, k in raw]
    return hits, kv


def trim_to_window(series: list[tuple[float, float]], lo: float, hi: float,
                   margin_s: float = 5.0) -> list[tuple[float, float]]:
    """Keep only points within [lo - margin, hi + margin] — used to cut a
    scrape stream that started before / ended after the run window, so
    both fig3 series share the same visual origin."""
    return [(t, v) for t, v in series
            if lo - margin_s <= t <= hi + margin_s]


# ---------- turn gap vs hit ----------


def gap_hit_pairs(turns: list) -> list[tuple[float, float]]:
    """(turn_gap_s, hit_ratio); turn_gap == away_s."""
    out = []
    for t in turns:
        h = _hit_ratio(t)
        if t.away_s is not None and h is not None:
            out.append((t.away_s, h))
    return out


def prev_reuse_ratio(turn, by_key: dict) -> float | None:
    """Of the KV the PREVIOUS turn left cached, what fraction does THIS
    turn actually reuse:

        cache_read(N) / (effective_input(N-1) + output_tokens(N-1))

    The denominator is the previous turn's full resident sequence (its
    prompt + its generation) — the tokens that WERE in cache. Unlike the
    raw hit ratio (cache_read / this turn's effective_input), this is NOT
    deflated when the current turn injects a lot of NEW content (e.g. a
    big `read`): a turn that reuses all of the previous turn's KV scores
    ~1.0 no matter how much fresh text it also prefills. That is the
    "did the next turn keep the previous turn's cache?" question."""
    prev = by_key.get((turn.session_id, turn.step - 1))
    if prev is None:
        return None
    prev_eff = getattr(prev, "effective_input", None)
    if prev_eff is None:
        return None
    denom = prev_eff + (getattr(prev, "output_tokens", None) or 0)
    if denom <= 0:
        return None
    cr = getattr(turn, "cache_read", 0) or 0
    return cr / denom


def gap_reuse_pairs(turns: list) -> list[tuple[float, float, str, int, bool]]:
    """(turn_gap, prev_reuse_ratio, session_id, step, prev_was_task) for
    turns with a gap and a resolvable previous turn. prev_was_task marks a
    RESUME turn (the previous turn called the `task` sub-agent), so fig4
    can flag whether returning from a sub-agent excursion changes reuse."""
    by_key = {(t.session_id, t.step): t for t in turns}
    out = []
    for t in turns:
        if t.away_s is None:
            continue
        r = prev_reuse_ratio(t, by_key)
        if r is None:
            continue
        prev_task = "task" in (t.prev_tools or ())
        out.append((t.away_s, r, t.session_id, t.step, prev_task))
    return out


# ---------- CSV writers ----------


def write_ordered_csv(path: Path, ordered: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ordinal", "session_id", "step", "llm_start_ts",
                    "llm_end_ts", "llm_time_s", "llm_wall_s", "prev_tools",
                    "cur_tools", "turn_gap_s", "cache_hit_ratio"])
        for i, t in enumerate(ordered):
            lt = _llm_time(t)
            w.writerow([i, t.session_id, t.step,
                        t.llm_start_ts if t.llm_start_ts is not None else "",
                        t.llm_end_ts if t.llm_end_ts is not None else "",
                        lt if lt is not None else "",
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


def fig_turn_llm_time(ordered: list, path: Path, *,
                      min_turns_boundary: int = 2) -> None:
    """Three stacked subplots on a shared TURN-index x axis (thin vline
    bars, width does not scale with value):
      (a) per-turn LLM time (log y),
      (b) per-turn tool execution time (log y); turns that ran the `task`
          sub-agent tool are drawn in a distinct color,
      (c) signed log2(LLM / tool): 0 = equal, POSITIVE (LLM-bound) bars
          go up in one color, NEGATIVE (tool-bound) go down in another,
          so the balance flips visibly around the zero line.
    Red lines mark top-level SAMPLE starts on every panel."""
    plt = _mpl()
    n = len(ordered)
    xs = list(range(n))
    llm, tool, tool_colors, ratio, ratio_colors = [], [], [], [], []
    for t in ordered:
        lw = _llm_time(t)
        tw = _tool_time(t)
        llm.append(lw if lw is not None and lw > 0 else float("nan"))
        tool.append(tw if tw is not None and tw > 0 else float("nan"))
        is_task = any(str(name) == "task" for name in t.tool_names)
        tool_colors.append("tab:orange" if is_task else "tab:green")
        if lw and tw and lw > 0 and tw > 0:
            r = math.log2(lw / tw)     # >0 LLM-bound, <0 tool-bound
            ratio.append(r)
            ratio_colors.append("tab:blue" if r >= 0 else "tab:orange")
        else:
            ratio.append(float("nan"))
            ratio_colors.append("none")
    bounds = sample_start_ordinals(ordered, min_turns_boundary)

    fig, axes = plt.subplots(3, 1, figsize=(18, 12), sharex=True)

    # (a) LLM time — log y
    lpos = [v for v in llm if v == v and v > 0]
    axes[0].set_yscale("log")
    axes[0].vlines(xs, min(lpos) if lpos else 1e-3, llm, color="tab:blue",
                   linewidth=0.7)
    if lpos:
        axes[0].set_ylim(min(lpos) * 0.8, max(lpos) * 1.2)
    axes[0].set_ylabel("LLM time / turn (s)")
    axes[0].set_title("Per-turn LLM Time vs turn")

    # (b) tool exec time — log y, task-tool turns colored distinctly
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

    # (c) signed log2 ratio — linear y around 0, up/down by sign+color
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
        for b in bounds:
            ax.axvline(b, color="crimson", linewidth=0.6, alpha=0.7)
    axes[2].set_xlabel("turn")
    axes[2].set_xlim(0, max(n - 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_llm_time_cdf(turns: list, path: Path) -> None:
    """CDF of per-turn LLM time — the distribution behind fig1's bars, on
    a log x-axis so the small/fast turns (CPU-offload candidates) and the
    long tail are both legible."""
    plt = _mpl()
    vals = sorted(w for w in (_llm_time(t) for t in turns)
                  if w is not None and w > 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    if vals:
        n = len(vals)
        ys = [(i + 1) / n for i in range(n)]
        ax.step(vals, ys, where="post", color="tab:blue")
        ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("LLM time / turn (s)")
    ax.set_ylabel("cumulative fraction of turns")
    ax.set_title("Per-turn LLM Time CDF")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_hit_vs_kv(ordered: list, kv: list[tuple[float, float]], path: Path,
                  *, min_turns: int = 2,
                  sample_ordinals: list[int] | None = None,
                  sample_times: list[float] | None = None) -> None:
    """Two stacked subplots:
      (top)    prefix-cache HIT TOKENS per turn (tokens.cache.read) on a
               TURN-index x axis, one line per session, BROKEN wherever a
               session's ordinals are non-consecutive (the parent is paused
               while its `task` sub-agent runs). Sub-agent turns are marked
               with a triangle, sample turns with a dot; red lines at
               sample-start ordinals.
      (bottom) GPU KV-cache usage (%) on a TIME x axis (seconds from the
               run window start; the scrape series is TRIMMED to the run
               window so an early/late scraper can't shift the origin);
               red lines at sample-start times."""
    plt = _mpl()
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(18, 10))

    # ---- top: hit tokens vs turn ordinal, ONE line per session ----
    # Parent sample and its nested `task` sub-agent are drawn as SEPARATE
    # lines (each session_id its own line, in ordinal order). They have
    # independent KV prefixes, so a sub-agent's line starting low is its
    # own fresh prefix, not a dip in the parent's.
    # marker per session: `task` sub-agent turns get a triangle, the
    # top-level sample gets a dot.
    assign = sample_assignment(ordered, min_turns)
    by_sess: dict[str, list[tuple[int, float]]] = {}
    for i, t in enumerate(ordered):
        h = _hit_tokens(t)
        if h is not None:
            by_sess.setdefault(t.session_id, []).append((i, h))
    for b in (sample_ordinals or []):
        ax_top.axvline(b, color="crimson", linewidth=0.5, alpha=0.7, zorder=1)
    drew_sub = False
    for sid, pts in by_sess.items():
        pts.sort()
        nested = assign.get(sid, sid) != sid
        marker, ms = ("^", 1.5) if nested else (".", 3)
        # legend shows the sub-agent triangle only (labeled once)
        label = None
        if nested and not drew_sub:
            label, drew_sub = "sub-agent", True
        # Break the line wherever this session's ordinals are NOT
        # consecutive: a gap means another session's turns run there (the
        # parent is PAUSED while its `task` sub-agent runs). Drawing one
        # connected line would span that gap and cross the sub-agent's
        # line — the apparent "overlap". Split into contiguous runs so the
        # paused stretch shows no line, matching the blocking semantics.
        seg_x: list[int] = []
        seg_y: list[float] = []
        prev_i = None
        for i, h in pts:
            if prev_i is not None and i - prev_i > 1:
                ax_top.plot(seg_x, seg_y, color="tab:blue", marker=marker,
                            ms=ms, lw=0.8, zorder=2, label=label)
                label = None            # label only the first segment
                seg_x, seg_y = [], []
            seg_x.append(i)
            seg_y.append(h)
            prev_i = i
        if seg_x:
            ax_top.plot(seg_x, seg_y, color="tab:blue", marker=marker,
                        ms=ms, lw=0.8, zorder=2, label=label)
    ax_top.set_xlim(0, max(len(ordered) - 1, 1))
    ax_top.set_ylim(bottom=0)
    if drew_sub:
        ax_top.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax_top.set_xlabel("turn")
    ax_top.set_ylabel("prefix-cache hit tokens / turn")
    ax_top.set_title("Prefix-cache Hit Tokens vs turn")

    # ---- bottom: GPU KV usage (%) vs time ----
    # run window from the turns' own clock; trim + rebase the scrape series
    ts_all = [t.llm_end_ts for t in ordered if t.llm_end_ts is not None] + \
             [t.llm_start_ts for t in ordered if t.llm_start_ts is not None]
    if ts_all:
        lo, hi = min(ts_all), max(ts_all)
        kv = trim_to_window(kv, lo, hi)
        t0 = min([lo] + [p[0] for p in kv])
    else:
        t0 = min((p[0] for p in kv), default=0.0)
    x_right = 0.0
    for b in (sample_times or []):
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


def fig_worker_hit_kv(hits: list[tuple[float, float]],
                      kv: list[tuple[float, float]], path: Path) -> None:
    """fig3-1: prefix-cache hit rate (left y) and GPU KV-cache usage
    (right y) parsed DIRECTLY from the vLLM worker log's periodic stats
    line. Both come from the same log/clock, so they share the x-axis
    (seconds from the first stats line) with no cross-source alignment."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(18, 6))
    x_right = 0.0
    if hits:
        hx = [t for t, _ in hits]
        ax.plot(hx, [v for _, v in hits], color="tab:blue", lw=1.0,
                marker=".", ms=3)
        x_right = max(x_right, hx[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("prefix-cache hit rate", color="tab:blue")
    ax.set_ylim(0, 1)
    if kv:
        ax2 = ax.twinx()
        kx = [t for t, _ in kv]
        ax2.plot(kx, [v for _, v in kv], color="tab:orange", lw=1.0)
        ax2.set_ylabel("GPU KV-cache usage", color="tab:orange")
        ax2.set_ylim(0, 1)
        x_right = max(x_right, kx[-1])
    ax.set_xlim(0, x_right if x_right > 0 else 1)
    ax.set_title("KV Cache Status vs time (from worker log)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_gap_vs_hit(pairs: list[tuple[float, float, str, int, bool]],
                   path: Path) -> None:
    """Scatter of turn gap (LOG x) vs the previous-turn KV reuse ratio
    (cache_read(N) / [effective_input(N-1) + output(N-1)]): how much of
    the previous turn's cached KV this turn actually reused. Turns whose
    previous turn called the `task` sub-agent (RESUME turns) are drawn as
    triangles so we can see whether returning from a sub-agent excursion
    changes reuse; other turns are dots."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    pos = [p for p in pairs if p[0] > 0]       # log x needs gap > 0
    normal = [(p[0], p[1]) for p in pos if not (len(p) > 4 and p[4])]
    resume = [(p[0], p[1]) for p in pos if len(p) > 4 and p[4]]
    if normal:
        ax.scatter([g for g, _ in normal], [h for _, h in normal],
                   s=14, alpha=0.6, color="tab:blue", label="turn")
    if resume:
        ax.scatter([g for g, _ in resume], [h for _, h in resume],
                   s=20, alpha=0.9, color="tab:blue", marker="^",
                   label="resume after sub-agent")
    if pos:
        ax.set_xscale("log")
    if resume:
        ax.legend(fontsize=8, loc="best", framealpha=0.7)
    ax.set_xlabel("turn gap (s)")
    ax.set_ylabel("prev-turn KV reuse ratio")
    ax.set_ylim(bottom=0)
    ax.set_title("Previous-turn KV reuse vs turn gap")
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


def _print_zero_turns(rows: list[dict], thr: float, total: int) -> None:
    print(f"\nnear-zero LLM turns (duration_s <= {thr:g}s): "
          f"{len(rows)}/{total}")
    if not rows:
        return
    with_out = sum(1 for r in rows
                   if isinstance(r["output_tokens"], (int, float))
                   and r["output_tokens"] > 0)
    fins = Counter(str(r["finish"]) for r in rows)
    step_durs = [r["step_duration_s"] for r in rows
                 if isinstance(r["step_duration_s"], (int, float))]
    print(f"  {with_out}/{len(rows)} DID generate output_tokens>0 -> these are "
          "NOT idle turns; duration_s collapsed because start-step fired after "
          "the tool call was already buffered (see zero_turns.csv).")
    print("  finish reasons: "
          + ", ".join(f"{k} {v}" for k, v in fins.most_common()))
    if step_durs:
        step_durs.sort()
        med = step_durs[len(step_durs) // 2]
        print(f"  step_duration_s (finish-step - start-step) of the same turns: "
              f"median {med:.3f}s, max {max(step_durs):.3f}s "
              "-> the real per-step wall is here, not in duration_s.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--metrics", type=Path, default=None,
                    help="vLLM scrape NDJSON for the KV-usage right axis")
    ap.add_argument("--worker-log", type=Path, default=None,
                    help="vLLM/dynamo worker log (e.g. logs/vllm-a0.log); "
                         "fig3-1 plots hit rate + KV usage parsed straight "
                         "from its periodic engine-stats line")
    ap.add_argument("--zero-threshold-s", type=float, default=0.01,
                    help="turns with llm.end duration_s <= this are dumped to "
                         "zero_turns.csv (the buffered-tool-call steps whose "
                         "duration_s collapses to ~0). Default 0.01.")
    ap.add_argument("--trace", required=True, type=Path,
                    help="run's trace.jsonl (REQUIRED). Only turns from its "
                         "session_ids (the MAIN per-sample sessions) are "
                         "analyzed — drops the title / task-subagent sessions "
                         "that also land in profiles/. Every kept session is a "
                         "real sample, so no turn-count heuristic is applied.")
    ap.add_argument("--cutoffs", default="10,20,30,40",
                    help="bottom-percentile cutoffs (comma list of %); the "
                         "fastest/smallest-LLM-time turns = CPU-offload "
                         "candidates. Default bottom 10,20,30,40%")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    if not args.profiles.is_dir():
        print(f"error: profiles dir not found: {args.profiles}", file=sys.stderr)
        return 2
    cutoffs = [float(c) / 100.0 for c in args.cutoffs.split(",") if c.strip()]

    if not args.trace.is_file():
        print(f"error: trace not found: {args.trace}", file=sys.stderr)
        return 2

    ats = _load_ats()
    turns = ats.load_turns(args.profiles)
    if not turns:
        print("error: no turns parsed from profiles", file=sys.stderr)
        return 2
    main_ids = trace_session_ids(args.trace)
    if not main_ids:
        print("error: no session_id in trace.jsonl", file=sys.stderr)
        return 2
    turns = [t for t in turns if t.session_id in main_ids]
    if not turns:
        print("error: no turns left after trace filter", file=sys.stderr)
        return 2

    # Every kept session is a main sample; no turn-count heuristic needed
    # (a 1-turn sample must still be kept), so the sample filter uses 1.
    min_boundary = 1

    ordered = order_turns(turns)
    samples_abs = sample_start_times_abs(ordered, min_boundary)
    sample_ords = sample_start_ordinals(ordered, min_boundary)
    reuse = gap_reuse_pairs(turns)
    have_metrics = args.metrics is not None and args.metrics.exists()
    kv = kv_usage_series(args.metrics) if have_metrics else []

    out_dir = args.out or (args.profiles / "e0")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Figures only. CSV side-outputs and the stdout digest are intentionally
    # omitted for now (analyzed later off the figures / raw profiles).
    if not args.no_figures:
        try:
            fig_turn_llm_time(ordered, out_dir / "fig1_turn_llm_time.pdf",
                              min_turns_boundary=min_boundary)
            fig_llm_time_cdf(turns, out_dir / "fig2_llm_time_cdf.pdf")
            fig_hit_vs_kv(ordered, kv, out_dir / "fig3_hit_vs_kv.pdf",
                          min_turns=min_boundary,
                          sample_ordinals=sample_ords,
                          sample_times=samples_abs)
            fig_gap_vs_hit(reuse, out_dir / "fig4_gap_vs_hit.pdf")
            if args.worker_log and args.worker_log.exists():
                wl_hits, wl_kv = worker_log_series(args.worker_log)
                if wl_hits or wl_kv:
                    fig_worker_hit_kv(wl_hits, wl_kv,
                                      out_dir / "fig3-1_worker_hit_kv.pdf")
        except ImportError:
            print("matplotlib not available (no figures written)",
                  file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
