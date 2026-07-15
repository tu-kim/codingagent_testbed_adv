#!/usr/bin/env python3
"""E0 turn characterization: LLM-time variation + tool composition of the
fast (small-LLM) turns + KV/hit dynamics over time.

E0 is the `--sequential` characterization run (one request in flight, no
eviction interference) whose purpose is to (a) see how LLM time varies
turn-to-turn, (b) identify which (previous tool, current tool) combos
produce the SMALL-LLM turns that are CPU-offload candidates, and (c)
establish the low-load baseline for prefix-cache hit vs GPU KV size.

Views (each figure -> PDF when matplotlib is present; CSVs always):

1. fig1_turn_llm_time      per-turn LLM time (llm.end duration_s) as FIXED-
                           WIDTH thin bars on a TURN-index axis (x = ordinal
                           turn; width does NOT scale with LLM time), LOG y
                           over the full range (no clip); red lines at TOP-
                           LEVEL sample starts (>= --min-turns-boundary turns
                           AND not nested inside another kept session's
                           window, so `task` sub-agents don't add lines).
2. fig2_llm_time_cdf       CDF of per-turn LLM time (log x) — the fast/small
                           turns (CPU-offload candidates) and the long tail.
   bottom_pct_tools.csv    for the bottom 10/20/30/40% of the LLM-time
                           distribution, the distribution and % of previous-
                           tool and current-tool.
3. fig3_hit_vs_kv          shared time axis (x from 0) framed by the RUN
                           WINDOW (first..last turn): per-turn prefix-cache
                           HIT TOKENS (tokens.cache.read; left y, PER-SESSION
                           line segments) + GPU KV-cache usage (right y,
                           trimmed to the window so an early-started scraper
                           can't shift the origin); red lines at sample
                           starts (as in fig1).
   fig3-1_worker_hit_kv    (with --worker-log) hit rate + KV usage parsed
                           STRAIGHT from the vLLM worker log's engine-stats
                           line — both off one clock, no alignment needed.
4. fig4_gap_vs_hit         turn gap (llm.start(N) - llm.end(N-1)) vs prefix-
                           cache HIT TOKENS, points COLORED by the previous
                           turn's tool (also gap_hit.csv).
   zero_turns.csv          turns whose llm.end duration_s is ~0 (buffered-
                           tool-call steps; see near_zero_turns) — proof
                           they still generated output_tokens.

Turn data is loaded via the sibling scripts/analyze_turn_scheduling.py
(TurnRec: llm_wall_s, llm_start_ts/llm_end_ts, prev_tools, tool_names,
cache_hit_ratio, away_s == turn gap).

Usage:
  scripts/arm/e0_turn_characterization.py \\
      --profiles <workspace_root>/profiles \\
      [--trace results/<run>/trace.jsonl]   # filter to MAIN sessions \\
      [--metrics logs/vllm_metrics.ndjson] \\
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


def sample_sessions(ordered: list, min_turns: int = 1) -> list[str]:
    """Ordered list of top-level SAMPLE session_ids.

    Two filters combine:
    1. >= min_turns turns. With `--trace` every remaining session already
       IS a main sample, so pass min_turns=1 (don't drop 1-turn samples).
       Without trace it defaults to 2 to shed single-turn helpers.
    2. TOP-LEVEL only: a session whose first turn starts INSIDE another
       kept session's [first_start, last_end) window is a nested helper
       (`task` sub-agent spawned mid-sample) and is dropped."""
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
    out: list[str] = []
    win_end = -math.inf
    for st, en, s in cands:
        if st < win_end:            # starts inside a kept session -> nested
            win_end = max(win_end, en)
            continue
        out.append(s)
        win_end = en
    return out


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
    """(turn_gap_s, hit_tokens); turn_gap == away_s."""
    out = []
    for t in turns:
        h = _hit_tokens(t)
        if t.away_s is not None and h is not None:
            out.append((t.away_s, h))
    return out


def categorize_gap_hit(turns: list) -> list[tuple[float, float, str, str, int]]:
    """(turn_gap, hit_tokens, category, session_id, step) for turns with
    gap + usage data.

    category = the previous turn's tool key (prev_key): in sequential runs
    the turn gap is ~the preceding tool's exec time, so coloring by prev
    tool explains most of the vertical spread at a given gap."""
    out = []
    for t in turns:
        h = _hit_tokens(t)
        if t.away_s is None or h is None:
            continue
        out.append((t.away_s, h, t.prev_key, t.session_id, t.step))
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
    """Each turn as a THIN bar on a TURN-index axis (x = ordinal turn
    number); height = LLM time on a LOG y-axis (full range, no clip).
    Red lines mark top-level SAMPLE starts (turn granularity)."""
    plt = _mpl()
    heights = []
    for t in ordered:
        w = _llm_time(t)
        wall = w if w is not None else 0.0
        heights.append(wall if wall > 0 else float("nan"))
    xs = list(range(len(heights)))
    pos = [h for h in heights if h == h and h > 0]
    fig, ax = plt.subplots(figsize=(18, 5))
    ax.set_yscale("log")
    ax.vlines(xs, 1e-3, heights, color="tab:blue", linewidth=0.7)
    for b in sample_start_ordinals(ordered, min_turns_boundary):
        ax.axvline(b, color="crimson", linewidth=0.6, alpha=0.7)
    ax.set_xlim(0, max(len(xs) - 1, 1))
    if pos:
        ax.set_ylim(min(pos) * 0.8, max(pos) * 1.2)
    ax.set_xlabel("turn")
    ax.set_ylabel("LLM time / turn (s)")
    ax.set_title("Per-turn LLM Time vs turn")
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


def fig_hit_vs_kv(hits: list[tuple[float, float, str]],
                  kv: list[tuple[float, float]], path: Path,
                  *, min_turns: int = 2,
                  sample_times: list[float] | None = None) -> None:
    """Prefix-cache HIT TOKENS per turn (left y, tokens.cache.read — the
    absolute KV volume reused) vs GPU KV usage (right y) on a SHARED
    time axis. The RUN WINDOW (first..last turn hit point) frames
    it: the KV series is TRIMMED to that window so an early/late scraper
    can't shift the origin, and both start together at x=0. The per-turn
    hit line is drawn PER SESSION (no segment crosses a session boundary)
    and sub-`min_turns` helper sessions are dropped. Red vertical lines
    mark top-level SAMPLE starts (times relative to the same origin, as
    in fig1)."""
    plt = _mpl()
    # keep only sessions with >= min_turns hit points
    counts: dict[str, int] = {}
    for _ts, _h, sid in hits:
        counts[sid] = counts.get(sid, 0) + 1
    kept = [(ts, h, sid) for ts, h, sid in hits if counts[sid] >= min_turns]

    if kept:
        lo = min(p[0] for p in kept)
        hi = max(p[0] for p in kept)
        kv = trim_to_window(kv, lo, hi)
    origins = [p[0] for p in kept] + [p[0] for p in kv]
    t0 = min(origins) if origins else 0.0
    x_right = 0.0

    fig, ax = plt.subplots(figsize=(18, 6))
    # sample_times are ABSOLUTE; subtract this axis's own t0 so they line
    # up with the hit points (which are on the same absolute clock).
    for b in (sample_times or []):
        ax.axvline(b - t0, color="crimson", linewidth=1.0, alpha=0.85,
                   zorder=1)
    # group kept hits by session, draw each as its own line segment
    by_sess: dict[str, list[tuple[float, float]]] = {}
    for ts, h, sid in kept:
        by_sess.setdefault(sid, []).append((ts, h))
    for sid, pts in by_sess.items():
        pts.sort()
        xs = [ts - t0 for ts, _ in pts]
        ys = [h for _, h in pts]
        ax.plot(xs, ys, color="tab:blue", marker=".", ms=3, lw=0.8, zorder=2)
        if xs:
            x_right = max(x_right, xs[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("prefix-cache hit tokens / turn", color="tab:blue")
    ax.set_ylim(bottom=0)
    if kv:
        ax2 = ax.twinx()
        kx = [t - t0 for t, _ in kv]
        ax2.plot(kx, [v for _, v in kv], color="tab:orange", lw=1.0,
                 label="KV usage")
        ax2.set_ylabel("GPU KV-cache usage", color="tab:orange")
        if kx:
            x_right = max(x_right, kx[-1])
    ax.set_xlim(0, x_right if x_right > 0 else 1)
    ax.set_title("KV Cache Status vs time")
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


def fig_gap_vs_hit(cats: list[tuple[float, float, str, str, int]],
                   path: Path) -> None:
    """Scatter of turn gap vs hit tokens, COLORED by the previous turn's
    tool so the vertical spread at a given gap is explained."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    groups: dict[str, list[tuple[float, float]]] = {}
    for g, h, c, _sid, _step in cats:
        groups.setdefault(c, []).append((g, h))
    for c in sorted(groups):
        pts = groups[c]
        gs = [p[0] for p in pts]
        hs = [p[1] for p in pts]
        ax.scatter(gs, hs, s=14, alpha=0.6, label=f"{c} ({len(pts)})")
    ax.set_xlabel("turn gap (s)")
    ax.set_ylabel("prefix-cache hit tokens")
    ax.set_ylim(bottom=0)
    if groups:
        ax.legend(fontsize=7, loc="best", framealpha=0.7)
    ax.set_title("Prefix-cache hit tokens vs turn gap")
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
    ap.add_argument("--trace", type=Path, default=None,
                    help="run's trace.jsonl; when given, only turns from its "
                         "session_ids (the MAIN per-sample sessions) are "
                         "analyzed — drops title/task-subagent sessions that "
                         "also land in profiles/")
    ap.add_argument("--cutoffs", default="10,20,30,40",
                    help="bottom-percentile cutoffs (comma list of %); the "
                         "fastest/smallest-LLM-time turns = CPU-offload "
                         "candidates. Default bottom 10,20,30,40%")
    ap.add_argument("--min-turns-boundary", type=int, default=2,
                    help="fig1/fig3: only sessions with >= this many turns get "
                         "a sample-boundary line (filters single-turn title / "
                         "task-subagent sessions). Default 2, but forced to 1 "
                         "when --trace is given (every trace session already "
                         "IS a main sample, so 1-turn samples must be kept).")
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
    if args.trace is not None:
        if not args.trace.is_file():
            print(f"error: trace not found: {args.trace}", file=sys.stderr)
            return 2
        main_ids = trace_session_ids(args.trace)
        if not main_ids:
            print("error: no session_id in trace.jsonl", file=sys.stderr)
            return 2
        before = len({t.session_id for t in turns})
        turns = [t for t in turns if t.session_id in main_ids]
        kept = {t.session_id for t in turns}
        print(f"trace filter: kept {len(kept)}/{before} sessions "
              f"({len(main_ids)} main sessions in trace"
              + (f"; {len(main_ids - kept)} with no profile" if main_ids - kept
                 else "") + ")")
        if not turns:
            print("error: no turns left after trace filter", file=sys.stderr)
            return 2

    # With --trace every remaining session is already a main sample, so the
    # >=2-turn heuristic must not shed a legitimate 1-turn sample.
    min_boundary = 1 if args.trace is not None else args.min_turns_boundary

    ordered = order_turns(turns)
    samples = sample_start_times(ordered, min_boundary)
    samples_abs = sample_start_times_abs(ordered, min_boundary)
    bottom_rows = bottom_pct_tool_dist(turns, cutoffs)
    hits = hit_series(turns)
    cats = categorize_gap_hit(turns)
    have_metrics = args.metrics is not None and args.metrics.exists()
    kv = kv_usage_series(args.metrics) if have_metrics else []

    out_dir = args.out or (args.profiles / "e0")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_ordered_csv(out_dir / "turns_ordered.csv", ordered)
    write_bottom_pct_csv(out_dir / "bottom_pct_tools.csv", bottom_rows)
    with (out_dir / "gap_hit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session_id", "step", "turn_gap_s", "cache_hit_tokens", "category"])
        for g, h, c, sid, step in cats:
            w.writerow([sid, step, g, h, c])

    # near-zero-duration turn diagnostic (buffered-tool-call steps)
    keep_ids = {t.session_id for t in turns}
    zrows = near_zero_turns(args.profiles, keep_ids, args.zero_threshold_s)
    with (out_dir / "zero_turns.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "session_id", "step", "duration_s", "step_duration_s",
            "post_stream_overhead_s", "finish", "output_tokens", "request_id"])
        w.writeheader()
        w.writerows(zrows)

    n_sessions = len({t.session_id for t in turns})
    print(f"turns: {len(turns)} across {n_sessions} sessions "
          f"({len(samples)} samples with >= {min_boundary} turns)")
    _print_bottom(bottom_rows)
    _print_zero_turns(zrows, args.zero_threshold_s, len(turns))
    if args.metrics and not kv:
        print("warning: no KV-usage series parsed from --metrics", file=sys.stderr)

    if not args.no_figures:
        try:
            fig_turn_llm_time(ordered, out_dir / "fig1_turn_llm_time.pdf",
                              min_turns_boundary=min_boundary)
            fig_llm_time_cdf(turns, out_dir / "fig2_llm_time_cdf.pdf")
            fig_hit_vs_kv(hits, kv, out_dir / "fig3_hit_vs_kv.pdf",
                          min_turns=min_boundary, sample_times=samples_abs)
            fig_gap_vs_hit(cats, out_dir / "fig4_gap_vs_hit.pdf")
            if args.worker_log and args.worker_log.exists():
                wl_hits, wl_kv = worker_log_series(args.worker_log)
                if wl_hits or wl_kv:
                    fig_worker_hit_kv(wl_hits, wl_kv,
                                      out_dir / "fig3-1_worker_hit_kv.pdf")
                else:
                    print("warning: no engine-stats lines parsed from "
                          f"--worker-log {args.worker_log}", file=sys.stderr)
            print(f"\nwrote figures under {out_dir}")
        except ImportError:
            print("matplotlib not available; wrote CSVs only "
                  "(--no-figures to silence)", file=sys.stderr)

    print(f"wrote {out_dir / 'turns_ordered.csv'}")
    print(f"wrote {out_dir / 'bottom_pct_tools.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
