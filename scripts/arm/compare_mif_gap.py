#!/usr/bin/env python3
"""Compare the turn-GAP decomposition across max-in-flight (mif) settings
to locate the cause of gap inflation.

CANONICAL DEFINITION (prefill is NOT part of the gap — it is real GPU
compute, not waiting/overhead, so it is subtracted out):

  turn_gap = tool + scaffold + queue_wait

    tool        tool execution wall of the PREVIOUS step (host, off-GPU)
    scaffold    gap_full - tool - ttft   (host: opencode post-processing,
                snapshot/DB/git, event loop, request build)
    queue_wait  vLLM ENGINE queue delay  (SCHED_DELAY queue_ms; server,
                waiting to be scheduled)

  where the raw full gap and the removed prefill are:
    gap_full = away_s(N) = llm.start(N) - llm.end(N-1)   (includes prefill)
    ttft     = server receive -> first token (frontend ttft_ms)
    prefill  = ttft - queue_wait                          (on-GPU compute)
    turn_gap = gap_full - prefill

Component sources: `tool` from profiles; `queue_wait` from --logs
SCHED_DELAY (clock-independent duration); `ttft` (-> prefill, scaffold,
turn_gap) from --frontend. Without --frontend only `host_ub` = gap_full -
tool - queue_wait is available (scaffold upper bound: still includes
prefill). All joins are by request_id.

If tool inflates with mif -> CPU contention on tool execution.
If scaffold inflates -> opencode post-processing / event-loop contention.
If queue_wait inflates -> LLM serving saturation (not host-side at all).

CRITICAL: across mif settings the processing ORDER differs and the run is
non-deterministic, so the SAME session's turns / llm / tool internals all
differ. We compare DISTRIBUTIONS (percentiles, CDFs), never paired
per-turn, and break tool time out PER TOOL NAME so a mif that merely ran
more expensive tools is not mistaken for contention.

Usage:
  scripts/arm/compare_mif_gap.py \\
      --run  mif4  <profiles> <trace.jsonl> \\
      --logs mif4  <logs_dir> --frontend mif4 <logs_dir>/frontend.log \\
      --run  mif16 <profiles> <trace.jsonl> \\
      --logs mif16 <logs_dir> --frontend mif16 <logs_dir>/frontend.log \\
      [--out <dir>] [--no-figures]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path

# frontend "request completed" line: request_id + ttft_ms (server-side time
# to first token = frontend/engine queue + prefill). ANSI-stripped.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJ]")
_REQID_RE = re.compile(r'(?:\b|")request_id\b"?\s*[=:]\s*"?(?P<v>[^\s",}]+)"?')
_TTFT_RE = re.compile(r'(?:\b|")ttft_ms\b"?\s*[=:]\s*"?(?P<v>[\d.]+)"?')


def parse_frontend_ttft(path: Path) -> dict[str, float]:
    """request_id -> ttft_ms from the dynamo frontend log."""
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if "request completed" not in line:
                continue
            line = _ANSI_RE.sub("", line)
            rid = _REQID_RE.search(line)
            ttft = _TTFT_RE.search(line)
            if rid and ttft:
                out[rid.group("v")] = float(ttft.group("v"))
    return out

_HERE = Path(__file__).resolve().parent
_ATS_PATH = _HERE.parent / "analyze_turn_scheduling.py"
_E0_PATH = _HERE / "e0_turn_characterization.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    import math
    idx = min(len(s) - 1, max(0, int(math.ceil(q * len(s))) - 1))
    return s[idx]


def gap_rows(profiles: Path, trace: Path, e0, ats,
             logs: Path | None = None,
             frontend: Path | None = None) -> list[dict]:
    """Per-turn CANONICAL gap decomposition for the run's MAIN sessions:

        turn_gap = tool + scaffold + queue_wait   (prefill removed)

    Columns per row (None when the needed source is absent):
      gap_full   away_s(N) = llm.start(N) - llm.end(N-1)  (raw, incl prefill)
      tool       previous step's tool wall
      queue_wait vLLM engine queue delay (SCHED_DELAY queue_ms, needs --logs)
      prefill    ttft - queue_wait   (on-GPU compute, needs --frontend)
      scaffold   gap_full - tool - ttft   (host residual, needs --frontend)
      turn_gap   tool + scaffold + queue_wait = gap_full - prefill
      host_ub    gap_full - tool - queue_wait  (scaffold upper bound when no
                 --frontend; still includes prefill)
    See the module docstring for the definition and clock caveats."""
    turns = ats.load_turns(profiles)
    ids = e0.trace_session_ids(trace)
    turns = [t for t in turns if t.session_id in ids]
    sched = ats.load_sched(logs) if logs is not None and logs.exists() else {}
    ttfts = parse_frontend_ttft(frontend) if frontend is not None else {}
    by_key = {(t.session_id, t.step): t for t in turns}
    rows: list[dict] = []
    for t in turns:
        if t.away_s is None:
            continue
        prev = by_key.get((t.session_id, t.step - 1))
        tool = (getattr(prev, "tool_time_s", None) or 0.0) if prev else 0.0
        gap_full = t.away_s
        rid = t.request_id
        queue_wait = (sched[rid].total_queue_ms / 1000.0
                      if rid and rid in sched else None)
        # ttft_ms (server receive -> first token) = queue_wait + prefill
        ttft_s = (ttfts[rid] / 1000.0) if rid and rid in ttfts else None
        prefill = (max(0.0, ttft_s - queue_wait)
                   if ttft_s is not None and queue_wait is not None else None)
        # scaffold = gap_full - tool - ttft (host residual; server chunk out)
        scaffold = (max(0.0, gap_full - tool - ttft_s)
                    if ttft_s is not None else None)
        # canonical turn_gap = tool + scaffold + queue_wait = gap_full-prefill
        turn_gap = (max(0.0, gap_full - prefill)
                    if prefill is not None else None)
        # fallback scaffold upper bound when no --frontend (still incl prefill)
        host_ub = (max(0.0, gap_full - tool - queue_wait)
                   if queue_wait is not None else None)
        rows.append({"turn_gap": turn_gap, "gap_full": gap_full, "tool": tool,
                     "scaffold": scaffold, "queue_wait": queue_wait,
                     "prefill": prefill, "host_ub": host_ub})
    return rows


def tool_durations(profiles: Path, keep_ids: set[str]) -> dict[str, list[float]]:
    """Per tool NAME, the list of individual tool.end duration_s (from the
    MAIN sessions only). Lets us compare same-tool timing across mif so a
    composition shift is not read as contention."""
    out: dict[str, list[float]] = {}
    for f in sorted(profiles.glob("*.jsonl")):
        if f.stem not in keep_ids:
            continue
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("ev") != "tool.end":
                continue
            name = ev.get("name")
            dur = ev.get("duration_s")
            if name and isinstance(dur, (int, float)):
                out.setdefault(str(name), []).append(float(dur))
    return out


def summarize(rows: list[dict]) -> dict:
    out = {}
    for k in ("turn_gap", "gap_full", "tool", "scaffold", "queue_wait", "prefill", "host_ub"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        out[k] = {"n": len(vals),
                  "p50": _pct(vals, 0.5),
                  "p90": _pct(vals, 0.9),
                  "p99": _pct(vals, 0.99),
                  "mean": (sum(vals) / len(vals)) if vals else float("nan")}
    return out


# ---------- figures ----------


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _cdf(ax, vals: list[float], label: str) -> None:
    v = sorted(x for x in vals if x == x)
    if not v:
        return
    ys = [(i + 1) / len(v) for i in range(len(v))]
    ax.step(v, ys, where="post", label=label, lw=1.0)


def fig_gap_cdfs(runs: list[tuple[str, list[dict]]], path: Path) -> None:
    """CDFs of the canonical components: turn_gap = tool + scaffold +
    queue_wait (and queue_wait separately)."""
    plt = _mpl()
    keys = ("turn_gap", "tool", "scaffold", "queue_wait")
    titles = ("turn_gap (= tool+scaffold+queue_wait)", "tool (host)",
              "scaffold (host)", "queue_wait (vLLM engine queue)")
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for key, ax, title in zip(keys, axes, titles):
        for label, rows in runs:
            _cdf(ax, [r[key] for r in rows if r.get(key) is not None], label)
        ax.set_xscale("log")
        ax.set_xlabel(f"{title} (s)")
        ax.set_ylabel("cumulative fraction of turns")
        ax.set_ylim(0, 1)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.7)
        ax.set_title(title)
    fig.suptitle("Turn-gap decomposition CDFs by max-in-flight "
                 "(turn_gap = tool + scaffold + queue_wait; prefill removed)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_tool_by_name(runs_tools: list[tuple[str, dict[str, list[float]]]],
                     path: Path, top_n: int = 8) -> None:
    """Per-tool-name p50 tool duration, grouped bars by mif — controls for
    composition (same tool across mif)."""
    plt = _mpl()
    # tool names by total call count across runs
    counts: dict[str, int] = {}
    for _label, td in runs_tools:
        for name, v in td.items():
            counts[name] = counts.get(name, 0) + len(v)
    names = [n for n, _ in sorted(counts.items(), key=lambda kv: -kv[1])][:top_n]
    labels = [lb for lb, _ in runs_tools]
    fig, ax = plt.subplots(figsize=(14, 5))
    nbar = len(labels)
    width = 0.8 / max(nbar, 1)
    for j, (label, td) in enumerate(runs_tools):
        ys = [_pct(td.get(n, []), 0.5) for n in names]
        xs = [i + j * width for i in range(len(names))]
        ax.bar(xs, ys, width=width, label=label)
    ax.set_xticks([i + 0.4 - width / 2 for i in range(len(names))])
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("tool duration p50 (s)")
    ax.legend(fontsize=8, framealpha=0.7)
    ax.set_title("Per-tool-name median duration by max-in-flight "
                 "(same tool across mif = contention control)")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", nargs=3, metavar=("LABEL",
                    "PROFILES", "TRACE"), required=True,
                    help="a run to compare: <label> <profiles_dir> "
                         "<trace.jsonl>; repeatable (one per mif)")
    ap.add_argument("--logs", action="append", nargs=2, metavar=("LABEL",
                    "LOGS_DIR"), default=[],
                    help="worker-log dir for a run (SCHED_DELAY queued_ts "
                         "fallback for the client/server split when the "
                         "profile carries no nvext recv_ts); label must "
                         "match a --run label")
    ap.add_argument("--frontend", action="append", nargs=2, metavar=("LABEL",
                    "FRONTEND_LOG"), default=[],
                    help="dynamo frontend.log for a run: supplies ttft_ms so "
                         "the gap can also subtract PREFILL (prefill = ttft - "
                         "queue). Adds `prefill` and `gap_no_server` columns; "
                         "label must match a --run label")
    ap.add_argument("--out", type=Path, default=Path("mif_gap"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    ats = _load("analyze_turn_scheduling", _ATS_PATH)
    e0 = _load("e0_turn_characterization", _E0_PATH)

    logs_by_label = {label: Path(p) for label, p in args.logs}
    frontend_by_label = {label: Path(p) for label, p in args.frontend}
    runs: list[tuple[str, list[dict]]] = []
    runs_tools: list[tuple[str, dict[str, list[float]]]] = []
    for label, prof_s, trace_s in args.run:
        prof, trace = Path(prof_s), Path(trace_s)
        if not prof.is_dir():
            print(f"error: profiles dir not found: {prof}", file=sys.stderr)
            return 2
        if not trace.is_file():
            print(f"error: trace not found: {trace}", file=sys.stderr)
            return 2
        rows = gap_rows(prof, trace, e0, ats, logs=logs_by_label.get(label),
                        frontend=frontend_by_label.get(label))
        runs.append((label, rows))
        runs_tools.append((label,
                           tool_durations(prof, e0.trace_session_ids(trace))))

    args.out.mkdir(parents=True, exist_ok=True)
    # summary table (CSV + stdout)
    with (args.out / "mif_gap_summary.csv").open("w", newline="",
                                                 encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["mif", "component", "n", "p50", "p90", "p99", "mean"])
        for label, rows in runs:
            s = summarize(rows)
            print(f"\n[{label}]  turns={len(rows)}")
            for comp in ("turn_gap", "gap_full", "tool", "scaffold",
                         "queue_wait", "prefill", "host_ub"):
                c = s[comp]
                if c["n"] == 0:
                    if comp in ("queue_wait", "host_ub"):
                        print(f"  {comp:10s} (no SCHED_DELAY match — pass "
                              "--logs and check request_id)")
                    elif comp in ("prefill", "scaffold", "turn_gap"):
                        print(f"  {comp:10s} (needs --frontend ttft_ms "
                              "+ --logs queue)")
                    continue
                print(f"  {comp:10s} p50={c['p50']:.3f}  p90={c['p90']:.3f}  "
                      f"p99={c['p99']:.3f}  mean={c['mean']:.3f}  n={c['n']}")
                w.writerow([label, comp, c["n"], f"{c['p50']:.4f}",
                            f"{c['p90']:.4f}", f"{c['p99']:.4f}",
                            f"{c['mean']:.4f}"])
        print("\nturn_gap = tool + scaffold + queue_wait  (= gap_full - "
              "prefill; prefill is on-GPU compute, not gap)."
              "\n  tool       host: previous step's tool execution."
              "\n  scaffold   host: opencode post-processing + request build "
              "(gap_full - tool - ttft)."
              "\n  queue_wait vLLM engine queue delay (SCHED_DELAY queue_ms)."
              "\n  prefill    on-GPU prompt compute (ttft - queue_wait); "
              "removed from turn_gap."
              "\n  host_ub    fallback scaffold upper bound (gap_full - tool - "
              "queue_wait) when --frontend absent; still includes prefill."
              "\nRead: tool/scaffold up = host contention; queue_wait up = "
              "LLM serving saturation.")

    if not args.no_figures:
        try:
            fig_gap_cdfs(runs, args.out / "fig_gap_cdfs.pdf")
            fig_tool_by_name(runs_tools, args.out / "fig_tool_by_name.pdf")
        except ImportError:
            print("matplotlib not available (CSV only)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
