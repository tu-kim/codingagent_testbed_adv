#!/usr/bin/env python3
"""Compare the turn-GAP decomposition across max-in-flight (mif) settings
to locate the cause of gap inflation:

  gap    = away_s(N) = llm.start(N) - llm.end(N-1)   (off-GPU time)
  tool   = tool execution wall of the PREVIOUS step (the tools that ran
           in this gap)                              -> CPU-contention signal
  others = max(0, gap - tool)                        -> scaffold / post-
           processing (opencode snapshot, DB writes, git, event loop)

If tool time itself inflates with mif -> CPU contention on tool execution.
If only `others` inflates -> opencode post-processing / event-loop
contention (the single server servicing more sessions).

CRITICAL: across mif settings the processing ORDER differs and the run is
non-deterministic, so the SAME session's turns / llm / tool internals all
differ. We therefore compare DISTRIBUTIONS (percentiles, CDFs), never
paired per-turn, and additionally break tool time out PER TOOL NAME so a
mif that merely happened to run more expensive tools is not mistaken for
contention.

Usage:
  scripts/arm/compare_mif_gap.py \\
      --run mif1 <profiles_dir> <trace.jsonl> \\
      --run mif4 <profiles_dir> <trace.jsonl> \\
      [--out <dir>] [--no-figures]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

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


def gap_rows(profiles: Path, trace: Path, e0, ats) -> list[dict]:
    """Per-turn gap decomposition for the run's MAIN sessions.

    gap = tool + others, with `tool` = the PREVIOUS step's tool wall.

    `others` is further split using dynamo's server-side receive timestamp
    (llm.end.dynamo.request_received_unix_s of THIS turn, recv_ts):
      client_s = recv_ts - llm.end(N-1)   # true client-side scaffold window
                                          # (post-processing, event loop,
                                          # request build) up to the moment
                                          # the server RECEIVED the request
      server_s = llm.start(N) - recv_ts   # request was already AT the server
                                          # but the profile hook hadn't fired:
                                          # frontend queue + engine queue +
                                          # prefill leaking into the gap
                                          # (start-step is consumed only when
                                          # the response stream begins)
    A large server_s means the gap inflation is NOT host contention at all
    but LLM-side queueing miscounted into the gap. None when recv_ts is
    missing (non-dynamo or unpatched runs)."""
    turns = ats.load_turns(profiles)
    ids = e0.trace_session_ids(trace)
    turns = [t for t in turns if t.session_id in ids]
    by_key = {(t.session_id, t.step): t for t in turns}
    rows: list[dict] = []
    for t in turns:
        if t.away_s is None:
            continue
        prev = by_key.get((t.session_id, t.step - 1))
        tool = (getattr(prev, "tool_time_s", None) or 0.0) if prev else 0.0
        gap = t.away_s
        client_s = server_s = None
        if (t.recv_ts is not None and t.llm_start_ts is not None
                and prev is not None and prev.llm_end_ts is not None):
            client_s = max(0.0, t.recv_ts - prev.llm_end_ts)
            server_s = max(0.0, t.llm_start_ts - t.recv_ts)
        rows.append({"gap": gap, "tool": tool,
                     "others": max(0.0, gap - tool),
                     "client": client_s, "server": server_s})
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
    for k in ("gap", "tool", "others", "client", "server"):
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
    plt = _mpl()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for key, ax, title in zip(("gap", "tool", "others"), axes,
                              ("turn gap", "tool time (prev step)",
                               "others (gap - tool)")):
        for label, rows in runs:
            _cdf(ax, [r[key] for r in rows], label)
        ax.set_xscale("log")
        ax.set_xlabel(f"{title} (s)")
        ax.set_ylabel("cumulative fraction of turns")
        ax.set_ylim(0, 1)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, loc="lower right", framealpha=0.7)
        ax.set_title(title)
    fig.suptitle("Turn-gap decomposition CDFs by max-in-flight")
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
    ap.add_argument("--out", type=Path, default=Path("mif_gap"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    ats = _load("analyze_turn_scheduling", _ATS_PATH)
    e0 = _load("e0_turn_characterization", _E0_PATH)

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
        rows = gap_rows(prof, trace, e0, ats)
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
            for comp in ("gap", "tool", "others", "client", "server"):
                c = s[comp]
                if c["n"] == 0:
                    if comp in ("client", "server"):
                        print(f"  {comp:7s} (no recv_ts — dynamo nvext "
                              "timing absent)")
                    continue
                print(f"  {comp:7s} p50={c['p50']:.3f}  p90={c['p90']:.3f}  "
                      f"p99={c['p99']:.3f}  mean={c['mean']:.3f}  n={c['n']}")
                w.writerow([label, comp, c["n"], f"{c['p50']:.4f}",
                            f"{c['p90']:.4f}", f"{c['p99']:.4f}",
                            f"{c['mean']:.4f}"])
        print("\nclient = llm.end(N-1) -> server RECEIVED request (true "
              "host-side scaffold+tool window)\nserver = request at server "
              "-> llm.start fired (LLM queue/prefill leaking into the gap)")

    if not args.no_figures:
        try:
            fig_gap_cdfs(runs, args.out / "fig_gap_cdfs.pdf")
            fig_tool_by_name(runs_tools, args.out / "fig_tool_by_name.pdf")
        except ImportError:
            print("matplotlib not available (CSV only)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
