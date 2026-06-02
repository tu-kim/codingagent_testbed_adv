#!/usr/bin/env python3
"""Compare N testbed trace.jsonl runs for reproducibility, per session.

Pass 2, 3, 4, ... runs. Each run's TaskRecords are joined by `instance_id`
(NOT session_id -- opencode mints a fresh `ses_...` every run, so it
always differs; instance_id is the stable per-sample key). Per session,
across the runs that contain it, we extract and compare:

    rtt_s              wall time (always varies a little -- informational)
    n_turns            LLM steps (step-start count) -- identical if reproducible
    n_tool_calls       tool invocations
    in/out/cache tokens (output_tokens is the load-bearing signal)
    tool_sequence      ordered tool names -- trajectory shape
    final answer       last assistant message text  ┐ the "정답": did every
    diffs              files changed (summary.diffs) ┘ run reach the same result?
    trajectory         full ordered (text + tool name + tool input)

Per session STATUS (across the runs that have it):
    REPRODUCIBLE          one distinct trajectory across all runs
    TRAJ_DIFF_SAME_ANSWER trajectories differ but final answer + code diff
                          are identical in every run (converged anyway)
    ANSWER_DIFF           >1 distinct final answer / code diff -- runs
                          produced DIFFERENT results
    INSUFFICIENT          present in <2 runs (nothing to compare)
A session is also flagged `complete` only if present in ALL runs.

Outputs:
  per_instance.csv   one row per instance_id: distinct counts + ranges + status
  runs_summary.csv   one row per run: n_sessions, total turns/output tokens, mean rtt
  fig_*.pdf          only with --figures (needs matplotlib):
                       fig_status_breakdown        status counts (bar)
                       fig_turns_spread            per-instance turn spread (max-min)
                       fig_rtt_by_run              rtt distribution per run (box)
  stdout             status counts + the first divergent sessions

Exit code 3 if any session is ANSWER_DIFF (so a reproducibility CI gate
can detect non-determinism).

Usage:
  scripts/compare_traces.py \\
      --traces results/run1/trace.jsonl results/run2/trace.jsonl results/run3/trace.jsonl \\
      --output results/repro_cmp [--labels run1 run2 run3] [--figures]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


# ---------- per-session extraction ----------


@dataclass
class SessionMetrics:
    instance_id: str
    session_id: str | None
    success: bool
    rtt_s: float | None
    n_turns: int
    n_tool_calls: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    tool_sequence: list[str]
    diff_signature: list[tuple[str, int, int]]   # (file, additions, deletions), sorted
    final_text: str
    trajectory: str

    def answer_key(self) -> tuple[str, tuple]:
        """The 'did it reach the same result' identity: final answer text
        + the set of code changes."""
        return (self.final_text, tuple(self.diff_signature))

    def trajectory_hash(self) -> str:
        return _sha(self.trajectory)


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:12]


def _tokens(info: dict) -> tuple[int, int, int]:
    # Defensive against non-dict shapes from the raw trace (same class of
    # bug as info.summary being a bool): `x or {}` would leave a truthy
    # non-dict in place and then .get() would crash.
    t = info.get("tokens")
    if not isinstance(t, dict):
        t = {}
    cache = t.get("cache")
    if not isinstance(cache, dict):
        cache = {}
    def _i(v) -> int:
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else 0
    return _i(t.get("input")), _i(t.get("output")), _i(cache.get("read"))


def _extract_diffs(messages: list) -> list[tuple[str, int, int]]:
    for m in messages:
        # info.summary is a DICT (with `diffs`) only on the user message;
        # on assistant messages opencode sets it to a BOOLEAN flag, so we
        # must skip non-dict values rather than call .get() on a bool.
        summary = ((m or {}).get("info") or {}).get("summary")
        if not isinstance(summary, dict):
            continue
        diffs = summary.get("diffs")
        if isinstance(diffs, list) and diffs:
            out = []
            for d in diffs:
                if not isinstance(d, dict):
                    continue
                out.append((
                    str(d.get("file", "")),
                    int(d.get("additions", 0) or 0),
                    int(d.get("deletions", 0) or 0),
                ))
            return sorted(out)
    return []


def extract_session(rec: dict) -> SessionMetrics:
    messages = rec.get("messages") or []
    n_steps = n_assistant = n_tool_calls = 0
    in_tok = out_tok = cache_tok = 0
    tool_seq: list[str] = []
    traj_parts: list[str] = []
    last_assistant_text = ""

    for m in messages:
        info = (m or {}).get("info") or {}
        if info.get("role") != "assistant":
            continue
        n_assistant += 1
        i, o, c = _tokens(info)
        in_tok += i
        out_tok += o
        cache_tok += c
        msg_text_parts: list[str] = []
        for p in (m.get("parts") or []):
            ptype = p.get("type")
            if ptype == "step-start":
                n_steps += 1
            elif ptype == "text":
                txt = p.get("text") or ""
                traj_parts.append("T:" + txt)
                msg_text_parts.append(txt)
            elif ptype == "tool":
                name = p.get("tool") or "?"
                n_tool_calls += 1
                tool_seq.append(name)
                tool_input = (p.get("state") or {}).get("input")
                traj_parts.append("C:" + name + ":" +
                                  json.dumps(tool_input, sort_keys=True, default=str))
        if msg_text_parts:
            last_assistant_text = "".join(msg_text_parts)

    return SessionMetrics(
        instance_id=rec.get("instance_id") or "?",
        session_id=rec.get("session_id"),
        success=bool(rec.get("success")),
        rtt_s=rec.get("rtt_s"),
        n_turns=n_steps if n_steps else n_assistant,
        n_tool_calls=n_tool_calls,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_tokens=cache_tok,
        tool_sequence=tool_seq,
        diff_signature=_extract_diffs(messages),
        final_text=last_assistant_text,
        trajectory="\n".join(traj_parts),
    )


def load_trace(path: Path) -> dict[str, SessionMetrics]:
    out: dict[str, SessionMetrics] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sm = extract_session(rec)
            out[sm.instance_id] = sm
    return out


# ---------- N-way comparison ----------


@dataclass
class InstanceComparison:
    instance_id: str
    labels: list[str]              # run labels that contain this instance
    metrics: list[SessionMetrics]  # parallel to labels
    n_runs: int                    # total runs being compared

    @property
    def complete(self) -> bool:
        return len(self.metrics) == self.n_runs

    @property
    def distinct_trajectories(self) -> int:
        return len({m.trajectory for m in self.metrics})

    @property
    def distinct_answers(self) -> int:
        return len({m.answer_key() for m in self.metrics})

    @property
    def turns(self) -> list[int]:
        return [m.n_turns for m in self.metrics]

    @property
    def output_tokens(self) -> list[int]:
        return [m.output_tokens for m in self.metrics]

    @property
    def rtts(self) -> list[float]:
        return [m.rtt_s for m in self.metrics if m.rtt_s is not None]

    @property
    def status(self) -> str:
        if len(self.metrics) < 2:
            return "INSUFFICIENT"
        if self.distinct_trajectories == 1:
            return "REPRODUCIBLE"
        if self.distinct_answers == 1:
            return "TRAJ_DIFF_SAME_ANSWER"
        return "ANSWER_DIFF"


def compare(runs: list[tuple[str, dict[str, SessionMetrics]]]) -> list[InstanceComparison]:
    n = len(runs)
    all_ids = sorted({i for _, d in runs for i in d})
    out: list[InstanceComparison] = []
    for iid in all_ids:
        labels, metrics = [], []
        for label, d in runs:
            if iid in d:
                labels.append(label)
                metrics.append(d[iid])
        out.append(InstanceComparison(instance_id=iid, labels=labels,
                                      metrics=metrics, n_runs=n))
    return out


# ---------- output ----------


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def _spread(xs: list) -> tuple:
    return (min(xs), max(xs)) if xs else (None, None)


def write_per_instance_csv(comps: list[InstanceComparison], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "instance_id", "status", "complete", "n_runs_present",
            "distinct_trajectories", "distinct_answers",
            "distinct_turns", "turns_min", "turns_max",
            "distinct_output_tokens", "out_tokens_min", "out_tokens_max",
            "rtt_min", "rtt_max",
        ])
        for c in comps:
            tmin, tmax = _spread(c.turns)
            omin, omax = _spread(c.output_tokens)
            rmin, rmax = _spread(c.rtts)
            w.writerow([
                c.instance_id, c.status, _fmt(c.complete), len(c.metrics),
                c.distinct_trajectories, c.distinct_answers,
                len(set(c.turns)), _fmt(tmin), _fmt(tmax),
                len(set(c.output_tokens)), _fmt(omin), _fmt(omax),
                _fmt(rmin), _fmt(rmax),
            ])


def write_runs_summary_csv(runs: list[tuple[str, dict[str, SessionMetrics]]],
                           path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "n_sessions", "total_turns",
                    "total_output_tokens", "mean_rtt_s"])
        for label, d in runs:
            ms = list(d.values())
            rtts = [m.rtt_s for m in ms if m.rtt_s is not None]
            w.writerow([
                label, len(ms),
                sum(m.n_turns for m in ms),
                sum(m.output_tokens for m in ms),
                _fmt(sum(rtts) / len(rtts)) if rtts else "",
            ])


def print_summary(comps: list[InstanceComparison], n_runs: int) -> None:
    counts = Counter(c.status for c in comps)
    print()
    print(f"Runs compared: {n_runs}   sessions (union): {len(comps)}")
    for st in ("REPRODUCIBLE", "TRAJ_DIFF_SAME_ANSWER", "ANSWER_DIFF", "INSUFFICIENT"):
        if counts.get(st):
            print(f"  {st:<22} {counts[st]}")
    incomplete = [c for c in comps if not c.complete]
    if incomplete:
        print(f"  (not present in all {n_runs} runs: {len(incomplete)})")

    comparable = [c for c in comps if len(c.metrics) >= 2]
    if comparable:
        repro = sum(1 for c in comparable if c.status == "REPRODUCIBLE")
        ans_ok = sum(1 for c in comparable if c.distinct_answers == 1)
        print()
        print(f"Comparable: {len(comparable)}  |  fully reproducible: {repro} "
              f"({100*repro/len(comparable):.0f}%)  |  same final answer: {ans_ok} "
              f"({100*ans_ok/len(comparable):.0f}%)")

    diverged = [c for c in comps if c.status == "ANSWER_DIFF"]
    if diverged:
        print()
        print("ANSWER_DIFF sessions (final text and/or code diff differ across runs):")
        for c in diverged[:20]:
            print(f"  {c.instance_id:<40} turns={c.turns} out_tokens={c.output_tokens}")
    print()


# ---------- figures (opt-in; matplotlib lazy) ----------

_PAPER_STYLE = {
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 10, "figure.figsize": (3.6, 2.4), "figure.dpi": 150,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
}


def _setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(_PAPER_STYLE)
    return plt


def plot_status_breakdown(comps: list[InstanceComparison], out: Path) -> Path | None:
    if not comps:
        return None
    plt = _setup_plt()
    order = ["REPRODUCIBLE", "TRAJ_DIFF_SAME_ANSWER", "ANSWER_DIFF", "INSUFFICIENT"]
    counts = Counter(c.status for c in comps)
    labels = [s for s in order if counts.get(s)]
    vals = [counts[s] for s in labels]
    colors = {"REPRODUCIBLE": "#31a354", "TRAJ_DIFF_SAME_ANSWER": "#fd8d3c",
              "ANSWER_DIFF": "#de2d26", "INSUFFICIENT": "#999999"}
    fig, ax = plt.subplots()
    ax.bar(range(len(labels)), vals, color=[colors[s] for s in labels])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([s.replace("_", "\n") for s in labels], fontsize=7)
    ax.set_ylabel("sessions")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    p = out / "fig_status_breakdown.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_turns_spread(comps: list[InstanceComparison], out: Path) -> Path | None:
    """Per-instance (max-min) turn count across runs, sorted descending.
    Bars > 0 are sessions whose turn count was NOT reproducible."""
    comparable = [c for c in comps if len(c.metrics) >= 2 and c.turns]
    if not comparable:
        return None
    plt = _setup_plt()
    spreads = sorted((max(c.turns) - min(c.turns) for c in comparable), reverse=True)
    fig, ax = plt.subplots()
    ax.bar(range(len(spreads)), spreads, color="#3182bd", width=1.0)
    ax.set_xlabel("session (sorted)")
    ax.set_ylabel("turn-count spread (max - min)")
    p = out / "fig_turns_spread.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_rtt_by_run(runs: list[tuple[str, dict[str, SessionMetrics]]],
                    out: Path) -> Path | None:
    data, labels = [], []
    for label, d in runs:
        rtts = [m.rtt_s for m in d.values() if m.rtt_s is not None]
        if rtts:
            data.append(rtts)
            labels.append(label)
    if not data:
        return None
    plt = _setup_plt()
    fig, ax = plt.subplots()
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("rtt (s)")
    ax.tick_params(axis="x", labelrotation=30, labelsize=7)
    p = out / "fig_rtt_by_run.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def make_figures(comps: list[InstanceComparison],
                 runs: list[tuple[str, dict[str, SessionMetrics]]],
                 out: Path) -> list[Path]:
    paths = []
    for fn, arg in ((plot_status_breakdown, comps),
                    (plot_turns_spread, comps)):
        p = fn(arg, out)
        if p is not None:
            paths.append(p)
    p = plot_rtt_by_run(runs, out)
    if p is not None:
        paths.append(p)
    return paths


# ---------- main ----------


def _label_for(path: Path) -> str:
    """results/run1/trace.jsonl -> 'run1'; otherwise the file stem."""
    if path.name == "trace.jsonl":
        return path.parent.name or path.stem
    return path.stem


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--traces", required=True, nargs="+", type=Path,
                    help="Two or more trace.jsonl files to compare")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="Optional run labels (must match --traces count)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory (created if missing)")
    ap.add_argument("--figures", action="store_true",
                    help="Also render PDF figures (needs matplotlib)")
    args = ap.parse_args(argv)

    if len(args.traces) < 2:
        print("need at least 2 traces to compare", file=sys.stderr)
        return 2
    for p in args.traces:
        if not p.exists():
            print(f"trace not found: {p}", file=sys.stderr)
            return 2
    if args.labels is not None and len(args.labels) != len(args.traces):
        print("--labels count must match --traces count", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    labels = args.labels or [_label_for(p) for p in args.traces]
    # Disambiguate duplicate labels (e.g. two trace.jsonl under same-named dirs).
    seen: Counter = Counter()
    uniq_labels = []
    for lb in labels:
        seen[lb] += 1
        uniq_labels.append(lb if seen[lb] == 1 else f"{lb}#{seen[lb]}")

    runs = [(lb, load_trace(p)) for lb, p in zip(uniq_labels, args.traces)]
    if all(not d for _, d in runs):
        print("all traces empty / unparseable", file=sys.stderr)
        return 1

    comps = compare(runs)
    per_inst = args.output / "per_instance.csv"
    runs_csv = args.output / "runs_summary.csv"
    write_per_instance_csv(comps, per_inst)
    write_runs_summary_csv(runs, runs_csv)
    print(f"  wrote {per_inst}")
    print(f"  wrote {runs_csv}")

    if args.figures:
        for p in make_figures(comps, runs, args.output):
            print(f"  wrote {p}")

    print_summary(comps, len(runs))

    if any(c.status == "ANSWER_DIFF" for c in comps):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
