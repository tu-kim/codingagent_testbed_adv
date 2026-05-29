#!/usr/bin/env python3
"""Per-turn sub-agent (task tool) time share from opencode profile NDJSON.

For every turn (one LLM step inside a session) this reports:
  - turn_duration_s   total wall time of the turn (turn.end.duration_s)
  - subagent_wall_s   wall time blocked on `task` tool calls in that turn
  - subagent_sum_s    naive SUM of task-call durations (overcounts when
                      multiple sub-agents ran in parallel)
  - ratio             subagent_wall_s / turn_duration_s

Why wall vs sum: the `task` tool spawns a nested opencode session and
blocks the parent until it finishes, so a single task call's duration
IS the sub-agent's execution time. When the model fires >1 task in one
step they run concurrently -- naively summing their durations would
exceed the turn's wall clock. We therefore compute the UNION of the
[tool.start, tool.end] intervals for `subagent_wall_s` (the real
blocked time) and also expose the raw sum so you can see the
parallelism gap (sum > wall ⇒ overlap).

Event sources (from the opencode profile patch):
  turn.end   {sessionID, step, duration_s, llm_wall_s, tool_wall_s, ...}
  tool.start {sessionID, step, callID, name, ts}
  tool.end   {sessionID, step, callID, name, ok, duration_s, ts}
`ts` is unix seconds; `duration_s` is seconds.

Note: a task tool's nested child session has its OWN sessionID and
shows up as separate turns (typically ratio 0 -- they don't call task
themselves). The parent turn's task duration already encapsulates the
full child execution, so we do NOT double-count; each (session, step)
is reported independently.

Outputs:
  subagent_time.csv   one row per turn
  stdout              summary: overall ratio, per-session rollup, top turns

Usage:
  scripts/analyze_subagent_time.py \\
      --profile /tmp/testbed-workspaces/profiles \\
      --output results/run1/subagent_time
  # --profile accepts either a directory of <sessionID>.jsonl files or a
  # single aggregated NDJSON file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------- model ----------


@dataclass
class _ToolInterval:
    call_id: str
    name: str
    start_ts: float | None = None
    end_ts: float | None = None
    duration_s: float | None = None
    ok: bool = True

    def interval(self) -> tuple[float, float] | None:
        """Best-effort [start, end] in unix seconds. Reconstruct the
        missing endpoint from duration_s when only one ts is present.
        Returns None when neither a usable pair nor a duration exists."""
        s, e, d = self.start_ts, self.end_ts, self.duration_s
        if s is not None and e is not None:
            return (s, e) if e >= s else (e, s)
        if s is not None and d is not None:
            return (s, s + d)
        if e is not None and d is not None:
            return (e - d, e)
        return None


@dataclass
class _Turn:
    session_id: str
    step: int
    # From turn.end.duration_s. The profile patch ALWAYS emits duration_s
    # on turn.end, so there's no ts-based fallback: a turn with no
    # turn.end (truncated/crashed mid-step) keeps this None, which
    # correctly yields a None ratio downstream rather than a fabricated
    # number.
    turn_duration_s: float | None = None
    tools: dict[str, _ToolInterval] = field(default_factory=dict)

    def _ensure(self, call_id: str, name: str) -> _ToolInterval:
        ti = self.tools.get(call_id)
        if ti is None:
            ti = _ToolInterval(call_id=call_id, name=name)
            self.tools[call_id] = ti
        return ti

    def total_duration(self) -> float | None:
        return self.turn_duration_s


@dataclass
class TurnSummary:
    session_id: str
    step: int
    turn_duration_s: float | None
    subagent_wall_s: float
    subagent_sum_s: float
    n_task: int
    n_task_failed: int

    @property
    def ratio(self) -> float | None:
        if self.turn_duration_s is None or self.turn_duration_s <= 0:
            return None
        return self.subagent_wall_s / self.turn_duration_s


# ---------- ingest ----------


def _iter_event_files(path: Path):
    if path.is_file():
        yield path
        return
    for f in sorted(path.glob("*.jsonl")):
        yield f


def load_turns(path: Path) -> list[_Turn]:
    """Parse profile NDJSON into per-(sessionID, step) turns carrying
    their task-tool intervals. Malformed lines are skipped."""
    turns: dict[tuple[str, int], _Turn] = {}

    def ensure(sid: str, step: int) -> _Turn:
        key = (sid, step)
        t = turns.get(key)
        if t is None:
            t = _Turn(session_id=sid, step=step)
            turns[key] = t
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
                etype = ev.get("ev")
                sid = ev.get("sessionID")
                step = ev.get("step")
                if not etype or not sid or step is None:
                    # query.start/query.end have no step; we don't need them.
                    continue
                ts = ev.get("ts")

                if etype == "turn.end":
                    ensure(sid, step).turn_duration_s = ev.get("duration_s")
                elif etype == "tool.start":
                    call_id = ev.get("callID")
                    name = ev.get("name") or "?"
                    if not call_id:
                        continue
                    ti = ensure(sid, step)._ensure(call_id, name)
                    ti.start_ts = ts
                    ti.name = name
                elif etype == "tool.end":
                    call_id = ev.get("callID")
                    name = ev.get("name") or "?"
                    if not call_id:
                        continue
                    ti = ensure(sid, step)._ensure(call_id, name)
                    ti.end_ts = ts
                    ti.duration_s = ev.get("duration_s")
                    ti.ok = bool(ev.get("ok", True))
                    if name != "?":
                        ti.name = name

    return list(turns.values())


# ---------- compute ----------


def union_length(intervals: list[tuple[float, float]]) -> float:
    """Total length covered by the union of [start, end] intervals.
    Overlapping intervals (parallel sub-agents) count once."""
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    cur_s, cur_e = ordered[0]
    for s, e in ordered[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def summarize_turn(t: _Turn, task_name: str = "task") -> TurnSummary:
    task_tools = [ti for ti in t.tools.values() if ti.name == task_name]
    intervals: list[tuple[float, float]] = []
    sub_sum = 0.0
    n_failed = 0
    for ti in task_tools:
        iv = ti.interval()
        if iv is not None:
            intervals.append(iv)
            sub_sum += iv[1] - iv[0]
        if not ti.ok:
            n_failed += 1
    return TurnSummary(
        session_id=t.session_id,
        step=t.step,
        turn_duration_s=t.total_duration(),
        subagent_wall_s=union_length(intervals),
        subagent_sum_s=sub_sum,
        n_task=len(task_tools),
        n_task_failed=n_failed,
    )


def summarize(turns: list[_Turn], task_name: str = "task") -> list[TurnSummary]:
    rows = [summarize_turn(t, task_name=task_name) for t in turns]
    # Stable, human-friendly ordering: by session then step.
    rows.sort(key=lambda r: (r.session_id, r.step))
    return rows


# ---------- output ----------


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.4f}"


def write_csv(rows: list[TurnSummary], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session_id", "step", "turn_duration_s",
            "subagent_wall_s", "subagent_sum_s",
            "n_task", "n_task_failed", "ratio",
        ])
        for r in rows:
            w.writerow([
                r.session_id, r.step, _fmt(r.turn_duration_s),
                _fmt(r.subagent_wall_s), _fmt(r.subagent_sum_s),
                r.n_task, r.n_task_failed, _fmt(r.ratio),
            ])


def print_summary(rows: list[TurnSummary]) -> None:
    turns_with_task = [r for r in rows if r.n_task > 0]
    parallel_turns = [r for r in turns_with_task
                      if r.subagent_sum_s - r.subagent_wall_s > 1e-6]

    total_turn = sum(r.turn_duration_s for r in rows if r.turn_duration_s)
    total_sub = sum(r.subagent_wall_s for r in rows)
    agg_ratio = (total_sub / total_turn) if total_turn else 0.0

    print()
    print(f"Turns total:                 {len(rows)}")
    print(f"Turns with >=1 task call:    {len(turns_with_task)}")
    print(f"Turns with PARALLEL tasks:   {len(parallel_turns)} "
          f"(sub-agent sum > wall ⇒ overlap)")
    print(f"Aggregate turn wall:         {total_turn:.2f}s")
    print(f"Aggregate sub-agent wall:    {total_sub:.2f}s")
    print(f"Aggregate sub-agent ratio:   {100 * agg_ratio:.1f}%")
    print()

    if turns_with_task:
        print("Top turns by sub-agent ratio:")
        hdr = (f"{'session_id':<34} {'step':>4} {'turn_s':>9} "
               f"{'sub_wall':>9} {'sub_sum':>9} {'n':>3} {'ratio':>7}")
        print(hdr)
        print("-" * len(hdr))
        ranked = sorted(
            turns_with_task,
            key=lambda r: (r.ratio if r.ratio is not None else -1.0),
            reverse=True,
        )[:15]
        for r in ranked:
            ratio_s = f"{100 * r.ratio:.1f}%" if r.ratio is not None else "-"
            sid = (r.session_id[:31] + "...") if len(r.session_id) > 34 else r.session_id
            print(f"{sid:<34} {r.step:>4} {_fmt(r.turn_duration_s):>9} "
                  f"{r.subagent_wall_s:>9.4f} {r.subagent_sum_s:>9.4f} "
                  f"{r.n_task:>3} {ratio_s:>7}")
        print()


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", required=True, type=Path,
                    help="Profile NDJSON dir (<sessionID>.jsonl files) or a "
                         "single aggregated NDJSON file")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory (created if missing)")
    ap.add_argument("--task-tool-name", default="task",
                    help="Tool name treated as the sub-agent spawn "
                         "(default: task)")
    args = ap.parse_args(argv)

    if not args.profile.exists():
        print(f"profile path not found: {args.profile}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    turns = load_turns(args.profile)
    if not turns:
        print("no turns found in profile NDJSON", file=sys.stderr)
        return 1

    rows = summarize(turns, task_name=args.task_tool_name)
    csv_path = args.output / "subagent_time.csv"
    write_csv(rows, csv_path)
    print(f"  wrote {csv_path}")
    print_summary(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
