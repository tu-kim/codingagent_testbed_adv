#!/usr/bin/env python3
"""Per-turn NON-sub-agent tool time share from opencode profile NDJSON.

The complement of analyze_subagent_time.py: instead of the `task`
(sub-agent) tool, this measures how much of each turn's wall time was
spent on the ORDINARY tools (read / grep / edit / bash / glob / ...).

For every turn (one LLM step inside a session) it reports:
  - turn_duration_s   total wall time of the turn (turn.end.duration_s)
  - tool_wall_s       wall time blocked on non-excluded tools (UNION of
                      their [tool.start, tool.end] intervals -- parallel
                      tool calls count once, so the ratio never exceeds 1)
  - tool_sum_s        naive SUM of those tool durations (sum > wall ⇒
                      tools overlapped, i.e. ran in parallel)
  - ratio             tool_wall_s / turn_duration_s

By default the `task` tool is excluded (it's a sub-agent spawn, covered
by analyze_subagent_time.py). Override with --exclude.

A second CSV breaks the wall time down BY TOOL NAME (read vs grep vs
edit ...). Note: per-tool unions are computed independently, so if two
different tool names overlap in time their per-tool walls both count
that overlap -- the per-tool numbers can therefore sum to more than the
turn's combined tool_wall_s. That's intentional: each row answers "how
much wall time involved tool X", not "X's exclusive slice".

Event sources (opencode profile patch):
  turn.end   {sessionID, step, duration_s}
  tool.start {sessionID, step, callID, name, ts}
  tool.end   {sessionID, step, callID, name, ok, duration_s, ts}
`ts` is unix seconds; `duration_s` is seconds.

Outputs:
  tool_time_per_turn.csv   one row per turn (aggregate non-task wall)
  tool_time_by_name.csv    one row per tool name (total wall across turns)
  stdout                   summary: overall ratio + per-tool breakdown

Usage:
  scripts/analyze_tool_time.py \\
      --profile /tmp/testbed-workspaces/profiles \\
      --output results/run1/tool_time
  # --exclude task,bash   to also drop bash, etc.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
        """Best-effort [start, end] in unix seconds, reconstructing the
        missing endpoint from duration_s. None when unrecoverable."""
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
    # turn.end.duration_s is the authoritative turn wall; read directly.
    # A truncated turn (no turn.end) stays None and yields a None ratio
    # rather than a fabricated number.
    turn_duration_s: float | None = None
    tools: dict[str, _ToolInterval] = field(default_factory=dict)

    def _ensure(self, call_id: str, name: str) -> _ToolInterval:
        ti = self.tools.get(call_id)
        if ti is None:
            ti = _ToolInterval(call_id=call_id, name=name)
            self.tools[call_id] = ti
        return ti


@dataclass
class TurnToolSummary:
    session_id: str
    step: int
    turn_duration_s: float | None
    tool_wall_s: float                       # union over all non-excluded tools
    tool_sum_s: float                        # naive sum (shows overlap)
    n_calls: int
    per_tool_wall: dict[str, float] = field(default_factory=dict)

    @property
    def ratio(self) -> float | None:
        if self.turn_duration_s is None or self.turn_duration_s <= 0:
            return None
        return self.tool_wall_s / self.turn_duration_s


# ---------- ingest ----------


def _iter_event_files(path: Path):
    if path.is_file():
        yield path
        return
    for f in sorted(path.glob("*.jsonl")):
        yield f


def load_turns(path: Path) -> list[_Turn]:
    """Parse profile NDJSON into per-(sessionID, step) turns carrying
    their tool intervals. Malformed lines are skipped."""
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
    Overlapping intervals (parallel tool calls) count once."""
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


def summarize_turn(t: _Turn, exclude: frozenset[str]) -> TurnToolSummary:
    selected = [ti for ti in t.tools.values() if ti.name not in exclude]
    all_intervals: list[tuple[float, float]] = []
    per_tool_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    tool_sum = 0.0
    for ti in selected:
        iv = ti.interval()
        if iv is None:
            continue
        all_intervals.append(iv)
        per_tool_intervals[ti.name].append(iv)
        tool_sum += iv[1] - iv[0]
    return TurnToolSummary(
        session_id=t.session_id,
        step=t.step,
        turn_duration_s=t.turn_duration_s,
        tool_wall_s=union_length(all_intervals),
        tool_sum_s=tool_sum,
        n_calls=len(selected),
        per_tool_wall={name: union_length(ivs)
                       for name, ivs in per_tool_intervals.items()},
    )


def summarize(turns: list[_Turn], exclude: frozenset[str]) -> list[TurnToolSummary]:
    rows = [summarize_turn(t, exclude) for t in turns]
    rows.sort(key=lambda r: (r.session_id, r.step))
    return rows


def by_tool_name(rows: list[TurnToolSummary]) -> dict[str, tuple[float, int]]:
    """Aggregate {tool_name: (total_wall_s, n_turns_used)} across turns."""
    wall: dict[str, float] = defaultdict(float)
    turns_used: dict[str, int] = defaultdict(int)
    for r in rows:
        for name, w in r.per_tool_wall.items():
            wall[name] += w
            turns_used[name] += 1
    return {name: (wall[name], turns_used[name]) for name in wall}


# ---------- output ----------


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.4f}"


def write_per_turn_csv(rows: list[TurnToolSummary], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session_id", "step", "turn_duration_s",
            "tool_wall_s", "tool_sum_s", "n_calls", "ratio",
        ])
        for r in rows:
            w.writerow([
                r.session_id, r.step, _fmt(r.turn_duration_s),
                _fmt(r.tool_wall_s), _fmt(r.tool_sum_s),
                r.n_calls, _fmt(r.ratio),
            ])


def write_by_name_csv(agg: dict[str, tuple[float, int]], path: Path) -> None:
    total_wall = sum(w for w, _ in agg.values())
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool_name", "total_wall_s", "n_turns_used", "pct_of_tool_wall"])
        for name in sorted(agg, key=lambda n: -agg[n][0]):
            wall, n = agg[name]
            pct = (100.0 * wall / total_wall) if total_wall else 0.0
            w.writerow([name, _fmt(wall), n, f"{pct:.2f}"])


def print_summary(rows: list[TurnToolSummary],
                  agg: dict[str, tuple[float, int]],
                  exclude: frozenset[str]) -> None:
    turns_with_tool = [r for r in rows if r.n_calls > 0]
    parallel_turns = [r for r in turns_with_tool
                      if r.tool_sum_s - r.tool_wall_s > 1e-6]
    total_turn = sum(r.turn_duration_s for r in rows if r.turn_duration_s)
    total_tool = sum(r.tool_wall_s for r in rows)
    agg_ratio = (total_tool / total_turn) if total_turn else 0.0

    print()
    print(f"Excluded tools:              {sorted(exclude)}")
    print(f"Turns total:                 {len(rows)}")
    print(f"Turns with >=1 non-excl tool:{len(turns_with_tool)}")
    print(f"Turns with PARALLEL tools:   {len(parallel_turns)} "
          f"(tool sum > wall ⇒ overlap)")
    print(f"Aggregate turn wall:         {total_turn:.2f}s")
    print(f"Aggregate non-task tool wall:{total_tool:.2f}s")
    print(f"Aggregate non-task ratio:    {100 * agg_ratio:.1f}%")
    print()

    if agg:
        total_wall = sum(w for w, _ in agg.values())
        print("Wall time by tool name (union per turn, summed):")
        hdr = f"{'tool':<16} {'wall_s':>10} {'turns':>6} {'pct':>7}"
        print(hdr)
        print("-" * len(hdr))
        for name in sorted(agg, key=lambda n: -agg[n][0]):
            wall, n = agg[name]
            pct = (100.0 * wall / total_wall) if total_wall else 0.0
            print(f"{name:<16} {wall:>10.4f} {n:>6} {pct:>6.1f}%")
        print()


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", required=True, type=Path,
                    help="Profile NDJSON dir (<sessionID>.jsonl) or single file")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory (created if missing)")
    ap.add_argument("--exclude", default="task",
                    help="Comma-separated tool names to exclude from the "
                         "ratio (default: task -- the sub-agent spawn). "
                         "Pass an empty string to include every tool.")
    args = ap.parse_args(argv)

    if not args.profile.exists():
        print(f"profile path not found: {args.profile}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    exclude = frozenset(n for n in args.exclude.split(",") if n)

    turns = load_turns(args.profile)
    if not turns:
        print("no turns found in profile NDJSON", file=sys.stderr)
        return 1

    rows = summarize(turns, exclude)
    agg = by_tool_name(rows)

    per_turn_csv = args.output / "tool_time_per_turn.csv"
    by_name_csv = args.output / "tool_time_by_name.csv"
    write_per_turn_csv(rows, per_turn_csv)
    write_by_name_csv(agg, by_name_csv)
    print(f"  wrote {per_turn_csv}")
    print(f"  wrote {by_name_csv}")

    print_summary(rows, agg, exclude)
    return 0


if __name__ == "__main__":
    sys.exit(main())
