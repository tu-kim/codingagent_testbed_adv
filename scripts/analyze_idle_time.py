#!/usr/bin/env python3
"""Inter-turn idle time in multi-turn opencode sessions, from profile NDJSON.

A session's wall clock is NOT just the sum of its turns. Between one
turn ending and the next starting, opencode does framework work
(snapshot tracking, DB writes, deciding the next step) and may sit idle
waiting. This script decomposes each session's query wall into:

  busy_turn_s         Σ (turn.end.ts - turn.start.ts)  -- inside turns
  bootstrap_s         first turn.start.ts - query.start.ts (LSP/plugin
                      init, first prompt assembly)
  inter_turn_idle_s   Σ gaps between consecutive turns
                      (turn[N+1].start.ts - turn[N].end.ts)
  teardown_s          query.end.ts - last turn.end.ts

  total_idle_s = bootstrap_s + inter_turn_idle_s + teardown_s
  query wall  ≈ busy_turn_s + total_idle_s

All four event types carry `ts` (unix seconds): query.start, turn.start,
turn.end, query.end -- so the gaps are exact timestamp differences, not
reconstructions.

Outputs:
  idle_gaps.csv        one row per inter-turn gap
                       (session_id, prev_step, next_step,
                        prev_end_ts, next_start_ts, idle_s)
  idle_per_session.csv one row per session (the decomposition above)
  stdout               aggregate idle share + gap distribution

Usage:
  scripts/analyze_idle_time.py \\
      --profile /tmp/testbed-workspaces/profiles \\
      --output results/run1/idle
  # --profile is a dir of <sessionID>.jsonl files OR one aggregated file.

Note on nested sub-agents: a `task` sub-agent runs as its OWN session
with its own query/turn events, so it's decomposed independently here.
From the PARENT session's view the whole child runs inside one tool
call, i.e. inside a turn's busy time -- it does NOT appear as parent
idle. So a high inter_turn_idle in the parent is genuine framework /
scheduling gap, not sub-agent execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------- model ----------


@dataclass
class _Turn:
    step: int
    start_ts: float | None = None
    end_ts: float | None = None


@dataclass
class _Session:
    session_id: str
    query_start_ts: float | None = None
    query_end_ts: float | None = None
    query_duration_s: float | None = None
    turns: dict[int, _Turn] = field(default_factory=dict)

    def _ensure(self, step: int) -> _Turn:
        t = self.turns.get(step)
        if t is None:
            t = _Turn(step=step)
            self.turns[step] = t
        return t


@dataclass
class Gap:
    session_id: str
    prev_step: int
    next_step: int
    prev_end_ts: float
    next_start_ts: float

    @property
    def idle_s(self) -> float:
        return self.next_start_ts - self.prev_end_ts


@dataclass
class SessionIdle:
    session_id: str
    n_turns: int
    query_duration_s: float | None
    busy_turn_s: float
    bootstrap_s: float | None
    inter_turn_idle_s: float
    teardown_s: float | None

    @property
    def total_idle_s(self) -> float:
        total = self.inter_turn_idle_s
        if self.bootstrap_s is not None:
            total += self.bootstrap_s
        if self.teardown_s is not None:
            total += self.teardown_s
        return total

    @property
    def idle_pct(self) -> float | None:
        if self.query_duration_s is None or self.query_duration_s <= 0:
            return None
        return 100.0 * self.total_idle_s / self.query_duration_s


# ---------- ingest ----------


def _iter_event_files(path: Path):
    if path.is_file():
        yield path
        return
    for f in sorted(path.glob("*.jsonl")):
        yield f


def load_sessions(path: Path) -> list[_Session]:
    sessions: dict[str, _Session] = {}

    def ensure(sid: str) -> _Session:
        s = sessions.get(sid)
        if s is None:
            s = _Session(session_id=sid)
            sessions[sid] = s
        return s

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
                if not etype or not sid:
                    continue
                ts = ev.get("ts")

                if etype == "query.start":
                    ensure(sid).query_start_ts = ts
                elif etype == "query.end":
                    s = ensure(sid)
                    s.query_end_ts = ts
                    s.query_duration_s = ev.get("duration_s")
                elif etype == "turn.start":
                    step = ev.get("step")
                    if step is None:
                        continue
                    ensure(sid)._ensure(step).start_ts = ts
                elif etype == "turn.end":
                    step = ev.get("step")
                    if step is None:
                        continue
                    ensure(sid)._ensure(step).end_ts = ts

    return list(sessions.values())


# ---------- compute ----------


def session_gaps(s: _Session) -> list[Gap]:
    """Inter-turn gaps for one session, ordered by step. Only emits a
    gap when both adjacent turns have the needed timestamp (prev.end,
    next.start)."""
    ordered = [t for _, t in sorted(s.turns.items())]
    gaps: list[Gap] = []
    for prev, nxt in zip(ordered, ordered[1:]):
        if prev.end_ts is None or nxt.start_ts is None:
            continue
        gaps.append(Gap(
            session_id=s.session_id,
            prev_step=prev.step,
            next_step=nxt.step,
            prev_end_ts=prev.end_ts,
            next_start_ts=nxt.start_ts,
        ))
    return gaps


def session_idle(s: _Session) -> SessionIdle:
    ordered = [t for _, t in sorted(s.turns.items())]
    busy = sum(
        t.end_ts - t.start_ts
        for t in ordered
        if t.start_ts is not None and t.end_ts is not None
    )
    gaps = session_gaps(s)
    inter = sum(g.idle_s for g in gaps)

    bootstrap = None
    if ordered and ordered[0].start_ts is not None and s.query_start_ts is not None:
        bootstrap = ordered[0].start_ts - s.query_start_ts

    teardown = None
    if ordered and ordered[-1].end_ts is not None and s.query_end_ts is not None:
        teardown = s.query_end_ts - ordered[-1].end_ts

    return SessionIdle(
        session_id=s.session_id,
        n_turns=len(ordered),
        query_duration_s=s.query_duration_s,
        busy_turn_s=busy,
        bootstrap_s=bootstrap,
        inter_turn_idle_s=inter,
        teardown_s=teardown,
    )


# ---------- output ----------


def _fmt(v: float | None) -> str:
    return "" if v is None else f"{v:.4f}"


def write_gaps_csv(gaps: list[Gap], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "prev_step", "next_step",
                    "prev_end_ts", "next_start_ts", "idle_s"])
        for g in gaps:
            w.writerow([g.session_id, g.prev_step, g.next_step,
                        f"{g.prev_end_ts:.3f}", f"{g.next_start_ts:.3f}",
                        f"{g.idle_s:.4f}"])


def write_per_session_csv(rows: list[SessionIdle], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session_id", "n_turns", "query_duration_s", "busy_turn_s",
            "bootstrap_s", "inter_turn_idle_s", "teardown_s",
            "total_idle_s", "idle_pct",
        ])
        for r in rows:
            w.writerow([
                r.session_id, r.n_turns, _fmt(r.query_duration_s),
                _fmt(r.busy_turn_s), _fmt(r.bootstrap_s),
                _fmt(r.inter_turn_idle_s), _fmt(r.teardown_s),
                _fmt(r.total_idle_s),
                "" if r.idle_pct is None else f"{r.idle_pct:.2f}",
            ])


def print_summary(rows: list[SessionIdle], gaps: list[Gap]) -> None:
    multi = [r for r in rows if r.n_turns > 1]
    total_q = sum(r.query_duration_s for r in rows if r.query_duration_s)
    total_idle = sum(r.total_idle_s for r in rows)
    total_inter = sum(r.inter_turn_idle_s for r in rows)
    agg_idle_pct = (100.0 * total_idle / total_q) if total_q else 0.0

    print()
    print(f"Sessions total:              {len(rows)}")
    print(f"Multi-turn sessions:         {len(multi)}")
    print(f"Inter-turn gaps observed:    {len(gaps)}")
    print(f"Aggregate query wall:        {total_q:.2f}s")
    print(f"Aggregate total idle:        {total_idle:.2f}s ({agg_idle_pct:.1f}%)")
    print(f"  of which inter-turn idle:  {total_inter:.2f}s")
    print()

    if gaps:
        idles = sorted(g.idle_s for g in gaps)
        print("Inter-turn gap distribution (seconds):")
        print(f"  n={len(idles)}  min={idles[0]:.4f}  "
              f"median={statistics.median(idles):.4f}  "
              f"max={idles[-1]:.4f}  mean={statistics.fmean(idles):.4f}")
        print()

        print("Largest inter-turn gaps:")
        hdr = f"{'session_id':<34} {'prev→next':>11} {'idle_s':>10}"
        print(hdr)
        print("-" * len(hdr))
        for g in sorted(gaps, key=lambda g: -g.idle_s)[:15]:
            sid = (g.session_id[:31] + "...") if len(g.session_id) > 34 else g.session_id
            print(f"{sid:<34} {f'{g.prev_step}→{g.next_step}':>11} {g.idle_s:>10.4f}")
        print()

    if rows:
        print("Top sessions by idle share:")
        hdr = (f"{'session_id':<34} {'turns':>5} {'query_s':>9} "
               f"{'idle_s':>9} {'idle%':>7}")
        print(hdr)
        print("-" * len(hdr))
        ranked = sorted(
            rows,
            key=lambda r: (r.idle_pct if r.idle_pct is not None else -1.0),
            reverse=True,
        )[:15]
        for r in ranked:
            sid = (r.session_id[:31] + "...") if len(r.session_id) > 34 else r.session_id
            pct = "-" if r.idle_pct is None else f"{r.idle_pct:.1f}%"
            print(f"{sid:<34} {r.n_turns:>5} {_fmt(r.query_duration_s):>9} "
                  f"{r.total_idle_s:>9.4f} {pct:>7}")
        print()


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", required=True, type=Path,
                    help="Profile NDJSON dir (<sessionID>.jsonl) or single file")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory (created if missing)")
    args = ap.parse_args(argv)

    if not args.profile.exists():
        print(f"profile path not found: {args.profile}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    sessions = load_sessions(args.profile)
    if not sessions:
        print("no sessions found in profile NDJSON", file=sys.stderr)
        return 1

    sessions.sort(key=lambda s: s.session_id)
    rows = [session_idle(s) for s in sessions]
    gaps: list[Gap] = []
    for s in sessions:
        gaps.extend(session_gaps(s))

    gaps_csv = args.output / "idle_gaps.csv"
    per_sess_csv = args.output / "idle_per_session.csv"
    write_gaps_csv(gaps, gaps_csv)
    write_per_session_csv(rows, per_sess_csv)
    print(f"  wrote {gaps_csv}")
    print(f"  wrote {per_sess_csv}")

    print_summary(rows, gaps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
