#!/usr/bin/env python3
"""Analyze how a sample's MAIN session and its nested `task` sub-agent
sessions actually interleave — the structure behind fig3's separate lines.

An opencode `task` tool call spawns a CHILD session with its OWN
session_id and its OWN KV prefix chain. Its turns run in the MIDDLE of the
parent sample's turn sequence (the parent is blocked while the sub-agent
runs), then the parent resumes. This script makes that structure explicit
and answers the KV questions it raises:

  * which parent TURN spawned each sub-agent (the turn whose tools include
    `task`), and how many turns / tokens the sub-agent ran;
  * the sub-agent's FIRST-turn cache hit — expected ~0 because it is a
    fresh prefix, confirming parent and child do NOT share KV;
  * the parent's RESUME turn after the excursion: did it still hit its own
    pre-task prefix (KV survived) or did it have to re-prefill (the
    sub-agent's KV + the elapsed time evicted the parent's blocks)?
  * the ordinal interleave (which global turns are parent vs child).

Consumes the profile NDJSON via the sibling analyze_turn_scheduling loader
(same TurnRec) and the e0 script's window/assignment helpers.

Usage:
  scripts/arm/analyze_session_nesting.py \\
      --profiles <workspace_root>/profiles \\
      --trace results/<run>/trace.jsonl \\
      [--out <dir>]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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


def _session_turns(turns: list) -> dict:
    """session_id -> its turns sorted by step."""
    by: dict[str, list] = {}
    for t in turns:
        by.setdefault(t.session_id, []).append(t)
    for v in by.values():
        v.sort(key=lambda t: t.step)
    return by


def _spawning_turn(parent_turns: list, child_start: float):
    """The parent turn that was 'active' when the child started: the last
    parent turn whose llm_start_ts <= child_start. Returns the TurnRec or
    None."""
    cand = None
    for t in parent_turns:
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if st is not None and child_start is not None and st <= child_start:
            cand = t
    return cand


def _resume_turn(parent_turns: list, child_end: float):
    """First parent turn that starts AFTER the child finished — where the
    parent resumes and we can read whether its own KV survived."""
    best = None
    for t in parent_turns:
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if st is not None and child_end is not None and st > child_end:
            if best is None or st < (best.llm_start_ts or best.llm_end_ts):
                best = t
    return best


def _first_start(turns_sorted: list):
    for t in turns_sorted:
        st = t.llm_start_ts if t.llm_start_ts is not None else t.llm_end_ts
        if st is not None:
            return st
    return None


def _last_end(turns_sorted: list):
    ends = [t.llm_end_ts for t in turns_sorted if t.llm_end_ts is not None]
    return max(ends) if ends else None


def build_nesting(turns: list, e0) -> tuple[list[dict], list[dict]]:
    """Return (child_rows, parent_rows).

    child_rows: one per (parent, nested child session) with spawning-turn,
    child size, fresh-prefix check, and parent-resume KV reuse.
    parent_rows: one per top-level sample with turn/child counts.
    """
    by_sess = _session_turns(turns)
    assign = e0.sample_assignment(e0.order_turns(turns), 1)
    # group children by parent
    children_of: dict[str, list[str]] = {}
    for sid, parent in assign.items():
        if parent != sid:
            children_of.setdefault(parent, []).append(sid)

    child_rows: list[dict] = []
    parent_rows: list[dict] = []
    for parent in sorted(by_sess):
        if assign.get(parent, parent) != parent:
            continue  # not a top-level sample
        p_turns = by_sess[parent]
        kids = sorted(children_of.get(parent, []),
                      key=lambda s: _first_start(by_sess[s]) or 0.0)
        parent_rows.append({
            "sample": parent,
            "parent_turns": len(p_turns),
            "n_task_subagents": len(kids),
            "child_turns_total": sum(len(by_sess[k]) for k in kids),
            "window_s": (round((_last_end(p_turns) or 0)
                               - (_first_start(p_turns) or 0), 3)),
        })
        for k in kids:
            c_turns = by_sess[k]
            c_start, c_end = _first_start(c_turns), _last_end(c_turns)
            spawn = _spawning_turn(p_turns, c_start)
            resume = _resume_turn(p_turns, c_end)
            first_child = c_turns[0] if c_turns else None
            child_rows.append({
                "sample": parent,
                "child": k,
                "spawn_step": spawn.step if spawn else "",
                "spawn_had_task": (spawn is not None
                                   and "task" in spawn.tool_names),
                "spawn_tools": "+".join(spawn.tool_names) if spawn else "",
                "child_turns": len(c_turns),
                "child_out_tokens": sum(t.output_tokens or 0 for t in c_turns),
                "child_first_cache_read": (first_child.cache_read
                                           if first_child else ""),
                "child_first_hit_ratio": (round(first_child.cache_hit_ratio, 4)
                                          if first_child
                                          and first_child.cache_hit_ratio
                                          is not None else ""),
                "child_wall_s": (round((c_end or 0) - (c_start or 0), 3)),
                "resume_step": resume.step if resume else "",
                "resume_hit_ratio": (round(resume.cache_hit_ratio, 4)
                                     if resume and resume.cache_hit_ratio
                                     is not None else ""),
                "resume_cache_read": resume.cache_read if resume else "",
                "resume_reprefill": (resume.input_tokens if resume else ""),
            })
    return child_rows, parent_rows


def ordinal_map(turns: list, e0) -> list[dict]:
    """Global turn ordinal -> which session (and parent/child role)."""
    ordered = e0.order_turns(turns)
    assign = e0.sample_assignment(ordered, 1)
    rows = []
    for i, t in enumerate(ordered):
        parent = assign.get(t.session_id, t.session_id)
        rows.append({
            "ordinal": i,
            "session_id": t.session_id,
            "role": "sample" if parent == t.session_id else "task_subagent",
            "parent_sample": parent,
            "step": t.step,
            "cache_read": t.cache_read,
            "hit_ratio": (round(t.cache_hit_ratio, 4)
                          if t.cache_hit_ratio is not None else ""),
        })
    return rows


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--trace", required=True, type=Path,
                    help="run's trace.jsonl; only its MAIN session_ids "
                         "(plus the task sub-agents nested inside them) are "
                         "analyzed")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.profiles.is_dir():
        print(f"error: profiles dir not found: {args.profiles}", file=sys.stderr)
        return 2
    if not args.trace.is_file():
        print(f"error: trace not found: {args.trace}", file=sys.stderr)
        return 2

    ats = _load("analyze_turn_scheduling", _ATS_PATH)
    e0 = _load("e0_turn_characterization", _E0_PATH)

    all_turns = ats.load_turns(args.profiles)
    if not all_turns:
        print("error: no turns parsed from profiles", file=sys.stderr)
        return 2

    # Keep main samples AND the task sub-agents nested inside them. A
    # sub-agent session is NOT in trace.jsonl, so we recover it by temporal
    # nesting: assign every session to its enclosing sample, keep those
    # whose sample is a trace main session.
    main_ids = e0.trace_session_ids(args.trace)
    if not main_ids:
        print("error: no session_id in trace.jsonl", file=sys.stderr)
        return 2
    assign = e0.sample_assignment(e0.order_turns(all_turns), 1)
    turns = [t for t in all_turns
             if assign.get(t.session_id, t.session_id) in main_ids
             or t.session_id in main_ids]

    child_rows, parent_rows = build_nesting(turns, e0)
    omap = ordinal_map(turns, e0)

    out_dir = args.out or (args.profiles / "nesting")
    out_dir.mkdir(parents=True, exist_ok=True)
    _write(out_dir / "session_summary.csv", parent_rows,
           ["sample", "parent_turns", "n_task_subagents",
            "child_turns_total", "window_s"])
    _write(out_dir / "session_nesting.csv", child_rows,
           ["sample", "child", "spawn_step", "spawn_had_task", "spawn_tools",
            "child_turns", "child_out_tokens", "child_first_cache_read",
            "child_first_hit_ratio", "child_wall_s", "resume_step",
            "resume_hit_ratio", "resume_cache_read", "resume_reprefill"])
    _write(out_dir / "ordinal_map.csv", omap,
           ["ordinal", "session_id", "role", "parent_sample", "step",
            "cache_read", "hit_ratio"])

    n_samples = len(parent_rows)
    n_with_kids = sum(1 for r in parent_rows if r["n_task_subagents"])
    n_children = len(child_rows)
    print(f"samples: {n_samples}  with task sub-agents: {n_with_kids}  "
          f"total sub-agent sessions: {n_children}")
    fresh = [r for r in child_rows
             if isinstance(r["child_first_hit_ratio"], (int, float))]
    if fresh:
        avg = sum(r["child_first_hit_ratio"] for r in fresh) / len(fresh)
        print(f"sub-agent FIRST-turn hit ratio: mean {avg:.3f} over "
              f"{len(fresh)} (only the shared boilerplate prefix -- system "
              "prompt/tools -- hits; the parent's CONVERSATION KV is not "
              "reused, so a fresh sub-agent prompt is mostly re-prefilled)")
    mism = [r for r in child_rows if r["spawn_had_task"] is False]
    if mism:
        print(f"note: {len(mism)} sub-agent(s) whose spawning parent turn did "
              "NOT carry a `task` tool — temporal nesting w/o a visible task "
              "call (check spawn_tools in session_nesting.csv)")
    resumes = [r for r in child_rows
               if isinstance(r["resume_hit_ratio"], (int, float))]
    if resumes:
        avg_r = sum(r["resume_hit_ratio"] for r in resumes) / len(resumes)
        print(f"parent RESUME-turn hit ratio after the excursion: mean "
              f"{avg_r:.3f} over {len(resumes)} (HIGH => parent's KV "
              "survived the sub-agent run; LOW => it was evicted)")
    print(f"wrote {out_dir / 'session_summary.csv'}")
    print(f"wrote {out_dir / 'session_nesting.csv'}")
    print(f"wrote {out_dir / 'ordinal_map.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
