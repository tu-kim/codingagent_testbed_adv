#!/usr/bin/env python3
"""Decompose SWE-agent APPS tasks into llm / tool / others wall-time --
the scaffold-comparison counterpart of analyze_profiles.py's fig6 /
latency-composition tables, computed from a run_sweagent_apps.py run dir.

Per task (trace.jsonl record):
  total  = rtt_s                       (sweagent subprocess wall)
  tool   = sum of per-step execution times found in the .traj trajectory
           (sweagent records the environment/tool wall per step; the field
           name has drifted across versions, so several candidates are
           probed -- see _STEP_TIME_KEYS. Missing everywhere -> NaN).
  llm    = sum of Dynamo frontend "request completed" elapsed_ms whose log
           timestamp falls inside [task_start_unix_s, task_end_unix_s]
           (recorded by run_sweagent_apps.py; runs are strictly
           sequential, so the window join is unambiguous). This is
           SERVER-side LLM wall -- add network/client overhead and it can
           only grow, so the llm share here is a lower bound.
  others = total - tool - llm  (clamped at 0): scaffold overhead --
           sweagent process startup, env setup/teardown, docker
           round-trips beyond the command itself, history serialization.

The per-task env-setup split rides along: env_head_s (task start -> first
LLM request start: docker create + swe-rex bootstrap + tool install; the
bulk of "others" on docker runs), env_tail_s (last completion -> exit:
patch extraction + container teardown), active_s = total - head - tail.
The stdout adds an ACTIVE-TIME share table (head/tail excluded) -- that is
the definition the literature's ~35% tool-share numbers use, and the fair
row against opencode's turn-level shares.

Outputs:
  <run>/sweagent_decomposition.csv   per-task seconds (llm/tool/others +
                                     env_head/env_tail/active)
  stdout                             pooled share table (total-based AND
                                     active-time; compare the latter with
                                     opencode's latency_pooled_share.csv)

Usage:
  scripts/analyze_sweagent_traj.py --run results/sweagent-apps1 \
      [--frontend logs/frontend.log]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Per-step env/tool wall-time field candidates across sweagent versions.
# Probed in order on every trajectory step dict.
_STEP_TIME_KEYS = ("execution_time", "execution_time_s", "env_time",
                   "action_execution_time")

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
# Dynamo frontend line: ISO timestamp ... "request completed" ... elapsed_ms
_COMPLETED_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?).*"
    r"request completed.*?elapsed_ms[=:\s\"]+(?P<ms>[0-9.]+)",
)


def load_trace(run_dir: Path) -> list[dict[str, Any]]:
    out = []
    with (run_dir / "trace.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def traj_tool_seconds(traj_dir: Path, instance_id: str) -> float:
    """Sum per-step execution time from the .traj file; NaN when the file
    or the timing fields are absent (older/newer sweagent)."""
    candidates = [traj_dir / f"{instance_id}.traj",
                  *sorted(traj_dir.glob("*.traj"))]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        return math.nan
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError:
        return math.nan
    steps = data.get("trajectory") or []
    total, found = 0.0, False
    for step in steps:
        if not isinstance(step, dict):
            continue
        for key in _STEP_TIME_KEYS:
            v = step.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                total += float(v)
                found = True
                break
    return total if found else math.nan


def load_frontend_completions(path: Path) -> list[tuple[float, float]]:
    """[(unix_ts, elapsed_s), ...] from the Dynamo frontend log. ANSI
    escapes are stripped first (the frontend colorizes its output)."""
    out: list[tuple[float, float]] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _COMPLETED_RE.match(_ANSI_RE.sub("", line).strip())
            if not m:
                continue
            ts_raw = m.group("ts").rstrip("Z")
            try:
                ts = datetime.fromisoformat(ts_raw).replace(
                    tzinfo=timezone.utc).timestamp()
                out.append((ts, float(m.group("ms")) / 1000.0))
            except ValueError:
                continue
    return out


def llm_seconds_in_window(completions: list[tuple[float, float]],
                          start: float, end: float) -> float:
    """Sum server-side elapsed of requests COMPLETED inside the window.
    Sequential runs -> no cross-task ambiguity. A request spanning the
    window start contributes fully (its completion lands inside), which
    matches how the task actually waited on it."""
    return sum(el for ts, el in completions if start <= ts <= end)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--frontend", default=None, type=Path,
                    help="Dynamo frontend log for server-side LLM walls; "
                         "omit to leave the llm column NaN (tool-only mode)")
    args = ap.parse_args(argv)

    records = load_trace(args.run)
    completions = (load_frontend_completions(args.frontend)
                   if args.frontend and args.frontend.exists() else [])
    if args.frontend and not completions:
        print(f"warning: no 'request completed' lines parsed from "
              f"{args.frontend}", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    for rec in records:
        total = rec.get("rtt_s")
        if total is None:
            continue
        tool = traj_tool_seconds(Path(rec["traj_dir"]), rec["instance_id"])
        llm = math.nan
        head = math.nan   # env setup: task start -> first LLM request start
        tail = math.nan   # env teardown: last LLM completion -> task end
        start, end = rec.get("task_start_unix_s"), rec.get("task_end_unix_s")
        if completions and start is not None and end is not None:
            win = [(ts, el) for ts, el in completions if start <= ts <= end]
            llm = sum(el for _, el in win)
            if win:
                # First completion's start = its completion ts minus its
                # elapsed. Everything before that is environment startup
                # (docker create, swe-rex bootstrap, tool-bundle install)
                # -- empirically the bulk of "others" on docker runs.
                head = max(0.0, (win[0][0] - win[0][1]) - start)
                tail = max(0.0, end - win[-1][0])
        others = math.nan
        if not math.isnan(tool) and not math.isnan(llm):
            others = max(0.0, total - tool - llm)
        active = math.nan
        if not math.isnan(head):
            active = max(0.0, total - head - tail)
        rows.append({
            "instance_id": rec["instance_id"],
            "success": rec.get("success"),
            "total_s": total,
            "llm_s": llm,
            "tool_s": tool,
            "others_s": others,
            "env_head_s": head,
            "env_tail_s": tail,
            "active_s": active,
        })

    if not rows:
        print("no usable records", file=sys.stderr)
        return 1

    out_csv = args.run / "sweagent_decomposition.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv}")

    # Pooled (time-weighted) shares over rows where every component is
    # known -- directly comparable with opencode's latency_pooled_share.csv.
    full = [r for r in rows
            if not math.isnan(r["tool_s"]) and not math.isnan(r["llm_s"])]
    print(f"\ntasks={len(rows)}  fully-decomposed={len(full)}")
    if full:
        tot = sum(r["total_s"] for r in full)
        llm = sum(r["llm_s"] for r in full)
        tool = sum(r["tool_s"] for r in full)
        others = sum(r["others_s"] for r in full)
        print(f"pooled share over {len(full)} tasks "
              f"(total {tot:.1f}s):")
        print(f"  llm    {llm:10.1f}s  {llm / tot:7.2%}")
        print(f"  tool   {tool:10.1f}s  {tool / tot:7.2%}")
        print(f"  others {others:10.1f}s  {others / tot:7.2%}")

        # Active-time shares: env head/tail excluded. This matches the
        # "share of ACTIVE time" definition used by the agent-workload
        # characterization literature (their ~35% tool numbers), and is
        # the apples-to-apples row against opencode's turn-level shares
        # (opencode turns carry no per-task env startup either).
        act = [r for r in full if not math.isnan(r["active_s"])]
        active_tot = sum(r["active_s"] for r in act)
        if act and active_tot > 0:
            head_tot = sum(r["env_head_s"] for r in act)
            tail_tot = sum(r["env_tail_s"] for r in act)
            llm_a = sum(r["llm_s"] for r in act)
            tool_a = sum(r["tool_s"] for r in act)
            others_a = sum(
                max(0.0, r["active_s"] - r["llm_s"] - r["tool_s"]) for r in act)
            print(f"\nactive-time share ({len(act)} tasks; env setup "
                  f"head {head_tot:.1f}s + tail {tail_tot:.1f}s excluded; "
                  f"active {active_tot:.1f}s):")
            print(f"  llm    {llm_a:10.1f}s  {llm_a / active_tot:7.2%}")
            print(f"  tool   {tool_a:10.1f}s  {tool_a / active_tot:7.2%}")
            print(f"  others {others_a:10.1f}s  {others_a / active_tot:7.2%}")
    else:
        known_tool = [r for r in rows if not math.isnan(r["tool_s"])]
        if known_tool:
            tot = sum(r["total_s"] for r in known_tool)
            tool = sum(r["tool_s"] for r in known_tool)
            print(f"tool-only pooled share ({len(known_tool)} tasks): "
                  f"{tool / tot:.2%} of wall "
                  f"(pass --frontend for the llm split)")
        else:
            print("no per-step execution times found in trajectories AND "
                  "no frontend log given -- nothing to decompose. Check "
                  "_STEP_TIME_KEYS against your sweagent version's .traj.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
