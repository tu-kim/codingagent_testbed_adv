#!/usr/bin/env python3
"""Find parallel tool calls and sub-agent (task tool) spawns in a
testbed trace.jsonl.

Two kinds of parallelism happen inside an opencode session:

1. Parallel tool calls within a single LLM step. opencode marks step
   boundaries with `step-start` / `step-finish` parts. Tool parts
   between them belong to one assistant LLM response -- if there are
   N > 1 of them, the model emitted N tool_calls in that one response
   and opencode executed them concurrently.

2. Sub-agent spawns via the `task` tool. The `task` tool runs a
   nested opencode session driven by a fresh agent. Its messages do
   NOT appear in this trace.jsonl (they live in a separate session id
   on the opencode server) -- we only see the parent's `task` tool
   call here.

Outputs:
  parallel_batches.csv   one row per step that contained >1 tool calls
                         (instance_id, msg_idx, step_idx, n_tools, tools)
  task_spawns.csv        one row per `task` tool invocation
                         (instance_id, msg_idx, callID, description)
  stdout                 summary table

Usage:
  scripts/analyze_trace_parallelism.py \\
      --trace results/run1/trace.jsonl \\
      --output results/run1/parallelism

For reproducibility comparisons across two runs, run twice and diff
the per-task parallel-step counts -- if run1 has 5 parallel steps and
run2 has 7 for the same instance_id, the agent loops diverged.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ParallelBatch:
    instance_id: str
    msg_idx: int       # index of the assistant message in trace.messages[]
    step_idx: int      # nth step inside that message (0-based)
    tools: list[str]   # tool names in the order opencode recorded them


@dataclass
class TaskSpawn:
    instance_id: str
    msg_idx: int
    call_id: str
    description: str   # best-effort -- input shape can vary by opencode version


# ---------- ingest ----------


def _task_input_description(inp: dict | None) -> str:
    """Pull a human-readable description out of a task tool input.

    Schema can drift: opencode has used `description`, `prompt`, plain
    `input` etc. across versions. Fall back to a truncated repr so the
    CSV never shows None when something was passed."""
    if not isinstance(inp, dict):
        return repr(inp)[:120]
    for key in ("description", "prompt", "instructions", "task"):
        v = inp.get(key)
        if isinstance(v, str) and v:
            return v[:200]
    return json.dumps(inp, sort_keys=True)[:200]


def analyze_trace(path: Path) -> tuple[list[ParallelBatch], list[TaskSpawn],
                                       dict[str, int], dict[str, int]]:
    """Walk every TaskRecord line in `path` and return:

      (parallel_batches, task_spawns, steps_per_task, parallel_per_task)

    where the two dicts are keyed by instance_id and let downstream code
    compute per-task parallel-step ratios."""
    parallel_batches: list[ParallelBatch] = []
    task_spawns: list[TaskSpawn] = []
    steps_per_task: dict[str, int] = defaultdict(int)
    parallel_per_task: dict[str, int] = defaultdict(int)

    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError:
                continue
            instance_id = rec.get("instance_id") or "?"
            for msg_idx, msg in enumerate(rec.get("messages") or []):
                if (msg.get("info") or {}).get("role") != "assistant":
                    continue
                cur_tools: list[str] = []
                in_step = False
                step_idx_in_msg = 0
                for part in msg.get("parts") or []:
                    ptype = part.get("type")
                    if ptype == "step-start":
                        # Reset accumulator on every step boundary so a
                        # half-finished prior step (e.g. error mid-stream)
                        # doesn't leak tools into the next step's batch.
                        cur_tools = []
                        in_step = True
                    elif ptype == "step-finish":
                        if in_step:
                            steps_per_task[instance_id] += 1
                            if len(cur_tools) > 1:
                                parallel_batches.append(ParallelBatch(
                                    instance_id=instance_id,
                                    msg_idx=msg_idx,
                                    step_idx=step_idx_in_msg,
                                    tools=cur_tools[:],
                                ))
                                parallel_per_task[instance_id] += 1
                            step_idx_in_msg += 1
                        cur_tools = []
                        in_step = False
                    elif ptype == "tool" and in_step:
                        tool_name = part.get("tool", "?")
                        cur_tools.append(tool_name)
                        if tool_name == "task":
                            task_spawns.append(TaskSpawn(
                                instance_id=instance_id,
                                msg_idx=msg_idx,
                                call_id=part.get("callID", ""),
                                description=_task_input_description(
                                    (part.get("state") or {}).get("input")
                                ),
                            ))

    return parallel_batches, task_spawns, dict(steps_per_task), dict(parallel_per_task)


# ---------- output ----------


def write_parallel_batches_csv(batches: list[ParallelBatch], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "msg_idx", "step_idx", "n_tools", "tools"])
        for b in batches:
            w.writerow([b.instance_id, b.msg_idx, b.step_idx,
                        len(b.tools), ",".join(b.tools)])


def write_task_spawns_csv(spawns: list[TaskSpawn], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "msg_idx", "callID", "description"])
        for s in spawns:
            w.writerow([s.instance_id, s.msg_idx, s.call_id, s.description])


def print_summary(batches: list[ParallelBatch],
                  spawns: list[TaskSpawn],
                  steps_per_task: dict[str, int],
                  parallel_per_task: dict[str, int]) -> None:
    total_steps = sum(steps_per_task.values())
    total_parallel = len(batches)
    rate_pct = (100.0 * total_parallel / total_steps) if total_steps else 0.0

    print()
    print(f"LLM steps total:          {total_steps}")
    print(f"Steps with parallel tools: {total_parallel}  ({rate_pct:.1f}%)")
    print(f"Sub-agent (task) spawns:   {len(spawns)}")
    print()

    # Top parallel-tool combinations (order-insensitive).
    combos: Counter[tuple[str, ...]] = Counter(
        tuple(sorted(b.tools)) for b in batches
    )
    if combos:
        print("Top parallel-tool combinations (sorted name tuple → count):")
        for combo, n in combos.most_common(10):
            print(f"  {n:>4}× {list(combo)}")
        print()

    # Distribution of batch sizes (how many tools in a step).
    sizes = Counter(len(b.tools) for b in batches)
    if sizes:
        print("Batch-size distribution (tools per parallel step):")
        for sz in sorted(sizes):
            print(f"  {sz} tools: {sizes[sz]:>4} steps")
        print()

    # Per-task parallel rate, top by absolute parallel-step count.
    if parallel_per_task:
        print("Per-task parallel-step rate (top 10 by count):")
        ranked = sorted(parallel_per_task.items(), key=lambda kv: -kv[1])[:10]
        for iid, n_par in ranked:
            n_tot = steps_per_task.get(iid, 0)
            pct = (100.0 * n_par / n_tot) if n_tot else 0.0
            print(f"  {iid:<45} {n_par:>3} / {n_tot:>3} steps  ({pct:>5.1f}%)")
        print()

    # Show a few task spawns so the user can eyeball what kind of work
    # is being delegated to sub-agents.
    if spawns:
        print("First task tool invocations:")
        for s in spawns[:10]:
            print(f"  [{s.instance_id:<40}] msg#{s.msg_idx:<3} {s.description!r}")


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trace", required=True, type=Path,
                    help="trace.jsonl from `testbed run --out <dir>`")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory (created if missing)")
    args = ap.parse_args(argv)

    if not args.trace.exists():
        print(f"trace not found: {args.trace}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    batches, spawns, steps_per_task, parallel_per_task = analyze_trace(args.trace)

    batches_csv = args.output / "parallel_batches.csv"
    spawns_csv = args.output / "task_spawns.csv"
    write_parallel_batches_csv(batches, batches_csv)
    write_task_spawns_csv(spawns, spawns_csv)
    print(f"  wrote {batches_csv}")
    print(f"  wrote {spawns_csv}")

    print_summary(batches, spawns, steps_per_task, parallel_per_task)
    return 0


if __name__ == "__main__":
    sys.exit(main())
