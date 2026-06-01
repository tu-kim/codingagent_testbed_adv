#!/usr/bin/env python3
"""Per-request prefill/decode scheduling delay from dynamo worker logs.

Reads the `SCHED_DELAY` lines emitted by the testbed patch
(deploy/patches/dynamo-scheduling-log.patch) into each worker's
logs/vllm-*.log. Each line carries one request's engine scheduler
queue-wait (time between entering the WAITING queue and being scheduled
for compute) read from vLLM's per-request RequestOutput.metrics:

    SCHED_DELAY request_id=<id> role=<prefill|decode> queue_ms=<f> \\
                queued_ts=<f> scheduled_ts=<f>

Why worker logs and not in-band: the value can't ride to the client --
the dynamo frontend re-serializes `usage` through upstream async-openai
types that drop unknown keys, and the `nvext` response block is
Rust-only with no Python write path. So the per-worker log is the
per-request sink. Role is split because, with PD disaggregation, the
prefill and decode workers each log with their own RequestOutput.

The worker label comes from the log FILE name (e.g. logs/vllm-p0.log ->
"vllm-p0"); the role comes from the line. So you get the distribution
of scheduling delay per worker AND per role.

Outputs:
  scheduling_per_request.csv     worker, role, request_id, queue_ms, ...
  scheduling_by_worker_role.csv  worker, role, n, mean/p50/p90/p99/max
  stdout                         summary table + prefill-vs-decode rollup

Usage:
  scripts/analyze_worker_scheduling.py --logs logs/ --output results/run1/sched
  # --logs accepts a directory of vllm-*.log files OR a single log file.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np


_SCHED_RE = re.compile(
    r"SCHED_DELAY\s+"
    r"request_id=(?P<rid>\S+)\s+"
    r"role=(?P<role>\S+)\s+"
    r"queue_ms=(?P<queue_ms>[-0-9.eE+]+)\s+"
    r"queued_ts=(?P<queued>[-0-9.eE+]+)\s+"
    r"scheduled_ts=(?P<scheduled>[-0-9.eE+]+)"
)


@dataclass
class SchedRecord:
    worker: str
    role: str
    request_id: str
    queue_ms: float
    queued_ts: float
    scheduled_ts: float


# ---------- ingest ----------


def _iter_log_files(path: Path):
    if path.is_file():
        yield path
        return
    for f in sorted(path.glob("vllm-*.log")):
        yield f


def _worker_label(path: Path) -> str:
    """logs/vllm-p0.log -> 'vllm-p0'. Plain stem so it matches the
    component name testbed.sh spawned the worker under."""
    return path.stem


def load_records(path: Path) -> list[SchedRecord]:
    """Parse every SCHED_DELAY line across the worker logs. Lines that
    don't match (ordinary log output) are skipped silently."""
    records: list[SchedRecord] = []
    for f in _iter_log_files(path):
        worker = _worker_label(f)
        with f.open(encoding="utf-8", errors="replace") as fp:
            for line in fp:
                if "SCHED_DELAY" not in line:
                    continue
                m = _SCHED_RE.search(line)
                if not m:
                    continue
                try:
                    records.append(SchedRecord(
                        worker=worker,
                        role=m.group("role"),
                        request_id=m.group("rid"),
                        queue_ms=float(m.group("queue_ms")),
                        queued_ts=float(m.group("queued")),
                        scheduled_ts=float(m.group("scheduled")),
                    ))
                except ValueError:
                    continue
    return records


# ---------- stats ----------


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "p50": None,
                "p90": None, "p99": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def by_worker_role(records: list[SchedRecord]) -> dict[tuple[str, str], dict]:
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in records:
        buckets[(r.worker, r.role)].append(r.queue_ms)
    return {k: _stats(v) for k, v in buckets.items()}


def by_role(records: list[SchedRecord]) -> dict[str, dict]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for r in records:
        buckets[r.role].append(r.queue_ms)
    return {k: _stats(v) for k, v in buckets.items()}


# ---------- output ----------


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def write_per_request_csv(records: list[SchedRecord], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["worker", "role", "request_id",
                    "queue_ms", "queued_ts", "scheduled_ts"])
        for r in records:
            w.writerow([r.worker, r.role, r.request_id,
                        f"{r.queue_ms:.3f}", f"{r.queued_ts:.6f}",
                        f"{r.scheduled_ts:.6f}"])


def write_by_worker_role_csv(stats: dict[tuple[str, str], dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["worker", "role", "n",
                    "mean_ms", "p50_ms", "p90_ms", "p99_ms", "max_ms"])
        for (worker, role) in sorted(stats):
            s = stats[(worker, role)]
            w.writerow([worker, role, s["n"], _fmt(s["mean"]), _fmt(s["p50"]),
                        _fmt(s["p90"]), _fmt(s["p99"]), _fmt(s["max"])])


def print_summary(wr_stats: dict[tuple[str, str], dict],
                  role_stats: dict[str, dict],
                  n_records: int) -> None:
    print()
    print(f"SCHED_DELAY records parsed: {n_records}")
    print()
    if wr_stats:
        print("Scheduling delay by worker/role (queue_ms):")
        hdr = (f"{'worker':<16} {'role':<8} {'n':>6} {'mean':>9} "
               f"{'p50':>9} {'p90':>9} {'p99':>9} {'max':>9}")
        print(hdr)
        print("-" * len(hdr))
        for (worker, role) in sorted(wr_stats):
            s = wr_stats[(worker, role)]
            print(f"{worker:<16} {role:<8} {s['n']:>6} {_fmt(s['mean']):>9} "
                  f"{_fmt(s['p50']):>9} {_fmt(s['p90']):>9} "
                  f"{_fmt(s['p99']):>9} {_fmt(s['max']):>9}")
        print()
    if role_stats:
        print("Aggregate by role (queue_ms):")
        hdr = (f"{'role':<8} {'n':>6} {'mean':>9} {'p50':>9} "
               f"{'p90':>9} {'p99':>9} {'max':>9}")
        print(hdr)
        print("-" * len(hdr))
        for role in sorted(role_stats):
            s = role_stats[role]
            print(f"{role:<8} {s['n']:>6} {_fmt(s['mean']):>9} {_fmt(s['p50']):>9} "
                  f"{_fmt(s['p90']):>9} {_fmt(s['p99']):>9} {_fmt(s['max']):>9}")
        print()


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", required=True, type=Path,
                    help="Directory of vllm-*.log files (or a single log file)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory (created if missing)")
    args = ap.parse_args(argv)

    if not args.logs.exists():
        print(f"logs path not found: {args.logs}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    records = load_records(args.logs)
    if not records:
        print("no SCHED_DELAY lines found -- is the dynamo patch applied "
              "(scripts/apply_dynamo_patches.sh) and were the workers "
              "restarted after applying?", file=sys.stderr)
        return 1

    per_req = args.output / "scheduling_per_request.csv"
    by_wr = args.output / "scheduling_by_worker_role.csv"
    write_per_request_csv(records, per_req)
    wr_stats = by_worker_role(records)
    write_by_worker_role_csv(wr_stats, by_wr)
    print(f"  wrote {per_req}")
    print(f"  wrote {by_wr}")

    print_summary(wr_stats, by_role(records), len(records))
    return 0


if __name__ == "__main__":
    sys.exit(main())
