#!/usr/bin/env python3
"""A/B-compare queue-corrected TTFT between two runs -> KVBM onboard cost.

KVBM onboarding is ON-DEMAND at vLLM scheduling time (connector hook
get_num_new_matched_tokens fires when the request is scheduled), so the
host/disk->device transfer rides INSIDE the request's prefill/TTFT and
this tag records no per-request transfer duration anywhere. The only
per-request attribution is therefore inference:

    prefill_compute_ms = ttft_ms - prefill_queue_ms      (per run)
    onboard_cost ~= prefill_compute(kvbm) - prefill_compute(baseline)

Run the SAME workload twice (baseline: kvbm off; treatment: kvbm on),
then point this script at both log sets. Requests differ between runs
(non-deterministic agent loops), so the comparison is DISTRIBUTIONAL
(percentile deltas), optionally stratified by ISL bucket via each run's
frontend log ISL field when present -- matching on load profile is the
operator's job (same seed/qps/samples).

Colocation note: with agg workers the SCHED_DELAY record is decode-only;
queue subtraction then uses that single record's queue_ms.

Usage:
  scripts/compare_prefill_compute.py \\
      --baseline-frontend logs_base/frontend.log --baseline-logs logs_base/ \\
      --kvbm-frontend logs_kvbm/frontend.log     --kvbm-logs logs_kvbm/ \\
      [--out <dir>]

Outputs:
  stdout               side-by-side percentiles of ttft_ms /
                       prefill_compute_ms + delta row (onboard estimate)
  <out>/compare_prefill_compute.csv   the same table, machine-readable
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJ]")
_REQID_RE = re.compile(r'(?:\b|")request_id\b"?\s*[=:]\s*"?(?P<v>[^\s",}]+)"?')
_ELAPSED_RE = re.compile(r'(?:\b|")elapsed_ms\b"?\s*[=:]\s*"?(?P<v>\d+)"?')
_TTFT_RE = re.compile(r'(?:\b|")ttft_ms\b"?\s*[=:]\s*"?(?P<v>[\d.]+)"?')
_SCHED_RE = re.compile(
    r"SCHED_DELAY\s+request_id=(?P<rid>\S+)\s+role=(?P<role>\S+)\s+"
    r"queue_ms=(?P<queue_ms>[-0-9.eE+]+)"
)

PCTS = (0.50, 0.90, 0.95, 0.99)


@dataclass
class RunStats:
    label: str
    n: int
    ttft: list[float]
    compute: list[float]          # ttft - total queue (prefill + decode records)


def _iter_log_files(path: Path):
    if path.is_file():
        yield path
        return
    for f in sorted(path.glob("vllm-*.log")):
        yield f


def parse_frontend_ttft(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
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


def parse_queues(path: Path) -> dict[str, float]:
    """request_id -> total queue_ms (sum of prefill + decode records)."""
    out: dict[str, float] = {}
    for fpath in _iter_log_files(path):
        with fpath.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _SCHED_RE.search(line)
                if not m:
                    continue
                out[m.group("rid")] = out.get(m.group("rid"), 0.0) + float(m.group("queue_ms"))
    return out


def load_run(label: str, frontend: Path, logs: Path) -> RunStats:
    ttfts = parse_frontend_ttft(frontend)
    queues = parse_queues(logs)
    ttft_vals: list[float] = []
    compute_vals: list[float] = []
    for rid, ttft in ttfts.items():
        ttft_vals.append(ttft)
        q = queues.get(rid)
        if q is not None:
            compute_vals.append(ttft - q)
    return RunStats(label=label, n=len(ttft_vals), ttft=ttft_vals, compute=compute_vals)


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def build_rows(base: RunStats, kvbm: RunStats) -> list[dict]:
    rows = []
    for metric, b_vals, k_vals in (
        ("ttft_ms", base.ttft, kvbm.ttft),
        ("prefill_compute_ms", base.compute, kvbm.compute),
    ):
        for q in PCTS:
            b = _pct(b_vals, q)
            k = _pct(k_vals, q)
            rows.append({
                "metric": metric,
                "percentile": f"p{q * 100:g}",
                "baseline": b,
                "kvbm": k,
                "delta": k - b if not (math.isnan(b) or math.isnan(k)) else math.nan,
            })
        b_mean = sum(b_vals) / len(b_vals) if b_vals else math.nan
        k_mean = sum(k_vals) / len(k_vals) if k_vals else math.nan
        rows.append({
            "metric": metric, "percentile": "mean",
            "baseline": b_mean, "kvbm": k_mean,
            "delta": k_mean - b_mean if not (math.isnan(b_mean) or math.isnan(k_mean)) else math.nan,
        })
    return rows


def print_table(base: RunStats, kvbm: RunStats, rows: list[dict]) -> None:
    print(f"baseline: n={base.n} requests ({len(base.compute)} with queue join)")
    print(f"kvbm:     n={kvbm.n} requests ({len(kvbm.compute)} with queue join)")
    print()
    print(f"{'metric':22s} {'pct':>5s} {'baseline':>10s} {'kvbm':>10s} {'delta':>10s}")
    for r in rows:
        print(f"{r['metric']:22s} {r['percentile']:>5s} "
              f"{r['baseline']:10.1f} {r['kvbm']:10.1f} {r['delta']:+10.1f}")
    # headline: mean prefill_compute delta = per-request onboard cost estimate
    mean_row = next(r for r in rows
                    if r["metric"] == "prefill_compute_ms" and r["percentile"] == "mean")
    d = mean_row["delta"]
    if not math.isnan(d):
        print(f"\nonboard-cost estimate (mean prefill_compute delta): {d:+.1f} ms/request")
        print("  positive = kvbm's on-demand host/disk->device transfer riding "
              "in TTFT; compare against the re-prefill it avoided (cache-hit "
              "gain) before judging net effect.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline-frontend", required=True, type=Path)
    ap.add_argument("--baseline-logs", required=True, type=Path)
    ap.add_argument("--kvbm-frontend", required=True, type=Path)
    ap.add_argument("--kvbm-logs", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    for p in (args.baseline_frontend, args.kvbm_frontend):
        if not p.exists():
            print(f"error: frontend log not found: {p}", file=sys.stderr)
            return 2

    base = load_run("baseline", args.baseline_frontend, args.baseline_logs)
    kvbm = load_run("kvbm", args.kvbm_frontend, args.kvbm_logs)
    if base.n == 0 or kvbm.n == 0:
        print("error: one of the runs has zero 'request completed' lines",
              file=sys.stderr)
        return 2

    rows = build_rows(base, kvbm)
    print_table(base, kvbm, rows)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        out_csv = args.out / "compare_prefill_compute.csv"
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
