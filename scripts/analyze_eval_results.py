#!/usr/bin/env python3
"""Join the official SWE-bench harness report with the testbed trace.jsonl
to get per-instance TRUE success/fail next to the testbed's HTTP-level
view (rtt, error stage).

Inputs:
  --run     run dir with trace.jsonl (+ where evaluate_predictions.sh left
            the report, unless --report overrides)
  --report  harness report json (<model>.<run_id>.json). Auto-discovered
            in --run when omitted (newest *.json with a "resolved" key).

Per-instance verdicts (column `verdict`):
  resolved        harness ran the tests, FAIL_TO_PASS + PASS_TO_PASS all good
  unresolved      patch applied (or ran) but tests did not all pass
  empty_patch     prediction had no patch (agent produced no change / task failed)
  error           harness errored on this instance (patch apply failure etc.)
  incomplete      harness did not finish this instance
  not_in_report   instance absent from the report entirely

Summary distinguishes:
  resolve_rate_all          resolved / all tasks in trace
  resolve_rate_http_ok      resolved / tasks whose agent loop completed
                            (trace success=true) -- isolates "agent finished
                            but fix is wrong" from infra failures

Usage:
  scripts/analyze_eval_results.py --run results/run1
  scripts/analyze_eval_results.py --run results/run1 --report results/run1/testbed.run1.json \
      --csv results/run1/eval_per_instance.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# The harness has used two key styles across versions; accept both.
_ID_LIST_KEYS = {
    "resolved": ("resolved_ids", "resolved"),
    "unresolved": ("unresolved_ids", "unresolved"),
    "error": ("error_ids", "error"),
    "empty_patch": ("empty_patch_ids", "empty_patch"),
    "incomplete": ("incomplete_ids", "incomplete"),
}


def load_report(path: Path) -> dict[str, str]:
    """Returns {instance_id: verdict}."""
    data = json.loads(path.read_text())
    verdicts: dict[str, str] = {}
    # Order matters: an id may appear in multiple lists in some versions
    # (e.g. error + unresolved); the more specific verdict wins, so apply
    # in increasing precedence.
    for verdict in ("incomplete", "empty_patch", "error", "unresolved", "resolved"):
        for key in _ID_LIST_KEYS[verdict]:
            ids = data.get(key)
            if isinstance(ids, list):
                for iid in ids:
                    verdicts[iid] = verdict
                break
    return verdicts


def find_report(run_dir: Path) -> Path | None:
    """Newest json in run_dir that looks like a harness report."""
    candidates = []
    for p in run_dir.glob("*.json"):
        if p.name in ("config.json", "summary.json"):
            continue
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if any(k in data for keys in _ID_LIST_KEYS.values() for k in keys):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_trace(run_dir: Path) -> list[dict]:
    records = []
    with (run_dir / "trace.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--report", type=Path, default=None,
                    help="Harness report json (auto-discovered in --run if omitted)")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Optional per-instance CSV output")
    args = ap.parse_args()

    report_path = args.report or find_report(args.run)
    if report_path is None:
        print(f"error: no harness report found in {args.run}; "
              f"run scripts/evaluate_predictions.sh first or pass --report",
              file=sys.stderr)
        return 1

    verdicts = load_report(report_path)
    records = load_trace(args.run)

    rows = []
    for rec in records:
        iid = rec["instance_id"]
        err = rec.get("error") or {}
        rows.append({
            "instance_id": iid,
            "verdict": verdicts.get(iid, "not_in_report"),
            "http_success": rec.get("success", False),
            "error_stage": err.get("stage", ""),
            "rtt_s": rec.get("rtt_s"),
        })

    # ---- stdout table ----
    print(f"report: {report_path}")
    print(f"{'instance_id':<42} {'verdict':<14} {'http':<6} {'stage':<9} {'rtt_s':>8}")
    for r in sorted(rows, key=lambda r: (r["verdict"], r["instance_id"])):
        rtt = f"{r['rtt_s']:.1f}" if isinstance(r["rtt_s"], (int, float)) else "-"
        print(f"{r['instance_id']:<42} {r['verdict']:<14} "
              f"{str(r['http_success']).lower():<6} {r['error_stage']:<9} {rtt:>8}")

    # ---- summary ----
    n = len(rows)
    n_resolved = sum(1 for r in rows if r["verdict"] == "resolved")
    http_ok = [r for r in rows if r["http_success"]]
    n_http_ok = len(http_ok)
    n_resolved_http_ok = sum(1 for r in http_ok if r["verdict"] == "resolved")

    by_verdict: dict[str, int] = {}
    for r in rows:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1

    print()
    print(f"tasks: {n}   verdicts: " +
          "  ".join(f"{k}={v}" for k, v in sorted(by_verdict.items())))
    print(f"resolve_rate_all     = {n_resolved}/{n}"
          f" = {n_resolved / n:.1%}" if n else "resolve_rate_all     = n/a")
    if n_http_ok:
        print(f"resolve_rate_http_ok = {n_resolved_http_ok}/{n_http_ok}"
              f" = {n_resolved_http_ok / n_http_ok:.1%}")
    else:
        print("resolve_rate_http_ok = n/a (no http-successful tasks)")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows
                               else ["instance_id", "verdict", "http_success",
                                     "error_stage", "rtt_s"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
