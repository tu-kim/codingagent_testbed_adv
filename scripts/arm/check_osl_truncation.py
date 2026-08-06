#!/usr/bin/env python3
"""Confirm (or refute) that frontend.log's output_tokens is truncated.

Background: dynamo records a request's OSL by having
ResponseMetricCollector::Drop write `self.osl` onto the enclosing span
(metrics.rs:1571-1598). Two things cut that value short:
  * the cancel path breaks the streaming loop early
    (disconnect.rs:272-286, `context.stopped()`), freezing self.osl at a
    partial count;
  * the InflightGuard that emits the "request completed" line races that
    write with no guaranteed drop order (vendor comment,
    disconnect.rs:275-276).
The Prometheus output_sequence_length histogram reads the same field, so
it is poisoned identically. The profile usage-chunk value
(llm.end.tokens.output == usage.completion_tokens, delta.rs:252-258) is
an independent accumulator and is the trusted source.

This script runs the three discriminating checks:

  1. status/error_type distribution over "request completed" lines. A
     large `cancelled`/error share is the cancel-path signature.
  2. per-request join frontend vs profile OSL: how many lines are short,
     by how much, and what the frontend caps out at.
  3. totals: sum of frontend output_tokens vs sum of profile OSL. The
     per-model `..._output_tokens_total` counter (metrics.rs:1484)
     increments per chunk and is immune, so if you also pass
     --metrics-total (scraped from /metrics), it anchors the comparison.

Usage:
  scripts/arm/check_osl_truncation.py --frontend logs/frontend.log \
      --profiles /tmp/testbed-workspaces/profiles \
      [--metrics-total 1234567]
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from collections import Counter
from pathlib import Path

_ARM = Path(__file__).resolve().parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_STATUS_RE = re.compile(r'(?:\b|")status\b"?\s*[=:]\s*"?(?P<v>[\w-]+)"?')
_ERRTYPE_RE = re.compile(r'(?:\b|")error_type\b"?\s*[=:]\s*"?(?P<v>[\w-]+)"?')


def status_distribution(e4, path: Path) -> tuple[Counter, Counter, int]:
    """(status counts, error_type counts, total completed lines)."""
    st, et = Counter(), Counter()
    total = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if "request completed" not in line:
                continue
            total += 1
            line = e4._ANSI_RE.sub("", line)
            m = _STATUS_RE.search(line)
            st[m.group("v") if m else "(absent)"] += 1
            m = _ERRTYPE_RE.search(line)
            if m:
                et[m.group("v")] += 1
    return st, et, total


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    idx = q / 100.0 * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frontend", required=True, type=Path)
    ap.add_argument("--profiles", required=True, type=Path,
                    help="profile NDJSON dir (or a single .jsonl)")
    ap.add_argument("--metrics-total", type=float, default=None,
                    help="value of the per-model ..._output_tokens_total "
                         "counter scraped from /metrics (immune to the "
                         "truncation); anchors the totals comparison")
    args = ap.parse_args(argv)

    if not args.frontend.is_file():
        print(f"error: frontend log not found: {args.frontend}", file=sys.stderr)
        return 2
    if not args.profiles.exists():
        print(f"error: profiles not found: {args.profiles}", file=sys.stderr)
        return 2

    e4 = _load("_osl_e4", _ARM / "e4_prefill_decode.py")

    # -------------------------------------------------- 1. cancel signature
    st, et, total = status_distribution(e4, args.frontend)
    print(f"[1] 'request completed' lines: {total}")
    print("    status:", dict(st.most_common()))
    if et:
        print("    error_type:", dict(et.most_common()))
    cancelish = sum(v for k, v in st.items()
                    if k.lower() not in ("success", "ok", "(absent)"))
    if total:
        print(f"    non-success share: {cancelish}/{total} "
              f"({100.0 * cancelish / total:.1f}%) "
              "-- high share = cancel-path signature")

    # -------------------------------------------------- 2. per-request join
    rows = e4.parse_frontend(args.frontend)
    prof = e4.profile_output_tokens(args.profiles)
    print(f"\n[2] frontend requests parsed: {len(rows)}  "
          f"profile llm.end with request_id: {len(prof)}")

    joined = [(r["output_tokens"], prof[r["request_id"]])
              for r in rows if r["request_id"] in prof]
    if not joined:
        print("    no request_id overlap -- is this the same run? "
              "(profiles need the request_id-capturing patch)",
              file=sys.stderr)
        return 1
    short = [(f, p) for f, p in joined if p > f]
    equal = sum(1 for f, p in joined if p == f)
    over = sum(1 for f, p in joined if p < f)
    fe_max = max(f for f, _ in joined)
    pr_max = max(p for _, p in joined)
    print(f"    joined: {len(joined)}  frontend<profile: {len(short)}  "
          f"equal: {equal}  frontend>profile: {over}")
    print(f"    max frontend OSL: {fe_max}   max profile OSL: {pr_max}")
    if short:
        ratios = [p / f for f, p in short if f > 0]
        miss = [p - f for f, p in short]
        print(f"    missing tokens per short request: "
              f"p50={_pct(miss, 50):.0f} p90={_pct(miss, 90):.0f} "
              f"max={max(miss)}")
        if ratios:
            print(f"    profile/frontend ratio: p50={_pct(ratios, 50):.1f}x "
                  f"max={max(ratios):.1f}x")

    # -------------------------------------------------- 3. totals
    fe_sum = sum(f for f, _ in joined)
    pr_sum = sum(p for _, p in joined)
    print(f"\n[3] totals over joined requests: frontend={fe_sum}  "
          f"profile={pr_sum}")
    if fe_sum:
        print(f"    frontend captures {100.0 * fe_sum / pr_sum:.1f}% "
              "of the real output tokens")
    if args.metrics_total is not None:
        print(f"    ..._output_tokens_total counter: "
              f"{args.metrics_total:.0f} (whole-process, immune) -- "
              "should be >= the profile sum for this run")

    # Judge by TOKEN share, not request share. The observed failure mode
    # (2026-08-06) truncated only 34/3006 requests -- 1.1%, which a
    # request-count threshold would wave through -- but those 34 were the
    # longest generations, so 47% of all output tokens vanished from the
    # log and every tail/aggregate statistic was wrong.
    req_share = len(short) / len(joined)
    tok_share = 1.0 - (fe_sum / pr_sum) if pr_sum else 0.0
    if tok_share > 0.01 or req_share > 0.01:
        print(f"\nVERDICT: TRUNCATION CONFIRMED -- {len(short)} requests "
              f"({100.0 * req_share:.1f}%) short, but {100.0 * tok_share:.1f}% "
              "of output tokens missing.")
        if req_share < 0.05 <= tok_share:
            print("  Shape: rare but concentrated on the LONGEST "
                  "generations. Central statistics (median OSL/ITL) from "
                  "the frontend log were roughly right; tails (p90/p99/"
                  "max), totals, throughput and aggregate ITL were not.")
        print("  Do not use frontend output_tokens, avg_itl_ms, or the "
              "output_sequence_length histogram. Run e4 with --profiles.")
    else:
        print("\nVERDICT: no meaningful truncation in this log -- frontend "
              "OSL is consistent with the profile usage chunks.")
    if st and sum(v for k, v in st.items()
                  if k.lower() not in ("success", "ok", "(absent)")) \
            < 0.05 * max(total, 1) and short:
        print("  Note: statuses are overwhelmingly success, so the cancel "
              "path (disconnect.rs:272-286) does NOT explain these -- the "
              "drop-order race on the span write is the remaining "
              "mechanism.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
