#!/usr/bin/env python3
"""Break-even model: when does CPU-side processing of a small-LLM turn beat
the GPU path?

Motivation (measured earlier in the pipeline):
  * Under load, a session's KV is evicted while it is away running tools
    (turn_sched.csv: away_s / away_displaced_tokens vs cache_hit_ratio).
  * Re-entry on the GPU then pays: scheduler queue wait + re-prefill of
    the evicted prefix (+ KVBM onboard transfer when tiering is on --
    on-demand at scheduling time, riding inside TTFT).
  * Turns with tiny new-compute and tiny output (identified per preceding
    tool in by_tool.csv) pay that full toll for almost no decode work.

Model per turn (all times ms):
  GPU path  = queue_ms                                (measured)
            + reprefill_tokens / gpu_prefill_tps * 1e3 (recompute of evicted prefix)
            + output_tokens   / gpu_decode_tps  * 1e3
  CPU path  = kv_read_bytes / host_kv_read_gbps       (stream host-resident KV)
            + new_tokens    / cpu_prefill_tps * 1e3   (attention over new tokens only;
                                                       cached prefix KV is NOT recomputed)
            + output_tokens / cpu_decode_tps  * 1e3

where per turn from turn_sched.csv:
  reprefill_tokens = input_tokens (profile input == non-cached tokens)
  new_tokens       = input_tokens  (same quantity -- what must be attended anew)
  kv_read_bytes    = (input+cache_read) * kv_bytes_per_token  (CPU must read the
                     full context KV from host memory for attention)
  queue_ms         = total_queue_ms (measured scheduler wait; assumed avoided on CPU)

CPU throughput knobs are inputs (measure them with a microbench on the
target host and pass here); the script reports, per turn and per
preceding-tool group, which path wins, and solves for the BREAK-EVEN
cpu_decode_tps at which the CPU path matches the GPU path.

Usage:
  scripts/cpu_offload_breakeven.py --turn-sched <run>/turn_sched.csv \\
      [--small-tokens 64] \\
      [--gpu-prefill-tps 8000] [--gpu-decode-tps 60] \\
      [--cpu-prefill-tps 300] [--cpu-decode-tps 8] \\
      [--host-kv-read-gbps 20] [--kv-bytes-per-token 98304] \\
      [--out <dir>]

kv_bytes_per_token default 96 KiB: layers*2(K,V)*kv_heads*head_dim*dtype
-- OVERRIDE with the real model's value (e.g. qwen3-coder-30b-a3b with
GQA + fp16; compute from the model config).

Outputs:
  stdout             per-group win-rate + break-even CPU decode tps
  <out>/breakeven_turns.csv   per-turn costs and winner
  <out>/breakeven_by_tool.csv per-preceding-tool aggregate
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TurnCost:
    session_id: str
    step: int
    prev_tools: str
    new_tokens: int
    cache_read: int
    output_tokens: int
    queue_ms: float
    gpu_ms: float
    cpu_ms: float

    @property
    def winner(self) -> str:
        return "cpu" if self.cpu_ms < self.gpu_ms else "gpu"

    @property
    def small(self) -> bool:
        return False  # set post-hoc; placeholder for csv


def _f(row: dict, key: str, default: float = 0.0) -> float:
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_turn_costs(path: Path, *, gpu_prefill_tps: float, gpu_decode_tps: float,
                    cpu_prefill_tps: float, cpu_decode_tps: float,
                    host_kv_read_gbps: float, kv_bytes_per_token: float,
                    ) -> list[TurnCost]:
    out: list[TurnCost] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            # turn_sched.csv input_tokens column is effective_input
            # (input + cache_read); recover the non-cached portion.
            eff = int(_f(row, "input_tokens"))
            cache = int(_f(row, "cache_read"))
            new_toks = max(0, eff - cache)
            out_toks = int(_f(row, "output_tokens"))
            queue_ms = _f(row, "total_queue_ms")
            gpu_ms = (
                queue_ms
                + new_toks / gpu_prefill_tps * 1e3
                + out_toks / gpu_decode_tps * 1e3
            )
            kv_bytes = eff * kv_bytes_per_token
            cpu_ms = (
                kv_bytes / (host_kv_read_gbps * 1e9) * 1e3
                + new_toks / cpu_prefill_tps * 1e3
                + out_toks / cpu_decode_tps * 1e3
            )
            out.append(TurnCost(
                session_id=row.get("session_id", ""),
                step=int(_f(row, "step")),
                prev_tools=row.get("prev_tools", "(none)"),
                new_tokens=new_toks,
                cache_read=cache,
                output_tokens=out_toks,
                queue_ms=queue_ms,
                gpu_ms=gpu_ms,
                cpu_ms=cpu_ms,
            ))
    return out


def breakeven_cpu_decode_tps(t: TurnCost, *, cpu_prefill_tps: float,
                             host_kv_read_gbps: float,
                             kv_bytes_per_token: float) -> float:
    """Solve for cpu_decode_tps making cpu_ms == gpu_ms for this turn.

    cpu_fixed + out/x*1e3 = gpu_ms  =>  x = out*1e3 / (gpu_ms - cpu_fixed).
    inf when even zero-cost decode cannot win (gpu_ms <= cpu_fixed);
    0 when the turn has no output tokens (any tps wins if fixed < gpu).
    """
    eff = t.new_tokens + t.cache_read
    cpu_fixed = (
        eff * kv_bytes_per_token / (host_kv_read_gbps * 1e9) * 1e3
        + t.new_tokens / cpu_prefill_tps * 1e3
    )
    slack = t.gpu_ms - cpu_fixed
    if slack <= 0:
        return math.inf
    if t.output_tokens <= 0:
        return 0.0
    return t.output_tokens * 1e3 / slack


def by_tool_summary(costs: list[TurnCost], small_tokens: int, **be_kw) -> list[dict]:
    groups: dict[str, list[TurnCost]] = defaultdict(list)
    for t in costs:
        groups[t.prev_tools].append(t)
    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        small = [t for t in items if t.output_tokens <= small_tokens]
        cpu_wins = sum(1 for t in items if t.winner == "cpu")
        cpu_wins_small = sum(1 for t in small if t.winner == "cpu")
        bes = sorted(breakeven_cpu_decode_tps(t, **be_kw) for t in small)
        be_p50 = bes[len(bes) // 2] if bes else math.nan
        rows.append({
            "prev_tools": key,
            "count": len(items),
            "small_count": len(small),
            "cpu_win_rate": cpu_wins / len(items) if items else math.nan,
            "cpu_win_rate_small": (cpu_wins_small / len(small)) if small else math.nan,
            "breakeven_cpu_decode_tps_small_p50": be_p50,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--turn-sched", required=True, type=Path)
    ap.add_argument("--small-tokens", type=int, default=64)
    ap.add_argument("--gpu-prefill-tps", type=float, default=8000.0)
    ap.add_argument("--gpu-decode-tps", type=float, default=60.0)
    ap.add_argument("--cpu-prefill-tps", type=float, default=300.0)
    ap.add_argument("--cpu-decode-tps", type=float, default=8.0)
    ap.add_argument("--host-kv-read-gbps", type=float, default=20.0)
    ap.add_argument("--kv-bytes-per-token", type=float, default=96 * 1024.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if not args.turn_sched.exists():
        print(f"error: turn_sched csv not found: {args.turn_sched}", file=sys.stderr)
        return 2

    costs = load_turn_costs(
        args.turn_sched,
        gpu_prefill_tps=args.gpu_prefill_tps,
        gpu_decode_tps=args.gpu_decode_tps,
        cpu_prefill_tps=args.cpu_prefill_tps,
        cpu_decode_tps=args.cpu_decode_tps,
        host_kv_read_gbps=args.host_kv_read_gbps,
        kv_bytes_per_token=args.kv_bytes_per_token,
    )
    if not costs:
        print("error: no rows in turn_sched csv", file=sys.stderr)
        return 2

    be_kw = dict(cpu_prefill_tps=args.cpu_prefill_tps,
                 host_kv_read_gbps=args.host_kv_read_gbps,
                 kv_bytes_per_token=args.kv_bytes_per_token)
    rows = by_tool_summary(costs, args.small_tokens, **be_kw)

    small = [t for t in costs if t.output_tokens <= args.small_tokens]
    cpu_wins_all = sum(1 for t in costs if t.winner == "cpu")
    cpu_wins_small = sum(1 for t in small if t.winner == "cpu")
    print(f"turns: {len(costs)}  small(<= {args.small_tokens} out-tokens): {len(small)}")
    print(f"cpu wins (given knobs): {cpu_wins_all}/{len(costs)} all, "
          f"{cpu_wins_small}/{len(small)} small")
    print(f"\nknobs: gpu_prefill={args.gpu_prefill_tps:g}tps gpu_decode={args.gpu_decode_tps:g}tps "
          f"cpu_prefill={args.cpu_prefill_tps:g}tps cpu_decode={args.cpu_decode_tps:g}tps "
          f"kv_read={args.host_kv_read_gbps:g}GB/s kv/token={args.kv_bytes_per_token:g}B")
    print(f"\n{'prev_tools':30s} {'n':>5s} {'n_small':>7s} {'cpu_win%':>8s} "
          f"{'cpu_win%_small':>14s} {'BE_cpu_tps(sm,p50)':>18s}")
    for r in rows:
        be = r["breakeven_cpu_decode_tps_small_p50"]
        be_s = "inf" if math.isinf(be) else ("-" if math.isnan(be) else f"{be:.1f}")
        win_s = r["cpu_win_rate_small"]
        win_s_str = f"{100 * win_s:13.1f}%" if not math.isnan(win_s) else f"{'-':>14s}"
        print(f"{r['prev_tools'][:30]:30s} {r['count']:5d} {r['small_count']:7d} "
              f"{100 * r['cpu_win_rate']:7.1f}% {win_s_str} {be_s:>18s}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "breakeven_turns.csv").open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["session_id", "step", "prev_tools", "new_tokens",
                        "cache_read", "output_tokens", "queue_ms",
                        "gpu_ms", "cpu_ms", "winner",
                        "breakeven_cpu_decode_tps"])
            for t in costs:
                be = breakeven_cpu_decode_tps(t, **be_kw)
                w.writerow([t.session_id, t.step, t.prev_tools, t.new_tokens,
                            t.cache_read, t.output_tokens,
                            f"{t.queue_ms:.1f}", f"{t.gpu_ms:.1f}",
                            f"{t.cpu_ms:.1f}", t.winner,
                            "inf" if math.isinf(be) else f"{be:.2f}"])
        with (args.out / "breakeven_by_tool.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out / 'breakeven_turns.csv'}")
        print(f"wrote {args.out / 'breakeven_by_tool.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
