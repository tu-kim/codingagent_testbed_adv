#!/usr/bin/env python3
"""E4: prefill vs decode time distribution from the dynamo frontend log.

Each frontend "request completed" line carries elapsed_ms, ttft_ms and
output_tokens. From those, per request:

  prefill_ms = ttft_ms                 (request received -> first token;
                                        includes engine queue + prefill)
  decode_ms  = elapsed_ms - ttft_ms    (first token -> last token)
  itl_ms     = decode_ms / max(output_tokens - 1, 1)   (per-token latency)
  decode_share = decode_ms / elapsed_ms

Outputs (into --out):
  prefill_decode.csv        per request: request_id, elapsed_ms, prefill_ms,
                            decode_ms, itl_ms, output_tokens, decode_share
  fig1_prefill_decode.pdf   left: prefill_ms vs decode_ms histograms
                            (log x); right: decode-ratio distribution
  fig2_batch_size.pdf       running-batch-size distribution (only with
                            --metrics; vllm:num_requests_running)
  stdout                    mean/p50/p90 of each quantity (+ batch size)

Usage:
  scripts/arm/e4_prefill_decode.py --frontend logs/frontend.log \
      [--metrics logs/vllm_metrics.ndjson] [--out <dir>] [--no-figures]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJ]")
_REQID_RE = re.compile(r'(?:\b|")request_id\b"?\s*[=:]\s*"?(?P<v>[^\s",}]+)"?')
_ELAPSED_RE = re.compile(r'(?:\b|")elapsed_ms\b"?\s*[=:]\s*"?(?P<v>\d+)"?')
_TTFT_RE = re.compile(r'(?:\b|")ttft_ms\b"?\s*[=:]\s*"?(?P<v>[\d.]+)"?')
_OUT_RE = re.compile(r'(?:\b|")output_tokens\b"?\s*[=:]\s*"?(?P<v>\d+)"?')


def parse_frontend(path: Path) -> list[dict]:
    """Per-request dicts with elapsed_ms/ttft_ms/output_tokens and the
    derived prefill_ms/decode_ms/itl_ms/decode_share. Lines missing any
    of the three source fields, or with decode_ms < 0 (clock skew), are
    dropped. Last write wins on duplicate request_id."""
    by_rid: dict[str, dict] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if "request completed" not in line:
                continue
            line = _ANSI_RE.sub("", line)
            rid = _REQID_RE.search(line)
            el = _ELAPSED_RE.search(line)
            tt = _TTFT_RE.search(line)
            ot = _OUT_RE.search(line)
            if not (rid and el and tt and ot):
                continue
            elapsed = float(el.group("v"))
            ttft = float(tt.group("v"))
            out = int(ot.group("v"))
            decode = elapsed - ttft
            if decode < 0:
                continue
            itl = decode / max(out - 1, 1)
            by_rid[rid.group("v")] = {
                "request_id": rid.group("v"),
                "elapsed_ms": elapsed,
                "prefill_ms": ttft,
                "decode_ms": decode,
                "itl_ms": itl,
                "output_tokens": out,
                "decode_share": (decode / elapsed) if elapsed > 0 else 0.0,
            }
    return list(by_rid.values())


def load_batch_sizes(metrics_path: Path,
                     metric: str = "vllm:num_requests_running"
                     ) -> list[float]:
    """Per scrape-tick, per worker: the running-batch size gauge
    (vllm:num_requests_running = requests in the engine's current
    running batch). Each (tick, worker) value is one sample. ok:false
    rows and missing-metric ticks are skipped."""
    import json
    out: list[float] = []
    with metrics_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not row.get("ok"):
                continue
            series = (row.get("metrics") or {}).get(metric)
            if not series:
                continue
            for e in series:
                v = e.get("value")
                if isinstance(v, (int, float)):
                    out.append(float(v))
    return out


def fig_batch_sizes(sizes: list[float], path: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    if sizes:
        hi = int(max(sizes))
        bins = range(0, hi + 2)
        ax.hist(sizes, bins=bins, color="tab:purple", alpha=0.8,
                align="left")
        mean = sum(sizes) / len(sizes)
        ax.axvline(mean, color="tab:red", ls="--", lw=1.2,
                   label=f"mean {mean:.1f}")
        ax.axvline(_pct(sizes, 0.5), color="tab:orange", ls=":", lw=1.2,
                   label=f"p50 {_pct(sizes, 0.5):.0f}")
        ax.legend(fontsize=9, framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "no batch-size samples", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    ax.set_xlabel("running batch size (num_requests_running)")
    ax.set_ylabel("scrape ticks")
    ax.set_title("Running batch size distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _stat_row(name: str, vals: list[float], fmt: str = "{:.1f}") -> None:
    if not vals:
        print(f"  {name:<16} (no data)")
        return
    mean = sum(vals) / len(vals)
    print(f"  {name:<16} mean {fmt.format(mean)}  "
          f"p50 {fmt.format(_pct(vals, 0.5))}  "
          f"p90 {fmt.format(_pct(vals, 0.9))}  n={len(vals)}")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_prefill_decode(rows: list[dict], path: Path) -> None:
    plt = _mpl()
    import numpy as np
    prefill = [r["prefill_ms"] for r in rows]
    decode = [r["decode_ms"] for r in rows]
    share = [r["decode_share"] for r in rows]
    itl = [r["itl_ms"] for r in rows if r["output_tokens"] > 1]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))

    # left: prefill vs decode ms, shared log-spaced bins
    pos = [v for v in prefill + decode if v > 0]
    if pos:
        lo, hi = min(pos), max(pos)
        bins = np.logspace(np.log10(max(lo, 1e-3)), np.log10(hi), 40)
        axL.hist(prefill, bins=bins, alpha=0.55, color="tab:blue",
                 label="prefill")
        axL.hist(decode, bins=bins, alpha=0.55, color="tab:orange",
                 label="decode")
        axL.set_xscale("log")
    axL.set_xlabel("time (ms)")
    axL.set_ylabel("requests")
    for vals, c, name in ((prefill, "tab:blue", "prefill"),
                          (decode, "tab:orange", "decode")):
        if vals:
            mv = sum(vals) / len(vals)
            axL.axvline(mv, color=c, ls="--", lw=1.2)
            axL.text(mv, 0.98, f" {mv:,.0f} ms", color=c,
                     rotation=90, fontsize=8, va="top", ha="left",
                     transform=axL.get_xaxis_transform())
    axL.set_title("Prefill vs decode time distribution")
    axL.legend(fontsize=9, framealpha=0.7)

    # right: decode share of end-to-end + ITL summary
    if share:
        axR.hist([s * 100 for s in share], bins=30, color="tab:green",
                 alpha=0.75)
        m = sum(share) / len(share)
        axR.axvline(m * 100, color="tab:red", ls="--", lw=1.2,
                    label=f"mean {m:.0%}")
        axR.axvline(_pct(share, 0.5) * 100, color="tab:purple", ls=":",
                    lw=1.2, label=f"p50 {_pct(share, 0.5):.0%}")
    axR.set_xlim(0, 100)
    axR.set_xlabel("Decode Ratio (%)")
    axR.set_ylabel("requests")
    axR.set_title("Decode Time Ratio of E2E")
    axR.legend(fontsize=9, framealpha=0.7)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frontend", required=True, type=Path,
                    help="dynamo frontend.log")
    ap.add_argument("--metrics", type=Path, default=None,
                    help="vLLM scrape NDJSON (logs/vllm_metrics.ndjson): "
                         "adds the running-batch-size distribution "
                         "(vllm:num_requests_running)")
    ap.add_argument("--out", type=Path, default=Path("e4_prefill_decode"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    if not args.frontend.is_file():
        print(f"error: frontend log not found: {args.frontend}",
              file=sys.stderr)
        return 2
    rows = parse_frontend(args.frontend)
    if not rows:
        print("error: no 'request completed' lines with "
              "elapsed_ms/ttft_ms/output_tokens", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "prefill_decode.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "request_id", "elapsed_ms", "prefill_ms", "decode_ms",
            "itl_ms", "output_tokens", "decode_share"])
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    print(f"parsed {len(rows)} requests")
    _stat_row("prefill_ms", [r["prefill_ms"] for r in rows])
    _stat_row("decode_ms", [r["decode_ms"] for r in rows])
    _stat_row("elapsed_ms", [r["elapsed_ms"] for r in rows])
    _stat_row("itl_ms", [r["itl_ms"] for r in rows if r["output_tokens"] > 1])
    _stat_row("decode_share", [r["decode_share"] for r in rows], "{:.3f}")
    tot_p = sum(r["prefill_ms"] for r in rows)
    tot_d = sum(r["decode_ms"] for r in rows)
    if tot_p + tot_d > 0:
        print(f"  aggregate prefill:decode = "
              f"{tot_p/(tot_p+tot_d):.1%} : {tot_d/(tot_p+tot_d):.1%}")

    batch_sizes: list[float] = []
    if args.metrics is not None and args.metrics.exists():
        batch_sizes = load_batch_sizes(args.metrics)
        if batch_sizes:
            _stat_row("batch_size", batch_sizes, "{:.1f}")
            print(f"  batch_size min {min(batch_sizes):.0f}  "
                  f"max {max(batch_sizes):.0f}  "
                  f"n={len(batch_sizes)} scrape samples")
        else:
            print("  batch_size: no vllm:num_requests_running in scrape")

    if not args.no_figures:
        try:
            fig_prefill_decode(rows, args.out / "fig1_prefill_decode.pdf")
            if batch_sizes:
                fig_batch_sizes(batch_sizes,
                                args.out / "fig2_batch_size.pdf")
        except ImportError:
            print("matplotlib/numpy unavailable -- figure skipped",
                  file=sys.stderr)
    print(f"outputs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
