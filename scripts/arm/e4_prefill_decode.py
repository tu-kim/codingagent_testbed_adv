#!/usr/bin/env python3
"""E4: prefill vs decode time distribution from the dynamo frontend log.

Each frontend "request completed" line carries elapsed_ms, ttft_ms and
output_tokens. From those, per request:

  prefill_ms = ttft_ms                 (request received -> first token;
                                        includes engine queue + prefill)
  prefill_net_ms = ttft_ms - queue_ms  (queue-removed prefill; needs the
                                        --logs SCHED_DELAY join)
  decode_ms  = elapsed_ms - ttft_ms    (first token -> last token)
  itl_ms     = decode_ms / max(output_tokens - 1, 1)   (per-token latency)
  decode_share = decode_ms / elapsed_ms

Outputs (into --out):
  prefill_decode.csv        per request: request_id, elapsed_ms, prefill_ms,
                            decode_ms, itl_ms, output_tokens, decode_share
  fig1_prefill_decode.pdf   left: prefill_ms vs decode_ms histograms
                            (log x); right: decode-ratio distribution
  fig2_batch_size.pdf       batch size by ROLE (prefill/decode), zeros
                            excluded, from --metrics
                            (vllm:num_requests_running), CLIPPED to the
                            frontend run window (first request start ->
                            last completion) so leading/trailing idle
                            scrape ticks don't skew the distribution
  stdout                    mean/p50/p90 of each quantity (+ batch size)

Note: frontend.log carries NO batch/concurrency field — batch size comes
from the scrape gauge (--metrics). Both are interval snapshots, so short
prefill bursts are under-sampled (see also the run-window clip).

Usage:
  scripts/arm/e4_prefill_decode.py --frontend logs/frontend.log \
      [--metrics logs/vllm_metrics.ndjson] [--out <dir>] [--no-figures]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from pathlib import Path

_ATS_PATH = Path(__file__).resolve().parents[1] / "analyze_turn_scheduling.py"


def _load_ats():
    spec = importlib.util.spec_from_file_location("analyze_turn_scheduling",
                                                  _ATS_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_turn_scheduling"] = mod
    spec.loader.exec_module(mod)
    return mod

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJ]")
_REQID_RE = re.compile(r'(?:\b|")request_id\b"?\s*[=:]\s*"?(?P<v>[^\s",}]+)"?')
_ELAPSED_RE = re.compile(r'(?:\b|")elapsed_ms\b"?\s*[=:]\s*"?(?P<v>\d+)"?')
_TTFT_RE = re.compile(r'(?:\b|")ttft_ms\b"?\s*[=:]\s*"?(?P<v>[\d.]+)"?')
_OUT_RE = re.compile(r'(?:\b|")output_tokens\b"?\s*[=:]\s*"?(?P<v>\d+)"?')
# leading ISO-8601 timestamp of the log line (dynamo default): the moment
# the request COMPLETED.
_ISO_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)")


def _iso_to_unix(s: str) -> float | None:
    from datetime import datetime, timezone
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_frontend(path: Path) -> list[dict]:
    """Per-request dicts with elapsed_ms/ttft_ms/output_tokens and the
    derived prefill_ms/decode_ms/itl_ms/decode_share, plus completed_unix_s
    (the log line's ISO timestamp) when parseable. Lines missing any of
    the three source fields, or with decode_ms < 0 (clock skew), are
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
            tsm = _ISO_RE.match(line)
            completed = _iso_to_unix(tsm.group("ts")) if tsm else None
            by_rid[rid.group("v")] = {
                "request_id": rid.group("v"),
                "elapsed_ms": elapsed,
                "prefill_ms": ttft,
                "decode_ms": decode,
                "itl_ms": itl,
                "output_tokens": out,
                "decode_share": (decode / elapsed) if elapsed > 0 else 0.0,
                "completed_unix_s": completed,
            }
    return list(by_rid.values())


def run_window(rows: list[dict]) -> tuple[float, float] | None:
    """(lo, hi) unix seconds spanning the run: earliest request START
    (completed_unix_s - elapsed_ms/1000) to latest completion. None when
    no line carried a parseable timestamp."""
    starts, ends = [], []
    for r in rows:
        c = r.get("completed_unix_s")
        if c is None:
            continue
        ends.append(c)
        starts.append(c - r["elapsed_ms"] / 1000.0)
    if not ends:
        return None
    return min(starts), max(ends)


def load_batch_sizes(metrics_path: Path,
                     window: tuple[float, float] | None = None,
                     metric: str = "vllm:num_requests_running"
                     ) -> dict[str, list[float]]:
    """{role: [running-batch sizes]} per scrape-tick/worker, grouped by
    the row's `role` (prefill / decode / agg). NON-ZERO samples only —
    a 0 means the engine was idle at that poll (between requests; for
    prefill it's mostly the poller missing the short prefill burst), not
    a real batch. When `window` (lo, hi) is given, only ticks whose ts is
    inside [lo, hi] are kept (clips the scrape's pre/post-run idle tail).
    ok:false rows and missing-metric ticks skipped."""
    import json
    out: dict[str, list[float]] = {}
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
            if window is not None:
                ts = row.get("ts")
                if ts is None or ts < window[0] or ts > window[1]:
                    continue
            series = (row.get("metrics") or {}).get(metric)
            if not series:
                continue
            role = str(row.get("role", "?"))
            for e in series:
                v = e.get("value")
                if isinstance(v, (int, float)) and v > 0:
                    out.setdefault(role, []).append(float(v))
    return out


def load_queue_hist(metrics_path: Path,
                    window: tuple[float, float] | None = None
                    ) -> dict[str, dict]:
    """Per role: engine queue-wait stats over the window, from the
    vllm:request_queue_time_seconds histogram (cumulative _sum/_count/
    _bucket; per-role first/last snapshot inside the window -> deltas).
    Returns {role: {total_s, n_requests, mean_s, p50_s, p90_s}} — p50/p90
    are linear-interpolated within the winning bucket (upper-bounded by
    each bucket's le), NaN when the delta-count is 0. Buckets are summed
    across a role's workers per tick."""
    import json
    import math

    def bucket_le(entry) -> float | None:
        le = (entry.get("labels") or {}).get("le")
        if le is None:
            return None
        return float("inf") if le in ("+Inf", "inf") else float(le)

    # (role) -> {"sum": (first,last), "count": (first,last),
    #            "buckets": {le: (first,last)}}
    acc: dict[str, dict] = {}
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
            if window is not None:
                ts = row.get("ts")
                if ts is None or ts < window[0] or ts > window[1]:
                    continue
            metrics = row.get("metrics") or {}
            role = str(row.get("role", "?"))
            a = acc.setdefault(role, {"sum": None, "count": None,
                                      "buckets": {}})
            for key, name in (("sum", "vllm:request_queue_time_seconds_sum"),
                              ("count",
                               "vllm:request_queue_time_seconds_count")):
                series = metrics.get(name)
                if series:
                    v = sum(e.get("value", 0.0) for e in series
                            if isinstance(e.get("value"), (int, float)))
                    first = a[key][0] if a[key] else v
                    a[key] = (first, v)
            series = metrics.get("vllm:request_queue_time_seconds_bucket")
            if series:
                by_le: dict[float, float] = {}
                for e in series:
                    le = bucket_le(e)
                    v = e.get("value")
                    if le is not None and isinstance(v, (int, float)):
                        by_le[le] = by_le.get(le, 0.0) + float(v)
                for le, v in by_le.items():
                    first = a["buckets"][le][0] if le in a["buckets"] else v
                    a["buckets"][le] = (first, v)

    out: dict[str, dict] = {}
    for role, a in acc.items():
        if not a["sum"] or not a["count"]:
            continue
        d_sum = max(0.0, a["sum"][1] - a["sum"][0])
        d_cnt = max(0.0, a["count"][1] - a["count"][0])
        stats = {"total_s": d_sum, "n_requests": d_cnt,
                 "mean_s": (d_sum / d_cnt) if d_cnt > 0 else math.nan,
                 "p50_s": math.nan, "p90_s": math.nan}
        deltas = sorted((le, max(0.0, last - first))
                        for le, (first, last) in a["buckets"].items())
        total = deltas[-1][1] if deltas else 0.0   # +Inf bucket delta
        if total > 0:
            for pname, q in (("p50_s", 0.5), ("p90_s", 0.9)):
                target = q * total
                prev_le, prev_c = 0.0, 0.0
                for le, c in deltas:
                    if c >= target:
                        span = c - prev_c
                        frac = ((target - prev_c) / span) if span > 0 else 1.0
                        hi = le if le != float("inf") else prev_le
                        stats[pname] = prev_le + (hi - prev_le) * frac
                        break
                    prev_le, prev_c = le, c
        out[role] = stats
    return out


def fig_batch_sizes(by_group: dict[str, list[float]], path: Path,
                    source: str) -> None:
    """Overlaid batch-size histograms, one per group (role or worker)."""
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = {g: v for g, v in by_group.items() if v}
    if groups:
        hi = int(max(max(v) for v in groups.values()))
        bins = range(0, hi + 2)
        palette = ["tab:blue", "tab:orange", "tab:green", "tab:purple",
                   "tab:red", "tab:brown"]
        for i, (g, vals) in enumerate(sorted(groups.items())):
            c = palette[i % len(palette)]
            mean = sum(vals) / len(vals)
            ax.hist(vals, bins=bins, color=c, alpha=0.5, align="left",
                    label=f"{g} (mean {mean:.1f})")
            ax.axvline(mean, color=c, ls="--", lw=1.2)
        ax.legend(fontsize=9, framealpha=0.7)
    else:
        ax.text(0.5, 0.5, "no non-zero batch samples",
                transform=ax.transAxes, ha="center", va="center",
                color="grey")
    ax.set_xlabel("running batch size (non-zero)")
    ax.set_ylabel("samples")
    ax.set_title(f"Running batch size distribution ({source})")
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
    prefill_net = [r["prefill_net_ms"] for r in rows
                   if "prefill_net_ms" in r]
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
        if prefill_net:
            axL.hist(prefill_net, bins=bins, alpha=0.45, color="tab:green",
                     label="prefill w/o queue")
        axL.hist(decode, bins=bins, alpha=0.55, color="tab:orange",
                 label="decode")
        axL.set_xscale("log")
    axL.set_xlabel("time (ms)")
    axL.set_ylabel("requests")
    for vals, c, name in ((prefill, "tab:blue", "prefill"),
                          (prefill_net, "tab:green", "prefill_net"),
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
                         "running-batch-size distribution split by role "
                         "(prefill/decode), zeros excluded, clipped to the "
                         "frontend run window")
    ap.add_argument("--logs", type=Path, default=None,
                    help="worker logs dir (SCHED_DELAY lines): joins the "
                         "engine queue wait by request_id so prefill_net_ms "
                         "= ttft - queue_ms (queue-removed prefill) can be "
                         "computed")
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

    # SCHED_DELAY join: queue-removed prefill (prefill_net = ttft - queue).
    n_q = 0
    if args.logs is not None and args.logs.exists():
        ats = _load_ats()
        sched = ats.load_sched(args.logs)
        for r in rows:
            rec = sched.get(r["request_id"])
            if rec is None:
                continue
            q = rec.total_queue_ms
            if q is not None and 0 <= q <= r["prefill_ms"]:
                r["queue_ms"] = q
                r["prefill_net_ms"] = r["prefill_ms"] - q
                n_q += 1
        print(f"queue join: {n_q}/{len(rows)} requests matched a "
              f"SCHED_DELAY record")

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "prefill_decode.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "request_id", "elapsed_ms", "prefill_ms", "queue_ms",
            "prefill_net_ms", "decode_ms", "itl_ms", "output_tokens",
            "decode_share", "completed_unix_s"], extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})

    print(f"parsed {len(rows)} requests")
    _stat_row("prefill_ms", [r["prefill_ms"] for r in rows])
    if n_q:
        _stat_row("queue_ms", [r["queue_ms"] for r in rows
                               if "queue_ms" in r])
        _stat_row("prefill_net_ms", [r["prefill_net_ms"] for r in rows
                                     if "prefill_net_ms" in r])
    _stat_row("decode_ms", [r["decode_ms"] for r in rows])
    _stat_row("elapsed_ms", [r["elapsed_ms"] for r in rows])
    _stat_row("itl_ms", [r["itl_ms"] for r in rows if r["output_tokens"] > 1])
    _stat_row("decode_share", [r["decode_share"] for r in rows], "{:.3f}")
    tot_p = sum(r["prefill_ms"] for r in rows)
    tot_d = sum(r["decode_ms"] for r in rows)
    if tot_p + tot_d > 0:
        print(f"  aggregate prefill:decode = "
              f"{tot_p/(tot_p+tot_d):.1%} : {tot_d/(tot_p+tot_d):.1%}")

    batch_by_role: dict[str, list[float]] = {}
    if args.metrics is not None and args.metrics.exists():
        window = run_window(rows)
        if window is not None:
            print(f"clip window (frontend): "
                  f"{window[1] - window[0]:.0f}s of activity")
        else:
            print("clip window: no parseable frontend timestamps "
                  "(scrape NOT clipped)")
        batch_by_role = load_batch_sizes(args.metrics, window)
        if batch_by_role:
            print("running batch size by role "
                  "(scrape, zeros excluded, clipped):")
            for role in sorted(batch_by_role):
                vals = batch_by_role[role]
                _stat_row(role, vals, "{:.1f}")
                print(f"    {role} min {min(vals):.0f} max {max(vals):.0f}")
        else:
            print("  batch_size: no non-zero vllm:num_requests_running "
                  "in the window")
        qh = load_queue_hist(args.metrics, window)
        if qh:
            print("engine queue wait by role (scrape "
                  "request_queue_time histogram, clipped):")
            for role in sorted(qh):
                s = qh[role]
                print(f"  {role:<10} total {s['total_s']:.1f}s  "
                      f"n {s['n_requests']:.0f}  mean {s['mean_s']:.3f}s  "
                      f"p50 {s['p50_s']:.3f}s  p90 {s['p90_s']:.3f}s")
        else:
            print("  queue wait: no vllm:request_queue_time_seconds in "
                  "the window")

    if not args.no_figures:
        try:
            fig_prefill_decode(rows, args.out / "fig1_prefill_decode.pdf")
            if batch_by_role:
                fig_batch_sizes(batch_by_role,
                                args.out / "fig2_batch_size.pdf",
                                "scrape num_requests_running, clipped")
        except ImportError:
            print("matplotlib/numpy unavailable -- figure skipped",
                  file=sys.stderr)
    print(f"outputs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
