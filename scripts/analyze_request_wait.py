#!/usr/bin/env python3
"""Per-request queue-wait fraction: join the dynamo frontend log (total
request time) with per-worker SCHED_DELAY lines (prefill/decode
scheduler queue-wait) by request_id.

For each request the dynamo frontend logs one "request completed" line
with `request_id` + `elapsed_ms` (server-side end-to-end wall). The
testbed patch (deploy/patches/dynamo-scheduling-log.patch) makes each
vLLM worker log `SCHED_DELAY request_id=.. role=prefill|decode
queue_ms=..` -- the time that request sat in that worker's scheduler
queue before compute. The request_id is the SAME string across the
frontend and both workers (verified: it's the propagated request
Context id; opencode sends single-prompt requests so there's no
`-<prompt_idx>` suffix to strip).

Joining them yields, per request:
    total_ms          frontend elapsed_ms (end-to-end)
    prefill_wait_ms   Σ prefill-worker queue_ms for this request_id
    decode_wait_ms    Σ decode-worker queue_ms
    total_wait_ms     prefill_wait_ms + decode_wait_ms
    wait_fraction     total_wait_ms / total_ms
The prefill and decode queues happen at different pipeline stages (they
don't overlap), so summing them is the total time spent WAITING in
scheduler queues; wait_fraction is the share of end-to-end spent
waiting rather than computing / transferring.

Per-session: dynamo logs carry NO opencode sessionID (x-session-affinity
is never surfaced dynamo-side). Pass --session-map <csv> with columns
`request_id,session_id` to roll the per-request rows up by session. See
the README/CLAUDE note on producing that map from the opencode profile.

Outputs:
  request_wait.csv     per-request join (one row per frontend request)
  wait_percentiles.csv p50..p99.9/max of wait_fraction / total_wait_ms / total_ms
  wait_tail.csv        slowest-X% buckets: mean wait_fraction + wait_share
  session_wait.csv     per-session rollup (only with --session-map)
  fig_*.pdf            only with --figures (needs matplotlib):
                         fig_wait_vs_total                (scatter, y=x ref)
                         fig_wait_fraction_by_percentile  (rises in the tail)
                         fig_latency_decomposition        (stacked compute|wait)
  stdout               coverage + wait distribution + tail table

Usage:
  scripts/analyze_request_wait.py \\
      --frontend logs/frontend.log \\
      --logs logs/ \\
      --output results/run1/wait \\
      [--session-map results/run1/req_to_session.csv] \\
      [--figures]
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------- frontend log ----------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJ]")
# Tolerant k=v / k="v" matcher (mirrors analyze_frontend_log).
_REQID_RE = re.compile(r'(?:\b|")request_id\b"?\s*[=:]\s*"?(?P<v>[^\s",}]+)"?')
_ELAPSED_RE = re.compile(r'(?:\b|")elapsed_ms\b"?\s*[=:]\s*"?(?P<v>\d+)"?')
_TTFT_RE = re.compile(r'(?:\b|")ttft_ms\b"?\s*[=:]\s*"?(?P<v>[\d.]+)"?')


@dataclass
class FrontendReq:
    request_id: str
    total_ms: float
    ttft_ms: float | None


def parse_frontend(path: Path) -> dict[str, FrontendReq]:
    """{request_id: FrontendReq} from the frontend log's
    'request completed' lines. Last write wins on duplicate id."""
    out: dict[str, FrontendReq] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if "request completed" not in line:
                continue
            line = _ANSI_RE.sub("", line)
            rid = _REQID_RE.search(line)
            elapsed = _ELAPSED_RE.search(line)
            if not rid or not elapsed:
                continue
            ttft = _TTFT_RE.search(line)
            out[rid.group("v")] = FrontendReq(
                request_id=rid.group("v"),
                total_ms=float(elapsed.group("v")),
                ttft_ms=float(ttft.group("v")) if ttft else None,
            )
    return out


# ---------- worker SCHED_DELAY logs ----------

_SCHED_RE = re.compile(
    r"SCHED_DELAY\s+request_id=(?P<rid>\S+)\s+role=(?P<role>\S+)\s+"
    r"queue_ms=(?P<queue_ms>[-0-9.eE+]+)"
)


@dataclass
class WorkerWait:
    prefill_ms: list[float] = field(default_factory=list)
    decode_ms: list[float] = field(default_factory=list)


def _iter_log_files(path: Path):
    if path.is_file():
        yield path
        return
    for f in sorted(path.glob("vllm-*.log")):
        yield f


def parse_worker_waits(path: Path) -> dict[str, WorkerWait]:
    """{request_id: WorkerWait} accumulating queue_ms per role across
    ALL prefill/decode worker logs (the fleet may have many of each;
    one request normally hits one prefill + one decode, but we sum
    defensively in case a request is re-prefilled / re-scheduled)."""
    out: dict[str, WorkerWait] = defaultdict(WorkerWait)
    for fpath in _iter_log_files(path):
        with fpath.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if "SCHED_DELAY" not in line:
                    continue
                m = _SCHED_RE.search(line)
                if not m:
                    continue
                try:
                    q = float(m.group("queue_ms"))
                except ValueError:
                    continue
                ww = out[m.group("rid")]
                if m.group("role") == "prefill":
                    ww.prefill_ms.append(q)
                elif m.group("role") == "decode":
                    ww.decode_ms.append(q)
    return dict(out)


# ---------- join ----------


@dataclass
class RequestWait:
    request_id: str
    total_ms: float
    ttft_ms: float | None
    prefill_wait_ms: float | None
    decode_wait_ms: float | None

    @property
    def total_wait_ms(self) -> float:
        return (self.prefill_wait_ms or 0.0) + (self.decode_wait_ms or 0.0)

    @property
    def matched(self) -> bool:
        return self.prefill_wait_ms is not None or self.decode_wait_ms is not None

    @property
    def wait_fraction(self) -> float | None:
        if self.total_ms <= 0:
            return None
        return self.total_wait_ms / self.total_ms


def join_requests(frontend: dict[str, FrontendReq],
                  waits: dict[str, WorkerWait]) -> list[RequestWait]:
    rows: list[RequestWait] = []
    for rid, fr in frontend.items():
        ww = waits.get(rid)
        prefill = sum(ww.prefill_ms) if (ww and ww.prefill_ms) else None
        decode = sum(ww.decode_ms) if (ww and ww.decode_ms) else None
        rows.append(RequestWait(
            request_id=rid,
            total_ms=fr.total_ms,
            ttft_ms=fr.ttft_ms,
            prefill_wait_ms=prefill,
            decode_wait_ms=decode,
        ))
    rows.sort(key=lambda r: r.request_id)
    return rows


# ---------- session rollup ----------


def load_session_map(path: Path) -> dict[str, str]:
    """CSV with header columns request_id,session_id."""
    out: dict[str, str] = {}
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.DictReader(f):
            rid = (row.get("request_id") or "").strip()
            sid = (row.get("session_id") or "").strip()
            if rid and sid:
                out[rid] = sid
    return out


@dataclass
class SessionWait:
    session_id: str
    n_requests: int
    total_ms: float
    total_wait_ms: float

    @property
    def wait_fraction(self) -> float | None:
        if self.total_ms <= 0:
            return None
        return self.total_wait_ms / self.total_ms


def rollup_sessions(rows: list[RequestWait],
                    req_to_session: dict[str, str]) -> list[SessionWait]:
    agg: dict[str, list[RequestWait]] = defaultdict(list)
    for r in rows:
        sid = req_to_session.get(r.request_id)
        if sid is not None:
            agg[sid].append(r)
    out = [
        SessionWait(
            session_id=sid,
            n_requests=len(rs),
            total_ms=sum(r.total_ms for r in rs),
            total_wait_ms=sum(r.total_wait_ms for r in rs),
        )
        for sid, rs in agg.items()
    ]
    out.sort(key=lambda s: s.session_id)
    return out


# ---------- output ----------


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_request_csv(rows: list[RequestWait], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["request_id", "matched", "total_ms", "ttft_ms",
                    "prefill_wait_ms", "decode_wait_ms",
                    "total_wait_ms", "wait_fraction"])
        for r in rows:
            w.writerow([
                r.request_id, int(r.matched), _fmt(r.total_ms), _fmt(r.ttft_ms),
                _fmt(r.prefill_wait_ms), _fmt(r.decode_wait_ms),
                _fmt(r.total_wait_ms), _fmt(r.wait_fraction),
            ])


def write_session_csv(rows: list[SessionWait], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["session_id", "n_requests", "total_ms",
                    "total_wait_ms", "wait_fraction"])
        for s in rows:
            w.writerow([s.session_id, s.n_requests, _fmt(s.total_ms),
                        _fmt(s.total_wait_ms), _fmt(s.wait_fraction)])


def _pctiles(values: list[float]) -> str:
    if not values:
        return "(none)"
    a = np.asarray(values, dtype=float)
    return (f"n={a.size} mean={a.mean():.4f} p50={np.percentile(a,50):.4f} "
            f"p90={np.percentile(a,90):.4f} p99={np.percentile(a,99):.4f} "
            f"max={a.max():.4f}")


# ---------- tail analysis ----------
#
# The "tail grows" argument: it's not the typical request that hurts, it's
# the slow ones -- and the slow ones are slow BECAUSE of queue wait, not
# compute. Two complementary views below make that case from the
# per-request join:
#   percentile_summary  -- the wait distribution itself, out to p99.9.
#   tail_buckets        -- take the slowest X% of requests (by e2e) and
#                          show (a) their mean wait_fraction and (b) what
#                          share of ALL wait time they concentrate. If the
#                          slowest 1% hold a wait_share far above 1%, the
#                          tail is wait-dominated.

_PCTILES = (50.0, 90.0, 95.0, 99.0, 99.9)


def percentile_summary(values: list[float]) -> dict[str, float | int] | None:
    """{n, mean, p50, p90, p95, p99, p99.9, max} or None if empty."""
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    out: dict[str, float | int] = {"n": int(a.size), "mean": float(a.mean())}
    for p in _PCTILES:
        out[f"p{p:g}"] = float(np.percentile(a, p))
    out["max"] = float(a.max())
    return out


@dataclass
class TailBucket:
    frac: float            # e.g. 0.01 = slowest 1% by e2e
    k: int                 # number of requests in the bucket
    mean_total_ms: float
    mean_wait_fraction: float
    wait_share: float | None   # bucket's share of total wait across all matched


def tail_buckets(rows: list[RequestWait],
                 fracs: tuple[float, ...] = (0.10, 0.05, 0.01)) -> list[TailBucket]:
    """For each tail fraction, take the slowest ceil(frac*n) requests by
    e2e total_ms and summarize their wait. `wait_share` = this bucket's
    total wait / all matched requests' total wait -- a number well above
    `frac` means the tail concentrates a disproportionate amount of the
    queue wait (the point of the 'tail grows' story)."""
    matched = [r for r in rows if r.matched and r.wait_fraction is not None]
    if not matched:
        return []
    s = sorted(matched, key=lambda r: r.total_ms, reverse=True)
    n = len(s)
    all_wait = sum(r.total_wait_ms for r in s)
    out: list[TailBucket] = []
    for frac in fracs:
        k = max(1, math.ceil(frac * n))
        top = s[:k]
        bucket_wait = sum(r.total_wait_ms for r in top)
        out.append(TailBucket(
            frac=frac,
            k=k,
            mean_total_ms=float(np.mean([r.total_ms for r in top])),
            mean_wait_fraction=float(np.mean([r.wait_fraction for r in top])),
            wait_share=(bucket_wait / all_wait) if all_wait else None,
        ))
    return out


def write_percentiles_csv(rows: list[RequestWait], path: Path) -> None:
    """Percentile table (p50..p99.9, max) for the three per-request
    metrics, one row per metric."""
    matched = [r for r in rows if r.matched]
    metrics = {
        "wait_fraction": [r.wait_fraction for r in matched if r.wait_fraction is not None],
        "total_wait_ms": [r.total_wait_ms for r in matched],
        "total_ms": [r.total_ms for r in matched],
    }
    cols = ["metric", "n", "mean"] + [f"p{p:g}" for p in _PCTILES] + ["max"]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, vals in metrics.items():
            s = percentile_summary(vals)
            if s is None:
                w.writerow([name, 0] + [""] * (len(cols) - 2))
                continue
            w.writerow([name, s["n"], _fmt(s["mean"])]
                       + [_fmt(s[f"p{p:g}"]) for p in _PCTILES]
                       + [_fmt(s["max"])])


def write_tail_csv(buckets: list[TailBucket], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tail_frac", "k_requests", "mean_total_ms",
                    "mean_wait_fraction", "wait_share"])
        for b in buckets:
            w.writerow([f"{b.frac:g}", b.k, _fmt(b.mean_total_ms),
                        _fmt(b.mean_wait_fraction), _fmt(b.wait_share)])


# ---------- figures (opt-in: --figures; matplotlib imported lazily) ----------

_PAPER_STYLE = {
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "figure.figsize": (3.4, 2.4),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
}


def _setup_plt():
    """Lazy matplotlib import with a headless backend so the CSV/stdout
    path never requires matplotlib (only --figures does)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(_PAPER_STYLE)
    return plt


def _matched_sorted(rows: list[RequestWait]) -> list[RequestWait]:
    return sorted(
        (r for r in rows if r.matched and r.wait_fraction is not None),
        key=lambda r: r.total_ms,
    )


def plot_wait_vs_total(rows: list[RequestWait], out: Path) -> Path | None:
    """Scatter: e2e (x) vs queue wait (y), coloured by wait fraction, with
    the y=x reference. Points hugging y=x at the high-x end ⇒ the slowest
    requests are slow because they're WAITING, not computing."""
    m = _matched_sorted(rows)
    if not m:
        return None
    plt = _setup_plt()
    x = [r.total_ms for r in m]
    y = [r.total_wait_ms for r in m]
    c = [r.wait_fraction for r in m]
    fig, ax = plt.subplots()
    sc = ax.scatter(x, y, c=c, s=9, cmap="viridis", vmin=0.0, vmax=1.0,
                    alpha=0.75, linewidths=0)
    lim = max(max(x), max(y)) or 1.0
    ax.plot([0, lim], [0, lim], ls="--", lw=0.6, color="0.5", zorder=0)
    ax.set_xlabel("end-to-end latency (ms)")
    ax.set_ylabel("scheduler queue wait (ms)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("wait fraction")
    p = out / "fig_wait_vs_total.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_wait_fraction_by_percentile(rows: list[RequestWait], out: Path) -> Path | None:
    """wait_fraction against e2e-latency percentile. A curve that rises
    toward the right = the tail of the latency distribution is
    increasingly wait-dominated."""
    m = _matched_sorted(rows)
    if not m:
        return None
    plt = _setup_plt()
    n = len(m)
    pct = [100.0 * (i + 0.5) / n for i in range(n)]
    wf = [r.wait_fraction for r in m]
    fig, ax = plt.subplots()
    ax.plot(pct, wf, lw=1.0)
    ax.set_xlabel("e2e latency percentile")
    ax.set_ylabel("wait fraction")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    p = out / "fig_wait_fraction_by_percentile.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def plot_latency_decomposition(rows: list[RequestWait], out: Path) -> Path | None:
    """Stacked area over the e2e percentile axis: compute+transfer (e2e
    minus queue wait) on the bottom, queue wait on top. The wait band
    thickening toward the right is the visual 'tail grows' statement."""
    m = _matched_sorted(rows)
    if not m:
        return None
    plt = _setup_plt()
    n = len(m)
    pct = [100.0 * (i + 0.5) / n for i in range(n)]
    wait = [r.total_wait_ms for r in m]
    compute = [max(0.0, r.total_ms - r.total_wait_ms) for r in m]
    fig, ax = plt.subplots()
    ax.stackplot(pct, compute, wait,
                 labels=["compute + transfer", "queue wait"],
                 colors=["#9ecae1", "#e6550d"])
    ax.set_xlabel("e2e latency percentile")
    ax.set_ylabel("latency (ms)")
    ax.set_xlim(0, 100)
    ax.legend(loc="upper left", frameon=False)
    p = out / "fig_latency_decomposition.pdf"
    fig.savefig(p)
    plt.close(fig)
    return p


def make_figures(rows: list[RequestWait], out: Path) -> list[Path]:
    paths: list[Path] = []
    for fn in (plot_wait_vs_total,
               plot_wait_fraction_by_percentile,
               plot_latency_decomposition):
        p = fn(rows, out)
        if p is not None:
            paths.append(p)
    return paths


def print_summary(rows: list[RequestWait], sessions: list[SessionWait] | None) -> None:
    matched = [r for r in rows if r.matched]
    print()
    print(f"Frontend requests:        {len(rows)}")
    print(f"Joined with SCHED_DELAY:  {len(matched)} "
          f"({100*len(matched)/len(rows):.0f}%)" if rows else "")
    if len(matched) < len(rows):
        print(f"  unmatched (no worker queue line): {len(rows) - len(matched)} "
              f"-- patch applied + workers restarted? cache-only reqs no-op.")
    print()
    fr = [r.wait_fraction for r in matched if r.wait_fraction is not None]
    pre = [r.prefill_wait_ms for r in matched if r.prefill_wait_ms is not None]
    dec = [r.decode_wait_ms for r in matched if r.decode_wait_ms is not None]
    print(f"wait_fraction (queue wait / e2e):  {_pctiles(fr)}")
    print(f"prefill_wait_ms:                   {_pctiles(pre)}")
    print(f"decode_wait_ms:                    {_pctiles(dec)}")
    tot_e2e = sum(r.total_ms for r in matched)
    tot_wait = sum(r.total_wait_ms for r in matched)
    if tot_e2e:
        print()
        print(f"Aggregate: {tot_wait:.1f}ms wait / {tot_e2e:.1f}ms e2e "
              f"= {100*tot_wait/tot_e2e:.1f}% of total time in scheduler queues")

    # Tail view: are the SLOWEST requests slow because of queue wait?
    buckets = tail_buckets(rows)
    if buckets:
        print()
        print("Tail (slowest X% by e2e) -- wait_share >> X% ⇒ tail is wait-dominated:")
        hdr = (f"{'tail':>6} {'k':>5} {'mean_e2e_ms':>12} "
               f"{'mean_wait_frac':>15} {'wait_share':>11}")
        print(hdr)
        print("-" * len(hdr))
        for b in buckets:
            share = "-" if b.wait_share is None else f"{100*b.wait_share:.1f}%"
            print(f"{100*b.frac:>5.0f}% {b.k:>5} {b.mean_total_ms:>12.1f} "
                  f"{b.mean_wait_fraction:>14.3f} {share:>11}")

    if sessions is not None:
        print()
        sfr = [s.wait_fraction for s in sessions if s.wait_fraction is not None]
        print(f"Sessions: {len(sessions)}")
        print(f"per-session wait_fraction:         {_pctiles(sfr)}")
    print()


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frontend", required=True, type=Path,
                    help="dynamo frontend.log")
    ap.add_argument("--logs", required=True, type=Path,
                    help="Directory of vllm-*.log (or a single worker log)")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory (created if missing)")
    ap.add_argument("--session-map", type=Path, default=None,
                    help="Optional CSV (request_id,session_id) to roll up "
                         "per-request rows by opencode session")
    ap.add_argument("--figures", action="store_true",
                    help="Also render PDF figures (scatter, wait-fraction-by-"
                         "percentile, stacked latency decomposition). Requires "
                         "matplotlib; the CSV/stdout path does not.")
    args = ap.parse_args(argv)

    if not args.frontend.exists():
        print(f"frontend log not found: {args.frontend}", file=sys.stderr)
        return 2
    if not args.logs.exists():
        print(f"logs path not found: {args.logs}", file=sys.stderr)
        return 2
    if args.session_map is not None and not args.session_map.exists():
        print(f"session map not found: {args.session_map}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    frontend = parse_frontend(args.frontend)
    if not frontend:
        print("no 'request completed' lines in frontend log", file=sys.stderr)
        return 1
    waits = parse_worker_waits(args.logs)
    rows = join_requests(frontend, waits)

    req_csv = args.output / "request_wait.csv"
    write_request_csv(rows, req_csv)
    print(f"  wrote {req_csv}")

    pct_csv = args.output / "wait_percentiles.csv"
    write_percentiles_csv(rows, pct_csv)
    print(f"  wrote {pct_csv}")

    tail_csv = args.output / "wait_tail.csv"
    write_tail_csv(tail_buckets(rows), tail_csv)
    print(f"  wrote {tail_csv}")

    if args.figures:
        for p in make_figures(rows, args.output):
            print(f"  wrote {p}")

    sessions = None
    if args.session_map is not None:
        sessions = rollup_sessions(rows, load_session_map(args.session_map))
        sess_csv = args.output / "session_wait.csv"
        write_session_csv(sessions, sess_csv)
        print(f"  wrote {sess_csv}")

    print_summary(rows, sessions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
