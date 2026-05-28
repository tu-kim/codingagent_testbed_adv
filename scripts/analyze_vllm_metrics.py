#!/usr/bin/env python3
"""Window-stats for vLLM /metrics scrape NDJSON (scrape_vllm_metrics.py).

Two modes (same shape as analyze_session_resources.py):

  Session-window mode (--profile <session.jsonl>):
    Joins scrape NDJSON with profile NDJSON on wall-clock ts, keeping
    only samples inside one session's `query.start` → `query.end`
    window.

  All-points mode (omit --profile):
    Aggregate stats across EVERY scrape tick in the input, regardless
    of session boundaries.

Metric handling (suffix-based, Prometheus convention):

  Gauge  (`vllm:num_requests_running`, `vllm:kv_cache_usage_perc`, …)
    → mean/median/p90/p99/max across the per-tick sample values
      (NOT counter-style; we treat each scrape as an independent
      observation of the gauge).
  Counter (`*_total`, e.g. `vllm:prompt_tokens_total`)
    → `delta` = last - first within window (clipped at 0 on resets)
    → `rate_per_s` = delta / window_duration_s
  Histogram (`<base>_bucket` + `<base>_count` + `<base>_sum`)
    → Build cumulative bucket deltas (last - first) per `le`
    → mean = sum_delta / count_delta
    → p50 / p90 / p99 via linear interpolation across bucket deltas
    → `delta` = count_delta (number of observations in the window)

Per (worker, role, metric, label-set excluding `le`), one row of stats.
A role aggregate (`<role>.<metric>`) is also emitted that combines all
workers sharing the role:
  - Gauge: per-tick mean across workers, then percentiles of that
  - Counter: sum of per-worker deltas
  - Histogram: sum of per-worker bucket-deltas; percentiles re-derived

Usage:
  # Session window
  scripts/analyze_vllm_metrics.py \\
      --profile /tmp/testbed-workspaces/profiles/ses_xxx.jsonl \\
      --metrics logs/vllm_metrics.ndjson \\
      --output results/run1/session_xxx_vllm

  # All measured ticks
  scripts/analyze_vllm_metrics.py \\
      --metrics logs/vllm_metrics.ndjson \\
      --output results/run1/global_vllm

Outputs:
  vllm_metrics_stats.csv   per (worker, role, metric, labels) stats
  stdout                   pretty table
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------- profile -> session window (mirrors analyze_session_resources) ----------


def load_session_windows(profile_path: Path) -> dict[str, tuple[float, float]]:
    """{sessionID: (query_start_ts, query_end_ts)} for every session that
    has both events. Sessions missing query.end (mid-flight or crashed)
    fall back to (start, last_ts_seen)."""
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    last_ts: dict[str, float] = {}
    with profile_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = ev.get("sessionID")
            ts = ev.get("ts")
            if sid is None or ts is None:
                continue
            last_ts[sid] = max(last_ts.get(sid, ts), ts)
            ev_type = ev.get("ev")
            if ev_type == "query.start":
                starts[sid] = ts
            elif ev_type == "query.end":
                ends[sid] = ts
    return {sid: (s, ends.get(sid, last_ts.get(sid, s))) for sid, s in starts.items()}


# ---------- scrape NDJSON ingest ----------


def load_rows(metrics_path: Path, start_ts: float, end_ts: float) -> list[dict]:
    """Stream scrape NDJSON, keep rows with `ok=True` whose ts is inside
    [start_ts, end_ts]. Failed scrapes (`ok=False`) are dropped — they
    have no `metrics` field."""
    out: list[dict] = []
    with metrics_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = row.get("ts")
            if not isinstance(ts, (int, float)):
                continue
            if not (start_ts <= ts <= end_ts):
                continue
            if not row.get("ok"):
                continue
            if "metrics" not in row:
                continue
            out.append(row)
    return out


# ---------- metric type classification ----------


def classify_metrics(metric_names: set[str]) -> dict[str, str]:
    """Map metric_name -> one of:
        'gauge' | 'counter' | 'histogram_bucket' | 'histogram_count' | 'histogram_sum'
    Histogram detection: if `<base>_bucket` exists for some base, then
    `<base>_count` and `<base>_sum` are bound to that histogram.
    """
    hist_bases: set[str] = {n[:-7] for n in metric_names if n.endswith("_bucket")}
    out: dict[str, str] = {}
    for n in metric_names:
        if n.endswith("_bucket") and n[:-7] in hist_bases:
            out[n] = "histogram_bucket"
        elif n.endswith("_count") and n[:-6] in hist_bases:
            out[n] = "histogram_count"
        elif n.endswith("_sum") and n[:-4] in hist_bases:
            out[n] = "histogram_sum"
        elif n.endswith("_total"):
            out[n] = "counter"
        else:
            out[n] = "gauge"
    return out


def _label_key(labels: dict, exclude: tuple[str, ...] = ()) -> str:
    """Unambiguous stable key for a label dict, with `exclude` removed.
    Empty dict → "" (so the "no labels" series sorts first).

    Uses JSON to encode so label values containing `,` or `=` can't
    collide with adjacent keys (e.g. `{"a":"b,c=d"}` vs `{"a":"b","c":"d"}`
    would otherwise both stringify to `a=b,c=d`)."""
    items = sorted((k, v) for k, v in labels.items() if k not in exclude)
    if not items:
        return ""
    return json.dumps(dict(items), ensure_ascii=False, separators=(",", ":"))


# ---------- per-series accumulators ----------


# Series key: (worker, role, metric_name, label_key_no_le)
SeriesKey = tuple[str, str, str, str]


def _series_key(row: dict, metric: str, labels: dict) -> SeriesKey:
    return (
        str(row.get("worker", "?")),
        str(row.get("role", "?")),
        metric,
        _label_key(labels, exclude=("le",)),
    )


def _parse_le(label_val) -> float:
    """Prometheus encodes `+Inf` as the string `+Inf`. Anything else
    parses as float."""
    if isinstance(label_val, (int, float)):
        return float(label_val)
    s = str(label_val).strip()
    if s in ("+Inf", "Inf", "inf"):
        return math.inf
    if s in ("-Inf", "-inf"):
        return -math.inf
    return float(s)


def collect_series(rows: list[dict], cls: dict[str, str]):
    """Walk rows once, partition into three buckets keyed by SeriesKey:
        gauges[key]      = [(ts, value), ...]
        counters[key]    = [(ts, value), ...]   # raw cumulative samples
        histograms[key]  = {
            "count":   [(ts, value), ...],
            "sum":     [(ts, value), ...],
            "buckets": {le_float: [(ts, value), ...]},
        }
    """
    gauges: dict[SeriesKey, list[tuple[float, float]]] = defaultdict(list)
    counters: dict[SeriesKey, list[tuple[float, float]]] = defaultdict(list)
    histograms: dict[SeriesKey, dict] = defaultdict(
        lambda: {"count": [], "sum": [], "buckets": defaultdict(list)}
    )

    for row in rows:
        ts = float(row["ts"])
        metrics = row.get("metrics") or {}
        for name, entries in metrics.items():
            kind = cls.get(name, "gauge")
            for entry in entries:
                labels = entry.get("labels") or {}
                value = entry.get("value")
                if not isinstance(value, (int, float)):
                    continue
                if math.isnan(value) or math.isinf(value):
                    # +Inf / NaN show up in some vLLM gauges (e.g. when no
                    # requests have completed yet). Skip — they pollute
                    # percentiles.
                    continue
                key = _series_key(row, name, labels)
                if kind == "gauge":
                    gauges[key].append((ts, float(value)))
                elif kind == "counter":
                    counters[key].append((ts, float(value)))
                elif kind == "histogram_count":
                    histograms[key]["count"].append((ts, float(value)))
                elif kind == "histogram_sum":
                    histograms[key]["sum"].append((ts, float(value)))
                elif kind == "histogram_bucket":
                    le = labels.get("le")
                    if le is None:
                        continue
                    try:
                        le_f = _parse_le(le)
                    except ValueError:
                        continue
                    # For histogram bucket keys, strip the `le` and the
                    # `_bucket` suffix so all three (bucket/count/sum)
                    # share the same SeriesKey.
                    base_metric = name[:-7]  # drop "_bucket"
                    key = _series_key(row, base_metric, labels)
                    histograms[key]["buckets"][le_f].append((ts, float(value)))
    return gauges, counters, histograms


# ---------- stats ----------


def gauge_stats(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": None, "median": None,
                "p90": None, "p99": None, "max": None}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(np.max(arr)),
    }


def counter_stats(samples: list[tuple[float, float]]) -> dict:
    """delta = last - first (clip at 0 to absorb process restarts);
    rate_per_s = delta / (last_ts - first_ts) when duration > 0."""
    if len(samples) < 2:
        return {"n": len(samples), "delta": None, "rate_per_s": None}
    samples = sorted(samples, key=lambda x: x[0])
    first_ts, first_v = samples[0]
    last_ts, last_v = samples[-1]
    delta = max(0.0, last_v - first_v)
    dur = last_ts - first_ts
    rate = (delta / dur) if dur > 0 else None
    return {"n": len(samples), "delta": delta, "rate_per_s": rate,
            "first": first_v, "last": last_v,
            "first_ts": first_ts, "last_ts": last_ts}


def histogram_quantile(q: float,
                       buckets: list[tuple[float, float]],
                      ) -> float | None:
    """Standard Prometheus linear interpolation.

    `buckets` = [(upper_bound, cumulative_count), ...] sorted ascending
    by upper_bound. Includes the implicit +Inf bucket (with total count).
    Returns the q-th quantile, or None if the histogram is empty.
    """
    if not buckets:
        return None
    total = buckets[-1][1]
    if total <= 0:
        return None
    target = q * total
    prev_ub, prev_cum = 0.0, 0.0
    for ub, cum in buckets:
        if cum >= target:
            if math.isinf(ub):
                return prev_ub if prev_ub > 0 else None
            if cum == prev_cum:
                return ub
            frac = (target - prev_cum) / (cum - prev_cum)
            return prev_ub + frac * (ub - prev_ub)
        prev_ub, prev_cum = ub, cum
    return buckets[-1][0]


def histogram_stats(hist: dict) -> dict:
    """Reduce a single histogram series to {n, mean, median, p90, p99,
    max, delta, sum}. `delta` = count delta (= number of observations
    in the window). `max` is the highest finite bucket upper bound that
    received any new observations."""
    count_samples = sorted(hist.get("count") or [], key=lambda x: x[0])
    sum_samples = sorted(hist.get("sum") or [], key=lambda x: x[0])
    bucket_samples = hist.get("buckets") or {}

    if len(count_samples) < 2:
        return {"n": len(count_samples), "delta": None, "sum": None,
                "mean": None, "median": None, "p90": None, "p99": None,
                "max": None}

    count_delta = max(0.0, count_samples[-1][1] - count_samples[0][1])
    sum_delta = (
        max(0.0, sum_samples[-1][1] - sum_samples[0][1])
        if len(sum_samples) >= 2 else None
    )

    # Build bucket-delta CDF: for each le, (last - first).
    bucket_deltas: list[tuple[float, float]] = []
    for le, samples in bucket_samples.items():
        samples = sorted(samples, key=lambda x: x[0])
        if len(samples) < 2:
            continue
        delta = max(0.0, samples[-1][1] - samples[0][1])
        bucket_deltas.append((le, delta))
    bucket_deltas.sort(key=lambda x: x[0])

    if count_delta <= 0 or not bucket_deltas:
        return {"n": len(count_samples), "delta": count_delta, "sum": sum_delta,
                "mean": None, "median": None, "p90": None, "p99": None,
                "max": None}

    # The bucket deltas above are already cumulative within their `le`
    # (since Prometheus histogram buckets are cumulative per scrape).
    # No further prefix-sum needed.
    mean = (sum_delta / count_delta) if (sum_delta and count_delta > 0) else None

    # `max` = highest finite bucket that saw an increment in the window.
    finite_with_inc = [(le, d) for le, d in bucket_deltas
                       if not math.isinf(le) and d > 0]
    max_val = finite_with_inc[-1][0] if finite_with_inc else None

    return {
        "n": len(count_samples),
        "delta": count_delta,
        "sum": sum_delta,
        "mean": mean,
        "median": histogram_quantile(0.50, bucket_deltas),
        "p90": histogram_quantile(0.90, bucket_deltas),
        "p99": histogram_quantile(0.99, bucket_deltas),
        "max": max_val,
    }


# ---------- role aggregation ----------


def aggregate_by_role(
    gauges, counters, histograms,
    cls: dict[str, str],
):
    """Combine per-worker series into per-role series.
    Returns (role_gauges, role_counters, role_histograms) keyed by
    SeriesKey with worker="*" (and role kept as the actual role).
    """
    # Gauges: per-tick mean across workers in the role
    role_g_per_ts: dict[SeriesKey, dict[float, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (worker, role, metric, lk), samples in gauges.items():
        rk = ("*", role, metric, lk)
        for ts, v in samples:
            role_g_per_ts[rk][ts].append(v)
    role_gauges: dict[SeriesKey, list[tuple[float, float]]] = {}
    for rk, by_ts in role_g_per_ts.items():
        role_gauges[rk] = [(ts, float(np.mean(vs))) for ts, vs in sorted(by_ts.items())]

    # Counters: sum of per-worker deltas. Treat as one synthetic series
    # whose stats are computed by summing deltas (rate = sum_delta / max
    # duration across workers).
    role_c_acc: dict[SeriesKey, list[dict]] = defaultdict(list)
    for (worker, role, metric, lk), samples in counters.items():
        rk = ("*", role, metric, lk)
        role_c_acc[rk].append(counter_stats(samples))
    role_counters_stats: dict[SeriesKey, dict] = {}
    for rk, per_worker in role_c_acc.items():
        valid = [s for s in per_worker if s.get("delta") is not None]
        if not valid:
            role_counters_stats[rk] = {"n": 0, "delta": None, "rate_per_s": None}
            continue
        total_delta = sum(s["delta"] for s in valid)
        dur = max(s["last_ts"] - s["first_ts"] for s in valid)
        rate = (total_delta / dur) if dur > 0 else None
        role_counters_stats[rk] = {
            "n": sum(s["n"] for s in valid),
            "delta": total_delta,
            "rate_per_s": rate,
        }

    # Histograms: sum the per-worker bucket-deltas / count-deltas /
    # sum-deltas, then re-derive quantiles.
    role_h_acc: dict[SeriesKey, dict] = defaultdict(
        lambda: {"count_delta": 0.0, "sum_delta": 0.0,
                 "bucket_deltas": defaultdict(float), "n_workers": 0}
    )
    for (worker, role, metric, lk), hist in histograms.items():
        rk = ("*", role, metric, lk)
        # reuse histogram_stats to get deltas
        # but we need raw bucket-deltas for re-quantile, not the
        # already-quantilized output -- so recompute here.
        count_samples = sorted(hist.get("count") or [], key=lambda x: x[0])
        sum_samples = sorted(hist.get("sum") or [], key=lambda x: x[0])
        if len(count_samples) >= 2:
            role_h_acc[rk]["count_delta"] += max(
                0.0, count_samples[-1][1] - count_samples[0][1])
        if len(sum_samples) >= 2:
            role_h_acc[rk]["sum_delta"] += max(
                0.0, sum_samples[-1][1] - sum_samples[0][1])
        for le, samples in (hist.get("buckets") or {}).items():
            samples = sorted(samples, key=lambda x: x[0])
            if len(samples) >= 2:
                role_h_acc[rk]["bucket_deltas"][le] += max(
                    0.0, samples[-1][1] - samples[0][1])
        role_h_acc[rk]["n_workers"] += 1

    role_histograms_stats: dict[SeriesKey, dict] = {}
    for rk, acc in role_h_acc.items():
        bucket_deltas = sorted(acc["bucket_deltas"].items(), key=lambda x: x[0])
        if acc["count_delta"] <= 0 or not bucket_deltas:
            role_histograms_stats[rk] = {
                "n": acc["n_workers"],
                "delta": acc["count_delta"],
                "sum": acc["sum_delta"] if acc["sum_delta"] > 0 else None,
                "mean": None, "median": None, "p90": None, "p99": None,
                "max": None,
            }
            continue
        mean = acc["sum_delta"] / acc["count_delta"] if acc["count_delta"] > 0 else None
        finite_with_inc = [(le, d) for le, d in bucket_deltas
                           if not math.isinf(le) and d > 0]
        max_val = finite_with_inc[-1][0] if finite_with_inc else None
        role_histograms_stats[rk] = {
            "n": acc["n_workers"],
            "delta": acc["count_delta"],
            "sum": acc["sum_delta"],
            "mean": mean,
            "median": histogram_quantile(0.50, bucket_deltas),
            "p90": histogram_quantile(0.90, bucket_deltas),
            "p99": histogram_quantile(0.99, bucket_deltas),
            "max": max_val,
        }

    return role_gauges, role_counters_stats, role_histograms_stats


# ---------- assemble rows ----------


def _row(key: SeriesKey, kind: str, st: dict, labels_key: str) -> dict:
    worker, role, metric, _ = key
    return {
        "worker": worker,
        "role": role,
        "metric": metric,
        "type": kind,
        "labels": labels_key,
        "n": st.get("n") or 0,
        "mean": st.get("mean"),
        "median": st.get("median"),
        "p90": st.get("p90"),
        "p99": st.get("p99"),
        "max": st.get("max"),
        "delta": st.get("delta"),
        "rate_per_s": st.get("rate_per_s"),
    }


def assemble_rows(gauges, counters, histograms, cls):
    """Reduce raw series → list of stat rows, including role aggregates.
    Sorts by (role, worker, metric, labels) for stable output."""
    rows: list[dict] = []
    for key, samples in gauges.items():
        rows.append(_row(key, "gauge", gauge_stats([v for _, v in samples]), key[3]))
    for key, samples in counters.items():
        rows.append(_row(key, "counter", counter_stats(samples), key[3]))
    for key, hist in histograms.items():
        rows.append(_row(key, "histogram", histogram_stats(hist), key[3]))

    role_gauges, role_counter_stats, role_hist_stats = aggregate_by_role(
        gauges, counters, histograms, cls)
    for key, samples in role_gauges.items():
        rows.append(_row(key, "gauge", gauge_stats([v for _, v in samples]), key[3]))
    for key, st in role_counter_stats.items():
        rows.append(_row(key, "counter", st, key[3]))
    for key, st in role_hist_stats.items():
        rows.append(_row(key, "histogram", st, key[3]))

    rows.sort(key=lambda r: (r["role"], r["worker"], r["metric"], r["labels"]))
    return rows


# ---------- output ----------


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if abs(v) >= 1e6:
            return f"{v:.3e}"
        return f"{v:.4f}"
    return str(v)


def write_csv(rows: list[dict], session_id: str,
              window: tuple[float, float], path: Path) -> None:
    s_ts, e_ts = window
    dur = e_ts - s_ts
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session_id", "window_start_unix_s", "window_end_unix_s",
            "window_duration_s",
            "worker", "role", "metric", "type", "labels",
            "n", "mean", "median", "p90", "p99", "max",
            "delta", "rate_per_s",
        ])
        for r in rows:
            w.writerow([
                session_id, f"{s_ts:.3f}", f"{e_ts:.3f}", f"{dur:.3f}",
                r["worker"], r["role"], r["metric"], r["type"], r["labels"],
                r["n"],
                _fmt(r["mean"]), _fmt(r["median"]),
                _fmt(r["p90"]), _fmt(r["p99"]), _fmt(r["max"]),
                _fmt(r["delta"]), _fmt(r["rate_per_s"]),
            ])


def print_table(rows: list[dict], session_id: str,
                window: tuple[float, float]) -> None:
    print()
    s_ts, e_ts = window
    print(f"Session: {session_id}")
    print(f"Window:  ts={s_ts:.3f} → {e_ts:.3f}  ({e_ts - s_ts:.2f}s)")
    print(f"         {len(rows)} series")

    def _p(v):
        if v is None:
            return f"{'-':>9}"
        if isinstance(v, float):
            if abs(v) >= 1e6:
                return f"{v:>9.2e}"
            return f"{v:>9.3f}"
        return f"{v:>9}"

    hdr = (f"{'worker':<8} {'role':<7} {'metric':<48} {'type':<9} "
           f"{'n':>5} {'mean':>9} {'median':>9} {'p90':>9} {'p99':>9} "
           f"{'max':>9} {'delta':>9} {'rate/s':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        metric = r["metric"]
        if r["labels"]:
            metric = f"{metric}{{{r['labels']}}}"
        if len(metric) > 48:
            metric = metric[:45] + "..."
        print(
            f"{r['worker']:<8} {r['role']:<7} {metric:<48} {r['type']:<9} "
            f"{r['n']:>5} {_p(r['mean'])} {_p(r['median'])} {_p(r['p90'])} "
            f"{_p(r['p99'])} {_p(r['max'])} {_p(r['delta'])} {_p(r['rate_per_s'])}"
        )


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default=None, type=Path,
                    help="Profile NDJSON; bounds the window to one session's "
                         "query.start → query.end. Omit to aggregate ALL ticks.")
    ap.add_argument("--metrics", required=True, type=Path,
                    help="vLLM scrape NDJSON from scrape_vllm_metrics.py")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory (created if missing)")
    ap.add_argument("--session-id", default=None,
                    help="When the profile has multiple sessions, select which "
                         "one. Defaults to the first session in the file.")
    args = ap.parse_args(argv)

    if not args.metrics.exists():
        print(f"input not found: {args.metrics}", file=sys.stderr)
        return 2
    if args.profile is not None and not args.profile.exists():
        print(f"input not found: {args.profile}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    if args.profile is None:
        rows_in = load_rows(args.metrics, float("-inf"), float("inf"))
        if not rows_in:
            print("no scrape rows found", file=sys.stderr)
            return 1
        ts_values = [r["ts"] for r in rows_in]
        window = (min(ts_values), max(ts_values))
        session_id = "ALL_POINTS"
        print(f"all-points mode: {len(rows_in)} scrape rows across "
              f"{window[1] - window[0]:.2f}s")
    else:
        windows = load_session_windows(args.profile)
        if not windows:
            print("no session windows found in profile NDJSON", file=sys.stderr)
            return 1
        if args.session_id:
            if args.session_id not in windows:
                print(f"session_id {args.session_id!r} not found "
                      f"(have {list(windows)})", file=sys.stderr)
                return 1
            session_id = args.session_id
        else:
            session_id = next(iter(windows))
            if len(windows) > 1:
                print(f"profile has {len(windows)} sessions; picking "
                      f"{session_id!r}. Pass --session-id to choose another.",
                      file=sys.stderr)
        window = windows[session_id]
        rows_in = load_rows(args.metrics, *window)
        print(f"window={window[1] - window[0]:.2f}s  matched "
              f"{len(rows_in)} scrape rows")
        if not rows_in:
            print("no scrape rows inside the window; was the scraper running?",
                  file=sys.stderr)
            return 1

    # Build the set of all metric names so the classifier can detect
    # histograms (which need _bucket / _count / _sum coexistence).
    all_names: set[str] = set()
    for r in rows_in:
        all_names.update((r.get("metrics") or {}).keys())
    cls = classify_metrics(all_names)

    gauges, counters, histograms = collect_series(rows_in, cls)
    rows = assemble_rows(gauges, counters, histograms, cls)

    csv_path = args.output / "vllm_metrics_stats.csv"
    write_csv(rows, session_id, window, csv_path)
    print(f"  wrote {csv_path}")
    print_table(rows, session_id, window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
