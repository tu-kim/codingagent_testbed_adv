#!/usr/bin/env python3
"""Resource utilization stats for a single agent session, OR globally
across all measured samples.

Two modes (selected by whether --profile is passed):

  Session-window mode (--profile <session.jsonl>):
    Joins profile NDJSON with resource NDJSON on wall-clock ts to
    compute summary statistics for every metric DURING that session's
    `query.start` → `query.end` window.

  All-points mode (omit --profile):
    Aggregate stats across EVERY sample in the resource NDJSON,
    regardless of session boundaries. Useful for "overall workload
    average" or steady-state monitoring snapshots that span many
    sessions / idle gaps.

Window-aggregate awareness: monitor_resources >= 2026-05-28 writes
each drain window as {mean,min,max,n} rather than a point sample.
This analyzer unpacks all four series so:
  - `mean`   = n-weighted average of window-means (true period mean)
  - `median/p90/p99` = percentiles of window-means (window-level
    distribution; per-window variation is smoothed out)
  - `min`    = MIN of all window-mins (true within-period valley)
  - `max`    = MAX of all window-maxs (true within-period peak,
    NOT the misleading max-of-window-means the old flat code emitted)
  - `n_windows` = drain windows contributing
  - `n_samples` = total DCGM internal samples (= sum of per-window n)
Plain-scalar values (host/process metrics, legacy data) collapse to
min=max=mean=value with n=1 so the same code path works on both shapes.

Counter handling: cumulative DCGM byte counters (PROF_PCIE_*_BYTES /
PROF_NVLINK_*_BYTES) are NOT statistically meaningful as-is -- the
"mean" of a monotonically-rising value is whatever happens to be in
the middle of the window. extract_metrics converts each counter to a
per-window DELTA (current - previous, clipped at 0 on resets) and
renames the metric with a `_delta` suffix. So you'll see e.g.
`gpu0.DCGM_FI_PROF_PCIE_RX_BYTES_delta` with `mean` = average bytes
per drain window, `max` = peak window. The first sample for each
(gpu, counter) seeds the baseline and is dropped from the output
(no prior cumulative to subtract). Role aggregate for counters SUMS
across role GPUs (total throughput) rather than averaging.

Optionally reads `deploy/testbed.yaml` to label each GPU as
prefill/decode/agg/other based on `vllm.prefill_workers[].gpus`,
`vllm.decode_workers[].gpus`, and `vllm.agg_workers[].gpus` (PD
colocation), and to emit role-aggregated rows (prefill-mean SM_ACTIVE
across both GPUs, etc).

Usage:
  # Session window
  scripts/analyze_session_resources.py \\
      --profile /tmp/testbed-workspaces/profiles/ses_xxx.jsonl \\
      --resource logs/resource.ndjson \\
      --output results/run1/session_xxx_resources \\
      [--session-id ses_xxx] \\
      [--testbed-yaml deploy/testbed.yaml]

  # All measured points (no profile filter)
  scripts/analyze_session_resources.py \\
      --resource logs/resource.ndjson \\
      --output results/run1/global_resources

Outputs:
  session_resources_stats.csv   per-metric stats
                                  (n_windows, n_samples, mean, median,
                                   p90, p99, min, max)
  stdout                        pretty table
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# ---------- ingest ----------


def load_session_windows(profile_path: Path) -> dict[str, tuple[float, float]]:
    """{sessionID: (query_start_ts, query_end_ts)} for every session
    that has both events. Sessions missing query.end (mid-flight or
    crashed runs) fall back to (start, last_ts_seen) so we still get
    a usable window."""
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
    out = {}
    for sid, s in starts.items():
        out[sid] = (s, ends.get(sid, last_ts.get(sid, s)))
    return out


def load_samples_in_window(resource_path: Path, start_ts: float, end_ts: float) -> list[dict]:
    """Stream the resource NDJSON, keeping only samples inside the
    [start_ts, end_ts] inclusive window. Pass start_ts=-inf, end_ts=+inf
    to load all samples regardless of ts (used by --no-window /
    all-points mode)."""
    out = []
    with resource_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = s.get("ts")
            if ts is None:
                continue
            if start_ts <= ts <= end_ts:
                out.append(s)
    return out


def load_all_samples(resource_path: Path) -> list[dict]:
    """All samples in the resource NDJSON, no window filter. Skips
    malformed JSON lines and samples missing `ts`."""
    return load_samples_in_window(resource_path, float("-inf"), float("inf"))


# ---------- testbed.yaml -> GPU role mapping ----------


def parse_gpu_role_map(testbed_yaml: Path | None) -> dict[int, tuple[str, str]]:
    """{gpu_index: (worker_name, role)} where role ∈ {"prefill", "decode",
    "agg"}. Returns {} if path is None or yaml is unreadable."""
    if testbed_yaml is None or not testbed_yaml.exists():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return {}
    try:
        cfg = yaml.safe_load(testbed_yaml.read_text())
    except (yaml.YAMLError, OSError):
        return {}
    out: dict[int, tuple[str, str]] = {}

    def _add(workers, role):
        for w in workers or []:
            name = w.get("name", "?")
            gpus_str = str(w.get("gpus", ""))
            for g in gpus_str.split(","):
                g = g.strip()
                if not g:
                    continue
                try:
                    out[int(g)] = (name, role)
                except ValueError:
                    continue

    vllm = (cfg or {}).get("vllm", {}) or {}
    _add(vllm.get("prefill_workers"), "prefill")
    _add(vllm.get("decode_workers"), "decode")
    _add(vllm.get("agg_workers"), "agg")  # PD colocation: one role does both
    return out


# ---------- metric extraction ----------


# Cumulative DCGM byte counters -- monitor_resources keeps the LAST
# observed value rather than a window aggregate. The analyzer converts
# them to per-window deltas (current - previous, clipped at 0 on resets)
# so mean/min/max become "bytes per window" rather than meaningless
# cumulative-value stats. Output metric name gets a `_delta` suffix to
# make the semantics explicit. MUST stay in sync with COUNTER_FIELDS in
# scripts/monitor_resources.py.
DCGM_COUNTER_FIELDS = frozenset({
    "DCGM_FI_PROF_PCIE_RX_BYTES",
    "DCGM_FI_PROF_PCIE_TX_BYTES",
    "DCGM_FI_PROF_NVLINK_RX_BYTES",
    "DCGM_FI_PROF_NVLINK_TX_BYTES",
})


def _empty_record() -> dict[str, list[float]]:
    return {"mean": [], "min": [], "max": [], "n": []}


def _push(metrics: dict[str, dict[str, list[float]]],
          name: str, mean: float, mn: float, mx: float, n: float) -> None:
    rec = metrics[name]
    rec["mean"].append(float(mean))
    rec["min"].append(float(mn))
    rec["max"].append(float(mx))
    rec["n"].append(float(n))


def _coerce_value(v) -> tuple[float, float, float, float] | None:
    """Normalize a sample value to (mean, min, max, n). Dict-shape gauges
    from monitor_resources >= 2026-05-28 unpack all four; plain scalars
    (counters, host/process fields, legacy gauges) collapse to
    mean=min=max=val, n=1. Returns None for non-numeric / malformed."""
    if isinstance(v, dict):
        m = v.get("mean")
        if not isinstance(m, (int, float)) or isinstance(m, bool):
            return None
        mn = v.get("min", m)
        mx = v.get("max", m)
        n = v.get("n", 1)
        if not isinstance(mn, (int, float)) or isinstance(mn, bool):
            mn = m
        if not isinstance(mx, (int, float)) or isinstance(mx, bool):
            mx = m
        if not isinstance(n, (int, float)) or isinstance(n, bool):
            n = 1
        return float(m), float(mn), float(mx), float(n)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        f = float(v)
        return f, f, f, 1.0
    return None


def extract_metrics(samples: list[dict],
                    gpu_role: dict[int, tuple[str, str]] | None = None,
                   ) -> dict[str, dict[str, list[float]]]:
    """Walk every sample row and bucket numeric values by metric name.
    Returns {metric_name: {"mean": [...], "min": [...], "max": [...], "n": [...]}}
    where each list has one entry per drain window.

    For dict-shape gauges (monitor_resources >= 2026-05-28), the four
    series carry the window's {mean, min, max, n}. For scalar values
    (counters, host/process fields, legacy gauges), mean/min/max all
    collapse to the same scalar and n=1. Downstream `stats()` reads the
    `min` / `max` series to report true within-run peaks (vs the
    misleading "max-of-window-means" the old flat structure produced).

    Metric naming:
      host.cpu_util_pct, host.mem_used_gib, host.mem_available_gib
      gpu<N>.<DCGM_FIELD>                       (always)
      gpu<N>[<worker>/<role>].<DCGM_FIELD>      (when role map present)
      <role>.<DCGM_FIELD>                       (aggregate; when role map present)
      process.<name>.cpu_util_pct
      process.<name>.rss_gib
    """
    gpu_role = gpu_role or {}
    metrics: dict[str, dict[str, list[float]]] = defaultdict(_empty_record)
    # Last cumulative value per (gpu_idx, counter_field). Persists across
    # the full sample loop so we can derive per-window deltas. The first
    # observation for any given (gpu, field) seeds the baseline and is
    # NOT pushed -- without a prior cumulative we'd be reporting "bytes
    # since process start" as if it were a window value.
    prev_counter: dict[tuple[int, str], float] = {}
    for s in samples:
        # host
        h = s.get("host") or {}
        if isinstance(h.get("cpu_util_pct"), (int, float)) and not isinstance(h["cpu_util_pct"], bool):
            v = float(h["cpu_util_pct"]); _push(metrics, "host.cpu_util_pct", v, v, v, 1.0)
        if isinstance(h.get("mem_used_bytes"), (int, float)) and not isinstance(h["mem_used_bytes"], bool):
            v = h["mem_used_bytes"] / (1 << 30); _push(metrics, "host.mem_used_gib", v, v, v, 1.0)
        if isinstance(h.get("mem_available_bytes"), (int, float)) and not isinstance(h["mem_available_bytes"], bool):
            v = h["mem_available_bytes"] / (1 << 30); _push(metrics, "host.mem_available_gib", v, v, v, 1.0)

        # gpus -- collect per-(role,field) tuples to derive a per-sample
        # role aggregate after the per-GPU push.
        role_field_vals: dict[str, dict[str, list[tuple[float, float, float, float]]]] = \
            defaultdict(lambda: defaultdict(list))
        for gpu in s.get("gpus") or []:
            idx = gpu.get("index")
            if not isinstance(idx, int):
                continue
            worker_role = gpu_role.get(idx)
            for k, v in gpu.items():
                if k == "index":
                    continue
                coerced = _coerce_value(v)
                if coerced is None:
                    continue
                mean, mn, mx, n = coerced
                base = f"gpu{idx}"
                if worker_role:
                    base = f"gpu{idx}[{worker_role[0]}/{worker_role[1]}]"

                if k in DCGM_COUNTER_FIELDS:
                    # Cumulative byte counter -> per-window delta.
                    # `mean` here is the LAST cumulative value in the
                    # window (monitor_resources doesn't aggregate counters,
                    # so mean==min==max already).
                    key = (idx, k)
                    prev = prev_counter.get(key)
                    prev_counter[key] = mean
                    if prev is None:
                        continue
                    delta = mean - prev
                    if delta < 0:
                        delta = 0.0  # counter reset / GPU reset
                    out_field = f"{k}_delta"
                    _push(metrics, f"{base}.{out_field}", delta, delta, delta, 1.0)
                    if worker_role:
                        role_field_vals[worker_role[1]][out_field].append(
                            (delta, delta, delta, 1.0)
                        )
                    continue

                # Gauge: per-window {mean,min,max,n} stats.
                _push(metrics, f"{base}.{k}", mean, mn, mx, n)
                if worker_role:
                    role_field_vals[worker_role[1]][k].append((mean, mn, mx, n))
        # Role aggregate:
        #   Gauges: arithmetic mean of per-GPU means (so p90/p99 at the
        #     role level reflects cross-sample variation), min of per-GPU
        #     mins, max of per-GPU maxs, sum of per-GPU ns.
        #   Counter deltas (`*_delta`): SUM of per-GPU deltas (total bytes
        #     across all role GPUs in this window) -- summing matches the
        #     "total throughput" intuition, whereas averaging would
        #     under-report by 1/N_gpus.
        for role, field_vals in role_field_vals.items():
            for field, tuples in field_vals.items():
                if not tuples:
                    continue
                ms = [t[0] for t in tuples]
                mns = [t[1] for t in tuples]
                mxs = [t[2] for t in tuples]
                ns = [t[3] for t in tuples]
                if field.endswith("_delta"):
                    total = float(sum(ms))
                    _push(metrics, f"{role}.{field}",
                          total, total, total, float(sum(ns)))
                else:
                    _push(metrics, f"{role}.{field}",
                          float(np.mean(ms)), float(min(mns)),
                          float(max(mxs)), float(sum(ns)))

        # processes
        for p in s.get("processes") or []:
            name = p.get("name") or "?"
            if isinstance(p.get("cpu_util_pct"), (int, float)) and not isinstance(p["cpu_util_pct"], bool):
                v = float(p["cpu_util_pct"]); _push(metrics, f"process.{name}.cpu_util_pct", v, v, v, 1.0)
            if isinstance(p.get("rss_bytes"), (int, float)) and not isinstance(p["rss_bytes"], bool):
                v = p["rss_bytes"] / (1 << 30); _push(metrics, f"process.{name}.rss_gib", v, v, v, 1.0)
            # n_procs = process-tree size summed into cpu/rss (monitor >=
            # 2026-06; older rows omit it). >1 confirms children were counted.
            if isinstance(p.get("n_procs"), (int, float)) and not isinstance(p["n_procs"], bool):
                v = float(p["n_procs"]); _push(metrics, f"process.{name}.n_procs", v, v, v, 1.0)

    return metrics


# ---------- stats ----------


def stats(record: dict[str, list[float]] | list[float]) -> dict[str, float | int | None]:
    """Reduce a metric's window-record to a row of summary statistics.

    Accepts either the rich {mean,min,max,n} dict produced by the
    refactored extract_metrics, OR a flat list of scalars (legacy
    callers / tests). For the flat-list path, min==max==mean==value
    and n=1 are synthesized so the output keys are stable.

    Returned keys:
      n_windows    drain windows contributing to this metric
      n_samples    sum of underlying DCGM samples (= sum of record["n"])
      mean         n-weighted mean of window-means (true period mean)
      median/p90/p99   percentiles of window-means
      min          min across window-mins (TRUE valley, not min-of-means)
      max          max across window-maxs (TRUE peak, not max-of-means)
    """
    if isinstance(record, list):
        record = {"mean": list(record), "min": list(record),
                  "max": list(record), "n": [1.0] * len(record)}
    means = np.asarray(record.get("mean", []), dtype=float)
    if means.size == 0:
        return {"n_windows": 0, "n_samples": 0, "mean": None, "median": None,
                "p90": None, "p99": None, "min": None, "max": None}
    mins = np.asarray(record.get("min", []), dtype=float)
    maxs = np.asarray(record.get("max", []), dtype=float)
    ns = np.asarray(record.get("n", []), dtype=float)
    if ns.size != means.size or ns.sum() <= 0:
        weights = None
    else:
        weights = ns
    return {
        "n_windows": int(means.size),
        "n_samples": int(ns.sum()) if ns.size else int(means.size),
        "mean": float(np.average(means, weights=weights)),
        "median": float(np.median(means)),
        "p90": float(np.percentile(means, 90)),
        "p99": float(np.percentile(means, 99)),
        "min": float(np.min(mins)) if mins.size else float(np.min(means)),
        "max": float(np.max(maxs)) if maxs.size else float(np.max(means)),
    }


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ---------- output ----------


def write_csv(stats_per_metric: dict[str, dict],
              session_id: str, window: tuple[float, float],
              path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "session_id", "window_start_unix_s", "window_end_unix_s",
            "window_duration_s", "metric",
            "n_windows", "n_samples",
            "mean", "median", "p90", "p99", "min", "max",
        ])
        s_ts, e_ts = window
        dur = e_ts - s_ts
        for name in sorted(stats_per_metric.keys()):
            s = stats_per_metric[name]
            w.writerow([
                session_id, f"{s_ts:.3f}", f"{e_ts:.3f}", f"{dur:.3f}",
                name, s["n_windows"], s["n_samples"],
                _fmt(s["mean"]), _fmt(s["median"]),
                _fmt(s["p90"]), _fmt(s["p99"]),
                _fmt(s["min"]), _fmt(s["max"]),
            ])


def print_table(stats_per_metric: dict[str, dict],
                session_id: str, window: tuple[float, float]) -> None:
    print()
    s_ts, e_ts = window
    print(f"Session: {session_id}")
    print(f"Window:  ts={s_ts:.3f} → {e_ts:.3f}  ({e_ts - s_ts:.2f}s)")
    print(f"         {len(stats_per_metric)} metrics")
    hdr = (f"{'metric':<55} {'win':>5} {'samp':>6} "
           f"{'mean':>10} {'median':>10} {'p90':>10} {'p99':>10} "
           f"{'min':>10} {'max':>10}")
    print(hdr)
    print("-" * len(hdr))

    def _p(v):
        return f"{v:>10.3f}" if isinstance(v, (int, float)) else f"{'-':>10}"

    for name in sorted(stats_per_metric.keys()):
        s = stats_per_metric[name]
        print(f"{name:<55} {s['n_windows']:>5} {s['n_samples']:>6} "
              f"{_p(s['mean'])} {_p(s['median'])} {_p(s['p90'])} {_p(s['p99'])} "
              f"{_p(s['min'])} {_p(s['max'])}")


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", default=None, type=Path,
                    help="Profile NDJSON to bound the window to one session's "
                         "query.start → query.end. Omit to aggregate ALL samples "
                         "in --resource regardless of session boundaries.")
    ap.add_argument("--resource", required=True, type=Path,
                    help="Resource NDJSON from monitor_resources.py")
    ap.add_argument("--output", required=True, type=Path,
                    help="Output directory (created if missing)")
    ap.add_argument("--session-id", default=None,
                    help="When the profile contains multiple sessions, "
                         "select which one. Defaults to the first/only session.")
    ap.add_argument("--testbed-yaml", default=None, type=Path,
                    help="Optional deploy/testbed.yaml to label GPUs with "
                         "prefill/decode role + worker name and emit "
                         "role-aggregated rows")
    args = ap.parse_args(argv)

    if not args.resource.exists():
        print(f"input not found: {args.resource}", file=sys.stderr)
        return 2
    if args.profile is not None and not args.profile.exists():
        print(f"input not found: {args.profile}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    if args.profile is None:
        # All-points mode: ignore session boundaries entirely.
        samples = load_all_samples(args.resource)
        if not samples:
            print("no resource samples found", file=sys.stderr)
            return 1
        ts_values = [s["ts"] for s in samples if isinstance(s.get("ts"), (int, float))]
        window = (min(ts_values), max(ts_values)) if ts_values else (0.0, 0.0)
        session_id = "ALL_POINTS"
        print(f"all-points mode: {len(samples)} samples across "
              f"{window[1] - window[0]:.2f}s")
    else:
        windows = load_session_windows(args.profile)
        if not windows:
            print("no session windows found in profile NDJSON", file=sys.stderr)
            return 1
        if args.session_id:
            if args.session_id not in windows:
                print(f"session_id {args.session_id!r} not found in profile "
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
        samples = load_samples_in_window(args.resource, *window)
        print(f"window={window[1] - window[0]:.2f}s  matched "
              f"{len(samples)} resource samples")
        if not samples:
            print("no resource samples inside the window; check that monitor was "
                  "running during this session", file=sys.stderr)
            return 1

    gpu_role = parse_gpu_role_map(args.testbed_yaml)
    if gpu_role:
        print(f"gpu role map: {gpu_role}")

    metrics = extract_metrics(samples, gpu_role=gpu_role)
    stats_per_metric = {name: stats(vals) for name, vals in metrics.items()}

    csv_path = args.output / "session_resources_stats.csv"
    write_csv(stats_per_metric, session_id, window, csv_path)
    print(f"  wrote {csv_path}")

    print_table(stats_per_metric, session_id, window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
