#!/usr/bin/env python3
"""Resource utilization stats for a single agent session.

Joins profile NDJSON (per-session opencode events) with resource NDJSON
(from `monitor_resources.py`) on wall-clock ts to compute mean / median /
p90 / p99 / max for every metric DURING that session's
`query.start` → `query.end` window.

Optionally reads `deploy/testbed.yaml` to label each GPU as
prefill/decode/other based on `vllm.prefill_workers[].gpus` and
`vllm.decode_workers[].gpus`, and to emit role-aggregated rows
(prefill-mean SM_ACTIVE across both GPUs, etc).

Usage:
  scripts/analyze_session_resources.py \\
      --profile /tmp/testbed-workspaces/profiles/ses_xxx.jsonl \\
      --resource logs/resource.ndjson \\
      --output results/run1/session_xxx_resources \\
      [--session-id ses_xxx] \\
      [--testbed-yaml deploy/testbed.yaml]

Outputs:
  session_resources_stats.csv   per-metric stats (mean/median/p90/p99/max)
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
    [start_ts, end_ts] inclusive window."""
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


# ---------- testbed.yaml -> GPU role mapping ----------


def parse_gpu_role_map(testbed_yaml: Path | None) -> dict[int, tuple[str, str]]:
    """{gpu_index: (worker_name, role)} where role ∈ {"prefill", "decode"}.
    Returns {} if path is None or yaml is unreadable."""
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
    return out


# ---------- metric extraction ----------


def extract_metrics(samples: list[dict],
                    gpu_role: dict[int, tuple[str, str]] | None = None,
                   ) -> dict[str, list[float]]:
    """Walk every sample row and bucket numeric values by metric name.
    Returns {metric_name: [values...]}.

    Metric naming:
      host.cpu_util_pct, host.mem_used_gib, host.mem_available_gib
      gpu<N>.<DCGM_FIELD>                       (always)
      gpu<N>[<worker>/<role>].<DCGM_FIELD>      (when role map present)
      <role>.<DCGM_FIELD>                       (aggregate; when role map present)
      process.<name>.cpu_util_pct
      process.<name>.rss_gib
    """
    gpu_role = gpu_role or {}
    metrics: dict[str, list[float]] = defaultdict(list)
    # For role aggregation, accumulate per-sample per-role per-field
    # AVERAGES across GPUs sharing that role, then push one value per
    # sample into metrics[<role>.<field>]. This keeps p90/p99 meaningful
    # at the role level (vs flattening all GPU samples together which
    # would double-count when N_prefill_gpus != N_decode_gpus).
    for s in samples:
        # host
        h = s.get("host") or {}
        if isinstance(h.get("cpu_util_pct"), (int, float)):
            metrics["host.cpu_util_pct"].append(float(h["cpu_util_pct"]))
        if isinstance(h.get("mem_used_bytes"), (int, float)):
            metrics["host.mem_used_gib"].append(h["mem_used_bytes"] / (1 << 30))
        if isinstance(h.get("mem_available_bytes"), (int, float)):
            metrics["host.mem_available_gib"].append(h["mem_available_bytes"] / (1 << 30))

        # gpus
        role_field_vals: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for gpu in s.get("gpus") or []:
            idx = gpu.get("index")
            if not isinstance(idx, int):
                continue
            worker_role = gpu_role.get(idx)
            for k, v in gpu.items():
                if k == "index":
                    continue
                if not isinstance(v, (int, float)):
                    continue
                # per-GPU (always)
                base = f"gpu{idx}"
                if worker_role:
                    base = f"gpu{idx}[{worker_role[0]}/{worker_role[1]}]"
                metrics[f"{base}.{k}"].append(float(v))
                if worker_role:
                    role_field_vals[worker_role[1]][k].append(float(v))
        # role aggregate: per-sample mean over GPUs sharing a role
        for role, field_vals in role_field_vals.items():
            for field, vals in field_vals.items():
                if vals:
                    metrics[f"{role}.{field}"].append(float(np.mean(vals)))

        # processes
        for p in s.get("processes") or []:
            name = p.get("name") or "?"
            if isinstance(p.get("cpu_util_pct"), (int, float)):
                metrics[f"process.{name}.cpu_util_pct"].append(float(p["cpu_util_pct"]))
            if isinstance(p.get("rss_bytes"), (int, float)):
                metrics[f"process.{name}.rss_gib"].append(p["rss_bytes"] / (1 << 30))

    return metrics


# ---------- stats ----------


def stats(values: list[float]) -> dict[str, float | int | None]:
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
            "n", "mean", "median", "p90", "p99", "max",
        ])
        s_ts, e_ts = window
        dur = e_ts - s_ts
        for name in sorted(stats_per_metric.keys()):
            s = stats_per_metric[name]
            w.writerow([
                session_id, f"{s_ts:.3f}", f"{e_ts:.3f}", f"{dur:.3f}",
                name, s["n"],
                _fmt(s["mean"]), _fmt(s["median"]),
                _fmt(s["p90"]), _fmt(s["p99"]), _fmt(s["max"]),
            ])


def print_table(stats_per_metric: dict[str, dict],
                session_id: str, window: tuple[float, float]) -> None:
    print()
    s_ts, e_ts = window
    print(f"Session: {session_id}")
    print(f"Window:  ts={s_ts:.3f} → {e_ts:.3f}  ({e_ts - s_ts:.2f}s)")
    print(f"         {len(stats_per_metric)} metrics")
    hdr = f"{'metric':<55} {'n':>5} {'mean':>10} {'median':>10} {'p90':>10} {'p99':>10} {'max':>10}"
    print(hdr)
    print("-" * len(hdr))

    def _p(v):
        return f"{v:>10.3f}" if isinstance(v, (int, float)) else f"{'-':>10}"

    for name in sorted(stats_per_metric.keys()):
        s = stats_per_metric[name]
        print(f"{name:<55} {s['n']:>5} {_p(s['mean'])} {_p(s['median'])} "
              f"{_p(s['p90'])} {_p(s['p99'])} {_p(s['max'])}")


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profile", required=True, type=Path,
                    help="Profile NDJSON (single session jsonl)")
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

    for p in (args.profile, args.resource):
        if not p.exists():
            print(f"input not found: {p}", file=sys.stderr)
            return 2
    args.output.mkdir(parents=True, exist_ok=True)

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
            print(f"profile has {len(windows)} sessions; picking {session_id!r}. "
                  f"Pass --session-id to choose another.", file=sys.stderr)
    window = windows[session_id]

    samples = load_samples_in_window(args.resource, *window)
    print(f"window={window[1] - window[0]:.2f}s  matched {len(samples)} resource samples")
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
