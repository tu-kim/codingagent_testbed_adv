#!/usr/bin/env python3
"""Periodically scrape per-worker Prometheus /metrics from dynamo.vllm
system status servers and emit NDJSON.

Each vllm worker spawned by `testbed.sh up workers` exposes
`http://<worker.host>:<system_port_base + rank>/metrics` when
`vllm.system_port_base > 0` (DYN_SYSTEM_PORT env is set per worker
in `spawn_worker` — see dynamo/lib/runtime/src/system_status_server.rs).

Worker enumeration is read directly from `testbed.yaml`:
  rank 0..N-1   = vllm.prefill_workers[]
  rank N..M-1   = vllm.decode_workers[]
so the rank counter matches what `spawn_worker` assigned via
`up_workers`'s sequential loop.

Output NDJSON (one row per worker per sample tick):
  {
    "ts": 1779200000.123,
    "interval_s": 1.0,
    "worker": "p0",
    "role": "prefill",
    "host": "127.0.0.1",
    "port": 21000,
    "ok": true,
    "metrics": {
      "vllm:num_requests_running": [{"labels": {...}, "value": 4}],
      "vllm:kv_cache_usage_perc": [...],
      "vllm:time_to_first_token_seconds_bucket": [...],
      "vllm:time_to_first_token_seconds_count": [...],
      "vllm:time_to_first_token_seconds_sum": [...],
      ...
    }
  }

By default we keep only a curated allowlist of EXACT metric names
focused on KV cache memory + queue depth + token throughput
(see DEFAULT_METRIC_NAMES). Latency histograms (TTFT/ITL/E2E) and
Python/process internals are dropped -- per-request latency is already
captured in dynamo's in-band `nvext.timing` + opencode profile NDJSON.

Override the allowlist by either:
  --keep-all                       # capture every metric (legacy behavior)
  --metric-names "name1,name2,..." # comma-list of exact names
  monitor.vllm_metric_names: [...] # in testbed.yaml (preferred)

Usage:
  scripts/scrape_vllm_metrics.py \\
      --testbed-yaml deploy/testbed.yaml \\
      --output logs/vllm_metrics.ndjson \\
      --interval 1.0

Stop with SIGTERM (`testbed.sh down scrape_metrics`) or Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


# Curated allowlist of vLLM Prometheus metric names. Names verified
# against vllm v0.19.0 (the dynamo-pinned version) -- the v1 engine
# renamed several gauges (dropped `gpu_` prefix, removed `cpu_` variant)
# vs the older v0 names. See dynamo/docs/observability/metrics-comparison.md
# for the live-scrape reference table. Most latency histograms are
# omitted because per-request latency is already in dynamo's in-band
# `nvext.timing` + opencode profile NDJSON -- EXCEPT the scheduler
# queue-wait histogram, which is the one signal NOT recoverable from
# the client side (it's the time a request sat in the worker's
# scheduler queue before compute started = scheduling delay). Since we
# scrape each worker's port separately, a prefill worker's queue-time
# histogram IS the prefill scheduling-delay distribution and a decode
# worker's is the decode one. Override via --metric-names or
# monitor.vllm_metric_names in testbed.yaml.
DEFAULT_METRIC_NAMES = frozenset({
    # KV cache memory (the headline signal)
    "vllm:kv_cache_usage_perc",           # 0.0-1.0 fraction of KV blocks in use
                                          # (v1 rename of v0 `gpu_cache_usage_perc`;
                                          # v1 dropped the cpu_cache_usage_perc variant)
    "vllm:num_preemptions_total",         # counter: evictions under cache pressure
    # Prefix cache effectiveness (v1 dropped the `gpu_` prefix)
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    # Cached-input-token volume -- aggregate counterpart to the per-
    # request `usage.prompt_tokens_details.cached_tokens` field that
    # the dynamo worker propagates to clients. Pairs with preempt count.
    "vllm:prompt_tokens_cached_total",       # hit tokens (counter)
    "vllm:prompt_tokens_recomputed_total",   # tokens lost to preemption + recomputed
    # Queue depth (instantaneous)
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    # Scheduling delay: time in the worker scheduler queue before compute.
    # Histogram -- the 3 suffixed series are all needed for
    # analyze_vllm_metrics' bucket-delta percentiles. Per-worker scrape
    # splits this into prefill vs decode scheduling-delay distributions.
    "vllm:request_queue_time_seconds_bucket",
    "vllm:request_queue_time_seconds_count",
    "vllm:request_queue_time_seconds_sum",
    # Token throughput counters (cumulative; analyze_vllm_metrics derives delta+rate)
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
})

# Prometheus text-format line (no labels): metric value [ts]
# With labels:                              metric{labels} value [ts]
_METRIC_LINE = re.compile(
    r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{(?P<labels>[^}]*)\})?'
    r'\s+(?P<value>-?[0-9.eE+]+|NaN|[+\-]?Inf)'
    r'(?:\s+\d+)?'
    r'\s*$'
)
_LABEL_PAIR = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str,
                     keep_names: frozenset[str] | set[str] | None = None,
                    ) -> dict[str, list[dict]]:
    """Parse Prometheus exposition-format text into
    {metric_name: [{labels: {...}, value: float}, ...]}.

    `keep_names` is an exact-match allowlist applied at parse time
    (only matching names enter the output). Pass None to keep
    everything (--keep-all)."""
    out: dict[str, list[dict]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name = m.group("name")
        if keep_names is not None and name not in keep_names:
            continue
        labels: dict[str, str] = {}
        if m.group("labels"):
            for k, v in _LABEL_PAIR.findall(m.group("labels")):
                # Unescape \" and \\ that Prometheus uses in label values
                labels[k] = v.replace(r"\\", "\\").replace(r"\"", '"')
        try:
            value = float(m.group("value"))
        except ValueError:
            continue
        out.setdefault(name, []).append({"labels": labels, "value": value})
    return out


def load_workers(testbed_yaml: Path) -> list[dict]:
    """Return ordered [{worker, role, host, port}, ...] matching the
    spawn order in testbed.sh's up_workers (prefill_workers then
    decode_workers, rank counting through both)."""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        raise SystemExit(
            "PyYAML required to read testbed.yaml; install with "
            "`pip install pyyaml` (or `sudo apt install python3-yaml` for root)"
        )
    cfg = yaml.safe_load(testbed_yaml.read_text()) or {}
    vllm = cfg.get("vllm") or {}
    base = vllm.get("system_port_base", -1)
    if not isinstance(base, int) or base <= 0:
        raise SystemExit(
            f"vllm.system_port_base = {base!r} in {testbed_yaml}; set a "
            "positive value to enable per-worker /metrics exposure."
        )
    workers: list[dict] = []
    rank = 0
    # Order MUST match testbed.sh up_workers spawn order (prefill -> decode
    # -> agg); rank feeds the same system_port_base + rank math.
    for role, key in (("prefill", "prefill_workers"),
                      ("decode", "decode_workers"),
                      ("agg", "agg_workers")):
        for w in vllm.get(key) or []:
            workers.append({
                "worker": w.get("name", f"{role}{rank}"),
                "role": role,
                "host": w.get("host", "127.0.0.1"),
                "port": base + rank,
            })
            rank += 1
    if not workers:
        raise SystemExit("no vllm workers configured in testbed.yaml")
    return workers


def scrape_one(host: str, port: int, timeout_s: float,
               keep_names: frozenset[str] | set[str] | None,
              ) -> tuple[bool, dict | str]:
    """GET http://<host>:<port>/metrics, parse, return (ok, payload).
    payload is the parsed metrics dict on success, an error string on
    failure. Network errors are returned (not raised) so one dead
    worker doesn't kill the whole loop."""
    url = f"http://{host}:{port}/metrics"
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
        return False, f"{type(e).__name__}: {e}"
    return True, parse_prometheus(text, keep_names)


def run_scraper(workers: list[dict], interval_s: float, output: Path,
                keep_names: frozenset[str] | set[str] | None, timeout_s: float,
                stop_fn) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", buffering=1) as f:
        while not stop_fn():
            tick_ts = time.time()
            for w in workers:
                ok, payload = scrape_one(w["host"], w["port"], timeout_s, keep_names)
                row = {
                    "ts": tick_ts,
                    "interval_s": interval_s,
                    "worker": w["worker"],
                    "role": w["role"],
                    "host": w["host"],
                    "port": w["port"],
                    "ok": ok,
                }
                if ok:
                    row["metrics"] = payload
                else:
                    row["error"] = payload
                f.write(json.dumps(row) + "\n")
                n += 1
            target = tick_ts + interval_s
            remaining = target - time.time()
            if remaining > 0:
                time.sleep(remaining)
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--testbed-yaml", required=True, type=Path,
                    help="Path to deploy/testbed.yaml (read for worker list "
                         "+ system_port_base)")
    ap.add_argument("--output", required=True, type=Path,
                    help="NDJSON output file (one row per worker per tick)")
    ap.add_argument("--interval", type=float, default=1.0, metavar="SECONDS",
                    help="Polling period; default 1.0s")
    ap.add_argument("--timeout", type=float, default=2.0, metavar="SECONDS",
                    help="HTTP timeout per scrape; default 2.0s")
    ap.add_argument("--keep-all", action="store_true",
                    help="Capture every Prometheus metric; bypasses the "
                         "default + --metric-names allowlist entirely")
    ap.add_argument("--metric-names", default="", metavar="N1,N2,...",
                    help="Comma-separated allowlist of exact metric names. "
                         "Empty = use DEFAULT_METRIC_NAMES (KV cache + queue "
                         "+ token throughput). Ignored if --keep-all is set.")
    args = ap.parse_args(argv)

    if not args.testbed_yaml.exists():
        print(f"testbed.yaml not found: {args.testbed_yaml}", file=sys.stderr)
        return 2

    workers = load_workers(args.testbed_yaml)
    print(f"scraping {len(workers)} workers at {args.interval}s interval:")
    for w in workers:
        print(f"  {w['worker']:<8} {w['role']:<8} {w['host']}:{w['port']}")

    if args.keep_all:
        keep: frozenset[str] | None = None
        print("metric filter: --keep-all (no filter)")
    elif args.metric_names:
        keep = frozenset(n for n in args.metric_names.split(",") if n)
        print(f"metric filter: {len(keep)} names from --metric-names")
    else:
        keep = DEFAULT_METRIC_NAMES
        print(f"metric filter: DEFAULT_METRIC_NAMES ({len(keep)} names)")

    stop = {"flag": False}
    def _sig(signum, frame): stop["flag"] = True
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    n = run_scraper(workers, args.interval, args.output, keep, args.timeout,
                    lambda: stop["flag"])
    print(f"scrape_vllm_metrics: wrote {n} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
