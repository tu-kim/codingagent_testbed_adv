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
      "vllm:gpu_cache_usage_perc": [...],
      "vllm:time_to_first_token_seconds_bucket": [...],
      "vllm:time_to_first_token_seconds_count": [...],
      "vllm:time_to_first_token_seconds_sum": [...],
      ...
    }
  }

By default we keep only metrics whose name starts with one of
{vllm:, dynamo_, nv_} -- skips Python/process-level prometheus_client
internals (process_cpu_seconds_total etc.) that aren't workload signal.

Usage:
  scripts/scrape_vllm_metrics.py \\
      --testbed-yaml deploy/testbed.yaml \\
      --output logs/vllm_metrics.ndjson \\
      --interval 1.0 \\
      [--keep-all]              # keep every metric, not just vllm:/dynamo_

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


DEFAULT_PREFIXES = ("vllm:", "dynamo_", "nv_")

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


def parse_prometheus(text: str, keep_prefixes: tuple[str, ...] | None = None) -> dict[str, list[dict]]:
    """Parse Prometheus exposition-format text into
    {metric_name: [{labels: {...}, value: float}, ...]}.

    `keep_prefixes` filters by name prefix at parse time. Pass None to
    keep everything (used by --keep-all)."""
    out: dict[str, list[dict]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _METRIC_LINE.match(line)
        if not m:
            continue
        name = m.group("name")
        if keep_prefixes is not None and not any(name.startswith(p) for p in keep_prefixes):
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
    for role, key in (("prefill", "prefill_workers"), ("decode", "decode_workers")):
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
               keep_prefixes: tuple[str, ...] | None,
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
    return True, parse_prometheus(text, keep_prefixes)


def run_scraper(workers: list[dict], interval_s: float, output: Path,
                keep_prefixes: tuple[str, ...] | None, timeout_s: float,
                stop_fn) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with output.open("w", buffering=1) as f:
        while not stop_fn():
            tick_ts = time.time()
            for w in workers:
                ok, payload = scrape_one(w["host"], w["port"], timeout_s, keep_prefixes)
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
                    help="Capture every Prometheus metric; default is to "
                         "keep only names starting with vllm:/dynamo_/nv_")
    args = ap.parse_args(argv)

    if not args.testbed_yaml.exists():
        print(f"testbed.yaml not found: {args.testbed_yaml}", file=sys.stderr)
        return 2

    workers = load_workers(args.testbed_yaml)
    print(f"scraping {len(workers)} workers at {args.interval}s interval:")
    for w in workers:
        print(f"  {w['worker']:<8} {w['role']:<8} {w['host']}:{w['port']}")

    keep = None if args.keep_all else DEFAULT_PREFIXES

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
