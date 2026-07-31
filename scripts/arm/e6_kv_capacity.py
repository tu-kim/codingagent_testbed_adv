#!/usr/bin/env python3
"""E6: multi-run KV-capacity / workload summary.

For each run directory (expected layout: <run>/logs/{frontend.log,
vllm_metrics.ndjson} + <run>/profiles[/.jsonl]; a bare logs dir also
works for the log-based parts) this produces:

1. summary.csv — one row per run:
     avg_batch_size          non-zero mean of vllm:num_requests_running
                             (all roles pooled, clipped to the frontend
                             run window, same rules as e4)
     total_input_tokens      sum of frontend.log input_tokens
     kv_hbm_frac_mean        mean vllm:kv_cache_usage_perc (0-1, worker mean)
     kv_hbm_gib_mean         above x --hbm-kv-gib (blank without the flag)
     kv_host_gib_mean        mean host-DRAM KV usage (LMCache metric,
                             auto-detected lmcache:* usage/size metric or
                             --host-kv-metric; assumed bytes)
     ttft_ms mean/p50/p90    frontend.log ttft_ms
     tpot_ms mean/p50/p90    (elapsed-ttft)/(output_tokens-1), out>1 only
     turns_per_session_mean  from profiles (main sessions only when
                             trace.jsonl exists)
2. session_tokens.csv — per run: session START tokens (first turn
   effective input = input + cache.read) and END tokens (last turn
   effective input + output): mean/p50/p90.
3. fig_kv_over_time.pdf — total KV size (HBM + host DRAM) vs time (left,
   one line per run, t=0 at each run's first sample) and its CDF
   (right). Without --hbm-kv-gib the HBM tier is plotted as usage
   fraction only when no host tier exists; with a host tier but no
   capacity the HBM tier is omitted from the total (warned).

Usage:
  scripts/arm/e6_kv_capacity.py run_a run_b [label=path ...] \
      [--hbm-kv-gib 24.0] [--host-kv-metric lmcache:local_cpu_usage] \
      [--out e6_kv]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

_ARM = Path(__file__).resolve().parent
_AP_PATH = _ARM.parent / "analyze_profiles.py"
_E4_PATH = _ARM / "e4_prefill_decode.py"
_E0_PATH = _ARM / "e0_turn_characterization.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- run resolution

def resolve_run(entry: str) -> dict:
    """{label, root, frontend?, metrics?, profiles?, trace?} for one CLI
    entry (label=path or path)."""
    label, _, rest = entry.partition("=")
    root = Path(rest) if rest else Path(entry)
    if not rest:
        label = root.name
    if not root.exists():
        raise FileNotFoundError(f"run not found: {entry}")
    logs = root / "logs" if (root / "logs").is_dir() else root
    out: dict = {"label": label, "root": root}
    fe = logs / "frontend.log"
    if fe.is_file():
        out["frontend"] = fe
    vm = logs / "vllm_metrics.ndjson"
    if vm.is_file():
        out["metrics"] = vm
    for cand in (root / "profiles", root / "profiles.jsonl"):
        if cand.exists():
            out["profiles"] = cand
            break
    tr = root / "trace.jsonl"
    if tr.is_file():
        out["trace"] = tr
    return out


# ---------------------------------------------------------------- frontend parse

_INPUT_RE = re.compile(r'(?:\b|")input_tokens\b"?\s*[=:]\s*"?(?P<v>\d+)"?')


def parse_frontend(e4, path: Path) -> list[dict]:
    """Per-request {ttft_ms, elapsed_ms, output_tokens, input_tokens?,
    completed_unix_s?}. Reuses e4's compiled field regexes/ANSI strip;
    adds input_tokens (which e4 doesn't parse). Lines missing
    elapsed/ttft/output are dropped; last write wins per request_id."""
    by_rid: dict[str, dict] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if "request completed" not in line:
                continue
            line = e4._ANSI_RE.sub("", line)
            rid = e4._REQID_RE.search(line)
            el = e4._ELAPSED_RE.search(line)
            tt = e4._TTFT_RE.search(line)
            ot = e4._OUT_RE.search(line)
            if not (rid and el and tt and ot):
                continue
            inp = _INPUT_RE.search(line)
            tsm = e4._ISO_RE.match(line)
            by_rid[rid.group("v")] = {
                "elapsed_ms": float(el.group("v")),
                "ttft_ms": float(tt.group("v")),
                "output_tokens": int(ot.group("v")),
                "input_tokens": int(inp.group("v")) if inp else None,
                "completed_unix_s": (e4._iso_to_unix(tsm.group("ts"))
                                     if tsm else None),
            }
    return list(by_rid.values())


# ---------------------------------------------------------------- stats helpers

def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    idx = q / 100.0 * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else math.nan


def _stats3(vals: list[float]) -> dict[str, float]:
    return {"mean": _mean(vals), "p50": _pct(vals, 50), "p90": _pct(vals, 90)}


# ---------------------------------------------------------------- KV time series

_HOST_METRIC_RE = re.compile(r"usage|size|bytes", re.I)


def detect_host_kv_metric(metrics_path: Path) -> str | None:
    """First lmcache:* metric name that looks like a CPU/local usage/size
    gauge (scrape keeps all lmcache:-prefixed names; exact names live in
    the external lmcache package, so detect rather than hardcode)."""
    cands: list[str] = []
    with metrics_path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for name in (row.get("metrics") or {}):
                if name.startswith("lmcache:") and _HOST_METRIC_RE.search(name):
                    if name not in cands:
                        cands.append(name)
            if cands:
                break
    # prefer explicitly cpu/local-tier names over generic matches
    for name in cands:
        if "cpu" in name or "local" in name:
            return name
    return cands[0] if cands else None


def kv_series(metrics_path: Path, window,
              hbm_kv_gib: float | None,
              host_metric: str | None) -> tuple[list[tuple[float, float]], str]:
    """[(ts, kv_gib_total)] per scrape tick + a unit tag ("GiB" or
    "fraction"). Per tick: HBM = mean worker kv_cache_usage_perc x
    hbm_kv_gib; host = sum of host_metric across workers (bytes -> GiB).
    Ticks are bucketed to 1 s so multi-worker rows merge."""
    # ts_bucket -> {"hbm": [fracs], "host": [bytes]}
    buckets: dict[int, dict[str, list[float]]] = {}
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
            ts = row.get("ts")
            if ts is None:
                continue
            if window is not None and (ts < window[0] or ts > window[1]):
                continue
            mets = row.get("metrics") or {}
            b = buckets.setdefault(int(ts), {"hbm": [], "host": []})
            for e in mets.get("vllm:kv_cache_usage_perc") or []:
                v = e.get("value")
                if isinstance(v, (int, float)):
                    b["hbm"].append(float(v))
            if host_metric:
                for e in mets.get(host_metric) or []:
                    v = e.get("value")
                    if isinstance(v, (int, float)):
                        b["host"].append(float(v))
    series: list[tuple[float, float]] = []
    have_host = any(b["host"] for b in buckets.values())
    unit = "GiB"
    if hbm_kv_gib is None and not have_host:
        unit = "fraction"           # nothing to convert; plot raw HBM fraction
    for ts in sorted(buckets):
        b = buckets[ts]
        total = 0.0
        got = False
        if b["hbm"]:
            frac = _mean(b["hbm"])
            if unit == "fraction":
                total += frac
                got = True
            elif hbm_kv_gib is not None:
                total += frac * hbm_kv_gib
                got = True
        if b["host"]:
            total += sum(b["host"]) / (2 ** 30)
            got = True
        if got:
            series.append((float(ts), total))
    return series, unit


# ---------------------------------------------------------------- profiles side

def session_token_stats(ap_mod, e0, profiles: Path,
                        trace: Path | None) -> dict:
    """{n_sessions, turns_per_session: stats, start_tokens: stats,
    end_tokens: stats}. start = first turn effective input (input +
    cache.read); end = last token-bearing turn effective input + output."""
    sessions = ap_mod.load_sessions(profiles)
    if trace is not None:
        keep = e0.trace_session_ids(trace)
        sessions = {sid: s for sid, s in sessions.items() if sid in keep}
    turn_counts: list[float] = []
    starts: list[float] = []
    ends: list[float] = []
    for s in sessions.values():
        turns = [s.turns[k] for k in sorted(s.turns)]
        if not turns:
            continue
        turn_counts.append(float(len(turns)))
        toked = [t for t in turns if t.llm_effective_input is not None]
        if not toked:
            continue
        starts.append(float(toked[0].llm_effective_input))
        last = toked[-1]
        end = last.llm_effective_input + (last.llm_output_tokens or 0)
        ends.append(float(end))
    return {
        "n_sessions": len(turn_counts),
        "turns_per_session": _stats3(turn_counts),
        "start_tokens": _stats3(starts),
        "end_tokens": _stats3(ends),
    }


# ---------------------------------------------------------------- figure

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_kv(per_run: dict[str, tuple[list[tuple[float, float]], str]],
           path: Path) -> None:
    """Left: total KV size vs time (t=0 at each run's first sample).
    Right: CDF of the same per-tick samples."""
    plt = _mpl()
    fig, (ax_t, ax_c) = plt.subplots(1, 2, figsize=(12, 4.2))
    units = {u for _, u in per_run.values() if _}
    ylab = "total KV size (GiB)" if units != {"fraction"} else \
        "HBM KV usage (fraction)"
    for label, (series, _unit) in per_run.items():
        if not series:
            continue
        t0 = series[0][0]
        xs = [ts - t0 for ts, _ in series]
        ys = [v for _, v in series]
        ax_t.plot(xs, ys, lw=1.0, label=label)
        s = sorted(ys)
        cdf = [(i + 1) / len(s) for i in range(len(s))]
        ax_c.plot(s, cdf, lw=1.2, label=label)
    ax_t.set_xlabel("time (s)")
    ax_t.set_ylabel(ylab)
    ax_t.set_title("KV cache size over time (host DRAM + HBM)")
    ax_t.legend(loc="upper right", fontsize=8)
    ax_c.set_xlabel(ylab)
    ax_c.set_ylabel("CDF")
    ax_c.set_ylim(0, 1)
    ax_c.set_title("KV cache size CDF")
    ax_c.legend(loc="lower right", fontsize=8)
    for ax in (ax_t, ax_c):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------- main

SUMMARY_COLS = [
    "run", "avg_batch_size", "total_input_tokens",
    "kv_hbm_frac_mean", "kv_hbm_gib_mean", "kv_host_gib_mean",
    "ttft_ms_mean", "ttft_ms_p50", "ttft_ms_p90",
    "tpot_ms_mean", "tpot_ms_p50", "tpot_ms_p90",
    "turns_per_session_mean",
]

TOKEN_COLS = [
    "run", "n_sessions",
    "start_tokens_mean", "start_tokens_p50", "start_tokens_p90",
    "end_tokens_mean", "end_tokens_p50", "end_tokens_p90",
]


def _fmt(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+",
                    help="run dirs, optionally label=path")
    ap.add_argument("--out", type=Path, default=Path("e6_kv"))
    ap.add_argument("--hbm-kv-gib", type=float, default=None,
                    help="total GPU KV-pool size across workers (GiB); "
                         "converts kv_cache_usage_perc into GiB for the "
                         "time-series/CDF")
    ap.add_argument("--host-kv-metric", default=None,
                    help="exact host-DRAM KV usage metric name (bytes "
                         "gauge); default: auto-detect lmcache:* "
                         "usage/size metric")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    e4 = _load("_e6_e4", _E4_PATH)
    ap_mod = _load("_e6_ap", _AP_PATH)
    e0 = _load("_e6_e0", _E0_PATH)

    summary_rows: list[dict] = []
    token_rows: list[dict] = []
    kv_per_run: dict[str, tuple[list[tuple[float, float]], str]] = {}

    for entry in args.runs:
        run = resolve_run(entry)
        label = run["label"]
        row: dict = {"run": label}
        print(f"\n=== {label} ({run['root']}) ===")

        window = None
        if "frontend" in run:
            fe_rows = parse_frontend(e4, run["frontend"])
            window = e4.run_window(fe_rows)
            inp = [r["input_tokens"] for r in fe_rows
                   if r.get("input_tokens") is not None]
            row["total_input_tokens"] = int(sum(inp)) if inp else None
            ttft = [r["ttft_ms"] for r in fe_rows]
            for k, v in _stats3(ttft).items():
                row[f"ttft_ms_{k}"] = v
            tpot = [(r["elapsed_ms"] - r["ttft_ms"]) / (r["output_tokens"] - 1)
                    for r in fe_rows
                    if r.get("output_tokens") and r["output_tokens"] > 1]
            for k, v in _stats3(tpot).items():
                row[f"tpot_ms_{k}"] = v
            print(f"  frontend: {len(fe_rows)} requests, "
                  f"total input tokens {row['total_input_tokens']}")
        else:
            print("  (no frontend.log — TTFT/TPOT/input-token stats skipped)")

        if "metrics" in run:
            batch = e4.load_batch_sizes(run["metrics"], window)
            pooled = [v for vs in batch.values() for v in vs]
            row["avg_batch_size"] = _mean(pooled)
            host_metric = args.host_kv_metric or \
                detect_host_kv_metric(run["metrics"])
            if host_metric:
                print(f"  host KV metric: {host_metric}")
            series, unit = kv_series(run["metrics"], window,
                                     args.hbm_kv_gib, host_metric)
            kv_per_run[label] = (series, unit)
            hbm = e4.load_batch_sizes(run["metrics"], window,
                                      metric="vllm:kv_cache_usage_perc")
            hbm_fracs = [v for vs in hbm.values() for v in vs]
            row["kv_hbm_frac_mean"] = _mean(hbm_fracs)
            if args.hbm_kv_gib is not None and hbm_fracs:
                row["kv_hbm_gib_mean"] = _mean(hbm_fracs) * args.hbm_kv_gib
            if host_metric:
                host_vals = [v for vs in e4.load_batch_sizes(
                    run["metrics"], window, metric=host_metric).values()
                    for v in vs]
                if host_vals:
                    row["kv_host_gib_mean"] = _mean(host_vals) / (2 ** 30)
            if args.hbm_kv_gib is None and host_metric:
                print("  NOTE: --hbm-kv-gib not given — HBM tier excluded "
                      "from the GiB total (host tier only)")
        else:
            print("  (no vllm_metrics.ndjson — batch/KV stats skipped)")

        if "profiles" in run:
            st = session_token_stats(ap_mod, e0, run["profiles"],
                                     run.get("trace"))
            row["turns_per_session_mean"] = st["turns_per_session"]["mean"]
            trow = {"run": label, "n_sessions": st["n_sessions"]}
            for k, v in st["start_tokens"].items():
                trow[f"start_tokens_{k}"] = v
            for k, v in st["end_tokens"].items():
                trow[f"end_tokens_{k}"] = v
            token_rows.append(trow)
            print(f"  sessions: {st['n_sessions']}, "
                  f"turns/session mean {st['turns_per_session']['mean']:.2f} "
                  f"p50 {st['turns_per_session']['p50']:.1f} "
                  f"p90 {st['turns_per_session']['p90']:.1f}")
            print(f"  session start tokens: "
                  f"mean {st['start_tokens']['mean']:.0f} "
                  f"p50 {st['start_tokens']['p50']:.0f} "
                  f"p90 {st['start_tokens']['p90']:.0f}")
            print(f"  session end tokens:   "
                  f"mean {st['end_tokens']['mean']:.0f} "
                  f"p50 {st['end_tokens']['p50']:.0f} "
                  f"p90 {st['end_tokens']['p90']:.0f}")
        else:
            print("  (no profiles — turn/session-token stats skipped)")

        summary_rows.append(row)

    # ---- CSVs
    sp = args.out / "summary.csv"
    with sp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: _fmt(r.get(k)) for k in SUMMARY_COLS})
    print(f"\nwrote {sp}")

    if token_rows:
        tp = args.out / "session_tokens.csv"
        with tp.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=TOKEN_COLS, extrasaction="ignore")
            w.writeheader()
            for r in token_rows:
                w.writerow({k: _fmt(r.get(k)) for k in TOKEN_COLS})
        print(f"wrote {tp}")

    if not args.no_figures and any(s for s, _ in kv_per_run.values()):
        fig_kv(kv_per_run, args.out / "fig_kv_over_time.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
