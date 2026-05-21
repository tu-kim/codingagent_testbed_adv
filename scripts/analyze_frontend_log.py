#!/usr/bin/env python3
"""Parse Dynamo `frontend.log` `request completed` lines into per-request
metrics and emit paper-style distribution figures + CSVs.

The frontend log carries one summary line per HTTP request that landed
on `/v1/chat/completions`. Format (after stripping ANSI colour codes
that tracing-subscriber's pretty formatter injects):

    <ISO ts>  INFO http-request: dynamo_llm::http::service::metrics:
    request completed request_id=<a> model=<m> endpoint=chat_completions
    request_type=stream status=success elapsed_ms=297 method=POST
    uri=/v1/chat/completions version=HTTP/1.1 request_id=<b>
    model="<m>" input_tokens=1187 output_tokens=13
    ttft_ms="187.48" avg_itl_ms="9.07"

We extract / derive these per request:

    elapsed_ms       end-to-end wall (frontend's InflightGuard timer)
    ttft_ms          time-to-first-token (== prefill + first-chunk wire)
    decode_ms        = elapsed_ms - ttft_ms                          (derived)
    itl_ms_per_token = (elapsed_ms - ttft_ms) / output_tokens         (derived;
                       NOT dynamo's avg_itl_ms -- that one's denominator
                       excludes first-chunk tokens and is noisy. This
                       value is the clean per-token decode wall mean.)
    input_tokens     ISL
    output_tokens    OSL
    isl_osl_ratio    = input_tokens / output_tokens                   (derived)

Outputs (under --output dir):

    requests.csv                 per-request raw rows (joinable)
    summary_stats.csv            avg / median / p90 / p99 per metric
    fig_e2e_latency.pdf          elapsed_s histogram + stat lines
    fig_ttft.pdf                 ttft_s histogram
    fig_itl.pdf                  itl_ms_per_token histogram
    fig_tokens.pdf               input / output token side-by-side
    fig_isl_osl_ratio.pdf        ISL/OSL ratio histogram

Usage:
    scripts/analyze_frontend_log.py \\
        --input logs/frontend.log \\
        --output results/run1/frontend_figs \\
        [--model qwen3-coder-30b-a3b-instruct-fp8]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------- paper style ----------

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.figsize": (3.3, 2.3),
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# ---------- log parsing ----------

# tracing-subscriber's pretty formatter wraps keys + `=` separators in
# SGR codes even when stdout isn't a TTY. Strip them per line before
# field regex matching.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJ]")


def _field(name: str, valpat: str) -> re.Pattern[str]:
    """Tolerant `k=v` / `k="v"` / `"k":v` / `"k":"v"` matcher."""
    return re.compile(
        rf'(?:\b|")\b{name}\b"?\s*[=:]\s*"?(?P<v>{valpat})"?'
    )


_ELAPSED_RE = _field("elapsed_ms", r"\d+")
_TTFT_RE = _field("ttft_ms", r"[\d.]+")
_INPUT_RE = _field("input_tokens", r"\d+")
_OUTPUT_RE = _field("output_tokens", r"\d+")
_STATUS_RE = _field("status", r"\w+")
_MODEL_RE = _field("model", r"[^\s\",}]+")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def parse_frontend_log(path: Path, model_filter: str | None = None) -> list[dict]:
    """One row per `request completed` line. Missing fields → None."""
    rows: list[dict] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = strip_ansi(raw)
            if "request completed" not in line:
                continue
            # `model=` appears twice (bare from InflightGuard, quoted
            # from span-recorded finalize); take the last (authoritative).
            models = _MODEL_RE.findall(line)
            model = models[-1] if models else None
            if model_filter and model != model_filter:
                continue
            elapsed = _ELAPSED_RE.search(line)
            ttft = _TTFT_RE.search(line)
            inp = _INPUT_RE.search(line)
            out = _OUTPUT_RE.search(line)
            status = _STATUS_RE.search(line)
            # Require at minimum elapsed + tokens; ttft can be missing
            # on error/very-short responses.
            if not (elapsed and inp and out):
                continue
            elapsed_ms = float(elapsed.group("v"))
            ttft_ms = float(ttft.group("v")) if ttft else None
            input_tokens = int(inp.group("v"))
            output_tokens = int(out.group("v"))
            decode_ms = (elapsed_ms - ttft_ms) if ttft_ms is not None else None
            # User-defined ITL: clean per-token decode wall mean.
            # NOT dynamo's avg_itl_ms field (that one's denominator
            # excludes first-chunk tokens, which makes it sensitive to
            # parser-buffering patterns).
            itl_ms_per_token = (
                decode_ms / output_tokens
                if (decode_ms is not None and output_tokens > 0)
                else None
            )
            isl_osl = (
                input_tokens / output_tokens if output_tokens > 0 else None
            )
            rows.append({
                "model": model,
                "status": status.group("v") if status else None,
                "elapsed_ms": elapsed_ms,
                "ttft_ms": ttft_ms,
                "decode_ms": decode_ms,
                "itl_ms_per_token": itl_ms_per_token,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "isl_osl_ratio": isl_osl,
            })
    return rows


# ---------- stats helpers ----------


def _stats(values) -> dict[str, float | None]:
    """Return mean / median / p90 / p99 of a numeric sequence.
    Empty input yields `None` for each stat (CSV writers turn this into
    an empty cell; previous behavior of returning NaN silently spelled
    the literal string "nan" into the CSV)."""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        return {"mean": None, "median": None, "p90": None, "p99": None, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "n": int(arr.size),
    }


def _fmt_stat(v) -> str:
    """Format a stat value for CSV. Empty string for None / NaN so
    downstream tools can distinguish 'no data' from a real 0.0."""
    import math
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return f"{v:.4f}"


def _stat_lines(ax, values, *, unit: str = "", fmt: str = ".2f") -> None:
    """Draw mean / p90 / p99 vertical lines with inline small-font labels.
    (median omitted by user request -- avg + p90 + p99 only.)"""
    s = _stats(values)
    if s["n"] == 0:
        return
    trans = ax.get_xaxis_transform()
    label_y = [0.97, 0.85, 0.73]
    for i, (name, color) in enumerate([("mean", "C0"), ("p90", "C2"), ("p99", "C3")]):
        v = s[name]
        ax.axvline(v, color=color, linestyle="--", linewidth=0.7, alpha=0.85)
        ax.text(
            v, label_y[i], f"{name}={v:{fmt}}{unit}",
            transform=trans, va="top", ha="left", fontsize=6.0, color=color,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=0.5),
        )


# ---------- plotting ----------


def _hist(ax, values, *, bins: int = 40, xlabel: str, unit: str = "",
          fmt: str = ".2f", color: str = "0.5") -> None:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if arr.size == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return
    ax.hist(arr, bins=bins, color=color, edgecolor="black", linewidth=0.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Request count")
    _stat_lines(ax, arr, unit=unit, fmt=fmt)


def plot_e2e_latency(rows: list[dict], out: Path) -> Path:
    vals = [r["elapsed_ms"] / 1000.0 for r in rows if r["elapsed_ms"] is not None]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    _hist(ax, vals, xlabel="E2E latency (s)", unit="s", fmt=".2f")
    fig.tight_layout()
    path = out / "fig_e2e_latency.pdf"
    fig.savefig(path); plt.close(fig)
    return path


def plot_ttft(rows: list[dict], out: Path) -> Path:
    vals = [r["ttft_ms"] / 1000.0 for r in rows if r["ttft_ms"] is not None]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    _hist(ax, vals, xlabel="TTFT (s)", unit="s", fmt=".2f")
    fig.tight_layout()
    path = out / "fig_ttft.pdf"
    fig.savefig(path); plt.close(fig)
    return path


def plot_itl(rows: list[dict], out: Path) -> Path:
    """ITL in ms/token (= (elapsed - ttft) / output_tokens)."""
    vals = [r["itl_ms_per_token"] for r in rows if r["itl_ms_per_token"] is not None]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    _hist(ax, vals, xlabel="ITL (ms/token)", unit="ms", fmt=".2f")
    fig.tight_layout()
    path = out / "fig_itl.pdf"
    fig.savefig(path); plt.close(fig)
    return path


def plot_tokens(rows: list[dict], out: Path) -> Path:
    """Two-panel: input_tokens vs output_tokens distributions."""
    inp = [r["input_tokens"] for r in rows if r["input_tokens"] is not None]
    outp = [r["output_tokens"] for r in rows if r["output_tokens"] is not None]
    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(6.6, 2.4))
    _hist(ax_in,  inp,  xlabel="Input tokens",  unit="",  fmt=".0f")
    _hist(ax_out, outp, xlabel="Output tokens", unit="",  fmt=".0f")
    ax_in.set_title("(a) ISL", fontsize=9)
    ax_out.set_title("(b) OSL", fontsize=9)
    fig.tight_layout()
    path = out / "fig_tokens.pdf"
    fig.savefig(path); plt.close(fig)
    return path


def plot_isl_osl_ratio(rows: list[dict], out: Path) -> Path:
    vals = [r["isl_osl_ratio"] for r in rows if r["isl_osl_ratio"] is not None]
    fig, ax = plt.subplots(figsize=(3.3, 2.3))
    _hist(ax, vals, xlabel="ISL / OSL ratio per request", unit="", fmt=".1f")
    fig.tight_layout()
    path = out / "fig_isl_osl_ratio.pdf"
    fig.savefig(path); plt.close(fig)
    return path


# ---------- CSV ----------


def write_requests_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = [
        "model", "status",
        "elapsed_ms", "ttft_ms", "decode_ms", "itl_ms_per_token",
        "input_tokens", "output_tokens", "isl_osl_ratio",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({
                "model": r.get("model") or "",
                "status": r.get("status") or "",
                "elapsed_ms": r["elapsed_ms"],
                "ttft_ms": r["ttft_ms"] if r["ttft_ms"] is not None else "",
                "decode_ms": (f"{r['decode_ms']:.3f}" if r["decode_ms"] is not None else ""),
                "itl_ms_per_token": (f"{r['itl_ms_per_token']:.4f}"
                                     if r["itl_ms_per_token"] is not None else ""),
                "input_tokens": r["input_tokens"],
                "output_tokens": r["output_tokens"],
                "isl_osl_ratio": (f"{r['isl_osl_ratio']:.4f}"
                                  if r["isl_osl_ratio"] is not None else ""),
            })


def write_stats_csv(rows: list[dict], path: Path) -> None:
    columns = [
        ("elapsed_ms",      [r["elapsed_ms"] for r in rows]),
        ("ttft_ms",         [r["ttft_ms"] for r in rows]),
        ("decode_ms",       [r["decode_ms"] for r in rows]),
        ("itl_ms_per_token", [r["itl_ms_per_token"] for r in rows]),
        ("input_tokens",    [r["input_tokens"] for r in rows]),
        ("output_tokens",   [r["output_tokens"] for r in rows]),
        ("isl_osl_ratio",   [r["isl_osl_ratio"] for r in rows]),
    ]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "n", "mean", "median", "p90", "p99"])
        for name, vals in columns:
            s = _stats(vals)
            w.writerow([
                name, s["n"],
                _fmt_stat(s["mean"]), _fmt_stat(s["median"]),
                _fmt_stat(s["p90"]), _fmt_stat(s["p99"]),
            ])


def print_stats_table(rows: list[dict]) -> None:
    columns = [
        ("elapsed_ms",      [r["elapsed_ms"] for r in rows]),
        ("ttft_ms",         [r["ttft_ms"] for r in rows]),
        ("decode_ms",       [r["decode_ms"] for r in rows]),
        ("itl_ms_per_token", [r["itl_ms_per_token"] for r in rows]),
        ("input_tokens",    [r["input_tokens"] for r in rows]),
        ("output_tokens",   [r["output_tokens"] for r in rows]),
        ("isl_osl_ratio",   [r["isl_osl_ratio"] for r in rows]),
    ]
    print()
    print(f"Per-request stats from frontend.log (n={len(rows)} requests):")
    hdr = f"{'metric':<20} {'n':>5} {'mean':>11} {'median':>11} {'p90':>11} {'p99':>11}"
    print(hdr)
    print("-" * len(hdr))
    def _p(v):
        return f"{v:>11.3f}" if v is not None else f"{'-':>11}"
    for name, vals in columns:
        s = _stats(vals)
        print(f"{name:<20} {s['n']:>5} {_p(s['mean'])} {_p(s['median'])} "
              f"{_p(s['p90'])} {_p(s['p99'])}")


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, type=Path,
                    help="Path to dynamo frontend.log")
    ap.add_argument("--output", required=True, type=Path,
                    help="Directory for figures + CSVs")
    ap.add_argument("--model", default=None,
                    help="Filter requests by model= value")
    args = ap.parse_args(argv)

    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 2
    args.output.mkdir(parents=True, exist_ok=True)

    rows = parse_frontend_log(args.input, args.model)
    print(f"parsed {len(rows)} request_completed rows from {args.input}")
    if not rows:
        print("nothing to plot", file=sys.stderr)
        return 1

    # Pretty stats first so users see numbers even if matplotlib errors.
    print_stats_table(rows)

    # CSVs
    write_requests_csv(rows, args.output / "requests.csv")
    write_stats_csv(rows, args.output / "summary_stats.csv")
    print(f"  wrote {args.output / 'requests.csv'}")
    print(f"  wrote {args.output / 'summary_stats.csv'}")

    # Figures
    plt.rcParams.update(PAPER_STYLE)
    for fn in (plot_e2e_latency, plot_ttft, plot_itl, plot_tokens, plot_isl_osl_ratio):
        path = fn(rows, args.output)
        print(f"  wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
