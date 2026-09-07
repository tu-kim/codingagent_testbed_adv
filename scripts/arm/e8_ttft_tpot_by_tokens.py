#!/usr/bin/env python3
"""E8: TTFT and TPOT as a function of token count.

TTFT here is QUEUE-REMOVED:

    ttft_net_ms = ttft_ms - prefill_queue_ms

`ttft_ms` (frontend "request completed") spans HTTP receipt -> first
token, so it contains the engine SCHEDULER QUEUE WAIT, which is a
function of load, not of prompt size.  Subtracting the per-request
SCHED_DELAY queue_ms (dynamo-scheduling-log.patch) leaves the
prefill-compute-ish remainder -- the part that should actually scale
with the number of tokens prefilled.  Same definition as
analyze_request_wait.py's `prefill_compute_ms`.  In AGG (colocation)
runs there is no role=prefill record; the decode record's queue_ms is
the same engine queue wait for the single scheduling event, so it is
used as the fallback (the `queue_role` column says which applied).

TPOT (== ITL) is per output token AFTER the first:

    tpot_ms = (elapsed_ms - ttft_ms) / max(output_tokens - 1, 1)

with `output_tokens` taken from the PROFILE (`llm.end.tokens.output`);
the frontend's own output_tokens is truncated (see CLAUDE.md) and would
inflate TPOT by orders of magnitude (this is the source of the absurd
`avg_itl_ms` ~31 s values: a 62 s decode divided by a 2-token count).
Requests the profile has no OSL for keep the truncated count and are
therefore EXCLUDED from every TPOT aggregate and from the figure --
they stay in ttft_tpot.csv tagged `osl_source=frontend` so the drop is
auditable.  Running without --profiles is allowed but warns, and leaves
the TPOT table empty for that reason.

Token axes.  TTFT is bucketed on TWO different counts, because with
prefix caching they diverge sharply:
  prompt_tokens    = tokens.input + tokens.cache.read   (the full ISL)
  reprefill_tokens = tokens.input                       (actually computed;
                                                         cached prefix is
                                                         ~free)
The reprefill axis is the one TTFT should scale linearly with; a flat
prompt-token curve with a steep reprefill curve is the signature of
prefix-cache reuse doing its job.  Both need --profiles; without it
only the raw per-request table is produced.
TPOT is bucketed on output_tokens.

Inputs:
  --frontend <path>  dynamo frontend log (elapsed_ms, ttft_ms per request)
  --logs <path>      worker log file/dir with SCHED_DELAY lines (queue removal)
  --profiles <dir>   opencode profile NDJSON dir (trusted OSL + ISL split)
  --out <dir>        output dir (default: e8_ttft_tpot)
  --no-figures       skip PDF rendering

Outputs (under --out):
  ttft_tpot.csv              one row per request (all columns above)
  ttft_by_prompt_tokens.csv  bucketed: n, mean/p50/p90 ttft_net_ms, per-token us
  ttft_by_reprefill.csv      same, bucketed on re-prefilled tokens
  tpot_by_output_tokens.csv  bucketed: n, mean/p50/p90 tpot_ms
  prefill_by_reprefill_grid.csv / _prompt_ / _cached_
                             prefill time at REPRESENTATIVE token counts --
                             a uniform --grid-step grid (1k, 2k, 3k, ...)
                             rather than log buckets, so rows compare directly
  prefill_grid_2d.csv        prefill p50 over the (computed, reused) token
                             plane on the same grid
  fig1_ttft_vs_tokens.pdf    scatter + bucket-p50 line (prompt | reprefill)
  fig2_tpot_vs_tokens.pdf    scatter + bucket-p50 line (output tokens)
  stdout                     the same three tables

Usage:
  scripts/arm/e8_ttft_tpot_by_tokens.py --frontend logs/frontend.log \
      --logs logs/ --profiles <workspace_root>/profiles --out results/e8
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ATS_PATH = _HERE.parents[1] / "analyze_turn_scheduling.py"
_E4_PATH = _HERE.parent / "e4_prefill_decode.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Bucket edges are left-inclusive upper bounds; the last bucket is open.
PROMPT_BINS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
OUTPUT_BINS = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def _bucket(v: float, bins: list[int]) -> str:
    lo = 0
    for b in bins:
        if v < b:
            return f"{lo}-{b}"
        lo = b
    return f"{lo}+"


def _bucket_order(bins: list[int]) -> list[str]:
    lo = 0
    labels = []
    for b in bins:
        labels.append(f"{lo}-{b}")
        lo = b
    labels.append(f"{lo}+")
    return labels


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return math.nan
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def profile_isl(profiles: Path) -> dict[str, dict]:
    """{request_id: {prompt_tokens, reprefill_tokens, cached_tokens}} from
    profile `llm.end.tokens`.

    CLAUDE.md invariant: `tokens.input` ALREADY has the cached tokens
    subtracted, so it IS the re-prefilled count and the real ISL is
    input + cache.read.
    """
    import json
    files = [profiles] if profiles.is_file() else sorted(profiles.glob("*.jsonl"))
    out: dict[str, dict] = {}
    for f in files:
        with f.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if '"llm.end"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("ev") != "llm.end":
                    continue
                rid = ev.get("request_id")
                tok = ev.get("tokens")
                if not rid or not isinstance(tok, dict):
                    continue
                inp = tok.get("input")
                if not isinstance(inp, (int, float)):
                    continue
                cache = tok.get("cache")
                read = (cache.get("read") if isinstance(cache, dict) else 0) or 0
                out[rid] = {
                    "reprefill_tokens": int(inp),
                    "cached_tokens": int(read),
                    "prompt_tokens": int(inp) + int(read),
                }
    return out


def build_rows(front: list[dict], sched: dict, isl: dict[str, dict]) -> tuple[list[dict], dict]:
    """Join frontend ⋈ SCHED_DELAY ⋈ profile-ISL by request_id.

    Requests without a SCHED_DELAY record keep a null queue and are
    EXCLUDED from every TTFT statistic (a raw ttft would silently mix
    queue wait back in); they still contribute to TPOT, which the queue
    does not touch.
    """
    rows = []
    rep = {"n": len(front), "queued": 0, "negative_net": 0, "with_isl": 0,
           "queue_role": {}, "osl_frontend": 0}
    for r in front:
        rid = r["request_id"]
        rec = sched.get(rid)
        q = None
        role = ""
        if rec is not None:
            if rec.prefill_queue_ms is not None:
                q, role = rec.prefill_queue_ms, "prefill"
            elif rec.decode_queue_ms is not None:
                q, role = rec.decode_queue_ms, "decode"
        net = None
        if q is not None:
            rep["queued"] += 1
            rep["queue_role"][role] = rep["queue_role"].get(role, 0) + 1
            net = r["prefill_ms"] - q
            if net < 0:
                # clock/attribution skew: queue_ms exceeding ttft cannot be
                # a real prefill compute time. Drop rather than clamp to 0,
                # which would fabricate a fast-prefill data point.
                rep["negative_net"] += 1
                net = None
        tok = isl.get(rid)
        if tok:
            rep["with_isl"] += 1
        out_tok = r["output_tokens"]
        # apply_profile_tokens() stamps output_tokens_frontend ONLY on the
        # requests it actually corrected. Anything without that stamp is
        # still carrying the frontend's truncated OSL, which is exactly the
        # denominator that produced the absurd ~31 s "avg_itl_ms" values
        # (a 62 s decode divided by a 2-token count). Mark it so TPOT
        # aggregation can drop it instead of averaging the garbage in.
        osl_source = "profile" if "output_tokens_frontend" in r else "frontend"
        if osl_source == "frontend":
            rep["osl_frontend"] += 1
        rows.append({
            "request_id": rid,
            "prompt_tokens": tok["prompt_tokens"] if tok else "",
            "reprefill_tokens": tok["reprefill_tokens"] if tok else "",
            "cached_tokens": tok["cached_tokens"] if tok else "",
            "output_tokens": out_tok,
            "osl_source": osl_source,
            "elapsed_ms": round(r["elapsed_ms"], 3),
            "ttft_ms": round(r["prefill_ms"], 3),
            "queue_ms": round(q, 3) if q is not None else "",
            "queue_role": role,
            "ttft_net_ms": round(net, 3) if net is not None else "",
            "decode_ms": round(r["decode_ms"], 3),
            "tpot_ms": round(r["decode_ms"] / max(out_tok - 1, 1), 4)
                       if out_tok > 1 else "",
        })
    return rows, rep


def bucket_rows(rows: list[dict], token_col: str, value_col: str,
                bins: list[int], per_token_us: bool) -> list[dict]:
    groups: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        t, v = r[token_col], r[value_col]
        if t == "" or v == "":
            continue
        groups.setdefault(_bucket(float(t), bins), []).append((float(t), float(v)))
    out = []
    for label in _bucket_order(bins):
        pairs = groups.get(label)
        if not pairs:
            continue
        vals = [v for _t, v in pairs]
        row = {
            "bucket": label,
            "n": len(vals),
            f"{token_col}_p50": round(_pct([t for t, _v in pairs], 0.50), 1),
            "mean_ms": round(sum(vals) / len(vals), 3),
            "p50_ms": round(_pct(vals, 0.50), 3),
            "p90_ms": round(_pct(vals, 0.90), 3),
        }
        if per_token_us:
            # us per token at the bucket median -- flat across buckets means
            # prefill is bandwidth-bound/linear; rising means quadratic
            # attention or batching effects.
            per = [1000.0 * v / t for t, v in pairs if t > 0]
            row["us_per_token_p50"] = round(_pct(per, 0.50), 2)
        out.append(row)
    return out


def grid_rows(rows: list[dict], token_col: str, value_col: str,
              step: int) -> list[dict]:
    """Prefill time at REPRESENTATIVE token counts: a fixed `step` grid
    (1k, 2k, 3k, ...) instead of the log-spaced buckets.

    Bucket k covers ((k-1)*step, k*step] and is LABELLED by its upper
    edge, so the "2k" row answers "what does a ~2k-token prefill cost".
    Uniform width makes the rows directly comparable -- a log bucket's
    width grows with the value, which hides curvature.
    """
    groups: dict[int, list[tuple[float, float]]] = {}
    for r in rows:
        t, v = r[token_col], r[value_col]
        if t == "" or v == "":
            continue
        t, v = float(t), float(v)
        k = max(1, math.ceil(t / step)) if t > 0 else 1
        groups.setdefault(k, []).append((t, v))
    out = []
    for k in sorted(groups):
        pairs = groups[k]
        vals = [v for _t, v in pairs]
        per = [1000.0 * v / t for t, v in pairs if t > 0]
        out.append({
            "tokens": k * step,
            "label": _ktok(k * step),
            "n": len(vals),
            f"{token_col}_p50": round(_pct([t for t, _v in pairs], 0.50), 1),
            "mean_ms": round(sum(vals) / len(vals), 3),
            "p50_ms": round(_pct(vals, 0.50), 3),
            "p90_ms": round(_pct(vals, 0.90), 3),
            "us_per_token_p50": round(_pct(per, 0.50), 2) if per else "",
        })
    return out


def _ktok(n: int) -> str:
    return f"{n // 1000}k" if n >= 1000 and n % 1000 == 0 else str(n)


def grid2d_rows(rows: list[dict], step: int) -> list[dict]:
    """Prefill time over the (re-prefilled, reused) token PLANE.

    The two axes are the two halves of the prompt: `reprefill_tokens`
    was recomputed, `cached_tokens` was served from the prefix cache.
    If the cache is doing its job, moving ALONG the cached axis at a
    fixed re-prefill count leaves prefill time roughly flat, while
    moving along the re-prefill axis raises it.
    """
    groups: dict[tuple[int, int], list[float]] = {}
    for r in rows:
        rp, ca, v = r["reprefill_tokens"], r["cached_tokens"], r["ttft_net_ms"]
        if rp == "" or ca == "" or v == "":
            continue
        kr = max(1, math.ceil(float(rp) / step)) if float(rp) > 0 else 0
        kc = math.ceil(float(ca) / step) if float(ca) > 0 else 0
        groups.setdefault((kr, kc), []).append(float(v))
    out = []
    for (kr, kc) in sorted(groups):
        vals = groups[(kr, kc)]
        out.append({
            "reprefill_tokens": kr * step,
            "cached_tokens": kc * step,
            "n": len(vals),
            "mean_ms": round(sum(vals) / len(vals), 3),
            "p50_ms": round(_pct(vals, 0.50), 3),
            "p90_ms": round(_pct(vals, 0.90), 3),
        })
    return out


def print_grid(title: str, rows: list[dict], token_col: str) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (no data)")
        return
    print(f"  {'tokens':>8} {'n':>6} {'tok_p50':>9} {'mean_ms':>10} "
          f"{'p50_ms':>10} {'p90_ms':>10} {'us/tok_p50':>11}")
    for r in rows:
        per = r["us_per_token_p50"]
        print(f"  {r['label']:>8} {r['n']:>6} {r[token_col + '_p50']:>9.0f} "
              f"{r['mean_ms']:>10.1f} {r['p50_ms']:>10.1f} {r['p90_ms']:>10.1f} "
              f"{(f'{per:.2f}' if per != '' else '-'):>11}")


def print_grid2d(rows: list[dict], step: int) -> None:
    """Pivot: rows = re-prefilled tokens, cols = reused tokens, cell = p50 ms."""
    print(f"\nPrefill p50 ms over (re-prefilled x reused) tokens, "
          f"{_ktok(step)} grid  [cell = p50 ms (n)]:")
    if not rows:
        print("  (no data)")
        return
    r_ax = sorted({r["reprefill_tokens"] for r in rows})
    c_ax = sorted({r["cached_tokens"] for r in rows})
    cell = {(r["reprefill_tokens"], r["cached_tokens"]): r for r in rows}
    print("  reused ->  " + "".join(f"{_ktok(c):>14}" for c in c_ax))
    for rv in r_ax:
        line = f"  {_ktok(rv):>9}  "
        for cv in c_ax:
            r = cell.get((rv, cv))
            line += (f"{r['p50_ms']:>9.0f}({r['n']:>2})" if r else f"{'-':>14}")
        print(line)


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def print_table(title: str, rows: list[dict], token_col: str) -> None:
    print(f"\n{title}")
    if not rows:
        print("  (no data)")
        return
    extra = "us_per_token_p50" in rows[0]
    head = (f"  {'bucket':>14} {'n':>6} {'tok_p50':>9} {'mean_ms':>10} "
            f"{'p50_ms':>10} {'p90_ms':>10}")
    if extra:
        head += f" {'us/tok_p50':>11}"
    print(head)
    for r in rows:
        line = (f"  {r['bucket']:>14} {r['n']:>6} {r[token_col + '_p50']:>9.0f} "
                f"{r['mean_ms']:>10.1f} {r['p50_ms']:>10.1f} {r['p90_ms']:>10.1f}")
        if extra:
            line += f" {r['us_per_token_p50']:>11.2f}"
        print(line)


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _panel(ax, rows, token_col, value_col, buckets, xlabel, ylabel, title):
    xs = [float(r[token_col]) for r in rows
          if r[token_col] != "" and r[value_col] != ""]
    ys = [float(r[value_col]) for r in rows
          if r[token_col] != "" and r[value_col] != ""]
    if not xs:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        ax.set_title(title)
        return
    ax.scatter(xs, ys, s=6, alpha=0.25, color="tab:blue", label="requests")
    if buckets:
        bx = [r[token_col + "_p50"] for r in buckets]
        ax.plot(bx, [r["p50_ms"] for r in buckets], "o-", color="tab:red",
                lw=1.8, ms=4, label="bucket p50")
        ax.plot(bx, [r["p90_ms"] for r in buckets], "s--", color="tab:orange",
                lw=1.2, ms=3, label="bucket p90")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, framealpha=0.7)


def fig_ttft(rows, b_prompt, b_repre, path: Path) -> None:
    plt = _mpl()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))
    _panel(axL, rows, "prompt_tokens", "ttft_net_ms", b_prompt,
           "prompt tokens (ISL = input + cache.read)", "TTFT - queue (ms)",
           "TTFT (queue-removed) vs prompt tokens")
    _panel(axR, rows, "reprefill_tokens", "ttft_net_ms", b_repre,
           "re-prefilled tokens (tokens.input)", "TTFT - queue (ms)",
           "TTFT (queue-removed) vs re-prefilled tokens")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_prefill_plane(rows, grid_rp, path: Path) -> None:
    """Left: prefill time vs re-prefilled tokens, points COLOURED by how
    many tokens were reused -- a colour-blind cloud means reuse does not
    change prefill cost. Right: the (computed, reused) plane itself,
    colour = prefill p50 ms."""
    plt = _mpl()
    pts = [(float(r["reprefill_tokens"]), float(r["cached_tokens"]),
            float(r["ttft_net_ms"]))
           for r in rows
           if r["reprefill_tokens"] != "" and r["cached_tokens"] != ""
           and r["ttft_net_ms"] != ""]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))
    if not pts:
        for ax in (axL, axR):
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                    ha="center", va="center", color="grey")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    rp = [a for a, _b, _c in pts]
    ca = [b for _a, b, _c in pts]
    ms = [c for _a, _b, c in pts]

    sc = axL.scatter(rp, ms, c=ca, s=10, alpha=0.6, cmap="viridis")
    fig.colorbar(sc, ax=axL, label="reused (cached) tokens")
    if grid_rp:
        axL.plot([r["reprefill_tokens_p50"] for r in grid_rp],
                 [r["p50_ms"] for r in grid_rp], "o-", color="tab:red",
                 lw=1.8, ms=4, label="grid p50")
        axL.legend(fontsize=8, framealpha=0.7)
    axL.set_xlabel("re-prefilled (computed) tokens")
    axL.set_ylabel("prefill time = TTFT - queue (ms)")
    axL.set_title("Prefill time vs computed tokens, coloured by reuse")
    axL.grid(alpha=0.3)

    sc2 = axR.scatter(rp, ca, c=ms, s=14, alpha=0.8, cmap="magma")
    fig.colorbar(sc2, ax=axR, label="prefill time (ms)")
    axR.set_xlabel("re-prefilled (computed) tokens")
    axR.set_ylabel("reused (cached) tokens")
    axR.set_title("Prefill time over the (computed, reused) plane")
    axR.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_tpot(rows, buckets, path: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8, 5))
    _panel(ax, rows, "output_tokens", "tpot_ms", buckets,
           "output tokens", "TPOT (ms/token)", "TPOT vs output tokens")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frontend", required=True, type=Path)
    ap.add_argument("--logs", type=Path, default=None,
                    help="worker log file/dir with SCHED_DELAY lines; without "
                         "it TTFT statistics are skipped (queue not removable)")
    ap.add_argument("--profiles", type=Path, default=None,
                    help="profile NDJSON dir: trusted output_tokens + the "
                         "prompt/re-prefill token split")
    ap.add_argument("--out", type=Path, default=Path("e8_ttft_tpot"))
    ap.add_argument("--grid-step", type=int, default=1000,
                    help="representative-token table step (default 1000 -> "
                         "rows at 1k, 2k, 3k, ...)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    if not args.frontend.exists():
        print(f"error: {args.frontend} not found", file=sys.stderr)
        return 2

    e4 = _load("e4_prefill_decode", _E4_PATH)
    ats = _load("analyze_turn_scheduling", _ATS_PATH)

    front = e4.parse_frontend(args.frontend)
    if not front:
        print("error: no parseable 'request completed' lines", file=sys.stderr)
        return 2

    if args.profiles:
        rep = e4.apply_profile_tokens(front, e4.profile_output_tokens(args.profiles))
        print(f"OSL corrected from profiles: {rep['n_fixed']}/{len(front)} "
              f"requests ({rep['n_grew']} grew)")
        isl = profile_isl(args.profiles)
    else:
        print("warning: no --profiles; output_tokens comes from the frontend "
              "log, which is TRUNCATED -- TPOT will be inflated and no token "
              "buckets can be built", file=sys.stderr)
        isl = {}

    sched = ats.load_sched(args.logs) if args.logs else {}
    if not sched:
        print("warning: no SCHED_DELAY records -- TTFT cannot have its queue "
              "wait removed, so TTFT tables/figures are skipped",
              file=sys.stderr)

    rows, rep = build_rows(front, sched, isl)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "ttft_tpot.csv", rows,
              ["request_id", "prompt_tokens", "reprefill_tokens",
               "cached_tokens", "output_tokens", "osl_source", "elapsed_ms",
               "ttft_ms",
               "queue_ms", "queue_role", "ttft_net_ms", "decode_ms", "tpot_ms"])

    print(f"\nrequests: {rep['n']}  with queue record: {rep['queued']} "
          f"({', '.join(f'{k}={v}' for k, v in sorted(rep['queue_role'].items())) or 'none'})"
          f"  with profile tokens: {rep['with_isl']}")
    if rep["negative_net"]:
        print(f"  dropped {rep['negative_net']} requests with queue_ms > ttft_ms "
              "(clock/attribution skew)")

    b_prompt = bucket_rows(rows, "prompt_tokens", "ttft_net_ms",
                           PROMPT_BINS, True)
    b_repre = bucket_rows(rows, "reprefill_tokens", "ttft_net_ms",
                          PROMPT_BINS, True)
    # TPOT aggregates only over requests whose OSL came from the profile;
    # a truncated frontend count inflates tpot_ms by orders of magnitude.
    tpot_rows = [r for r in rows if r["osl_source"] == "profile"]
    if rep["osl_frontend"]:
        print(f"  excluded {rep['osl_frontend']} requests from TPOT: no "
              "profile OSL, so output_tokens is the TRUNCATED frontend value "
              "(they remain in ttft_tpot.csv, tagged osl_source=frontend)")
    b_tpot = bucket_rows(tpot_rows, "output_tokens", "tpot_ms", OUTPUT_BINS,
                         False)

    write_csv(args.out / "ttft_by_prompt_tokens.csv", b_prompt,
              ["bucket", "n", "prompt_tokens_p50", "mean_ms", "p50_ms",
               "p90_ms", "us_per_token_p50"])
    write_csv(args.out / "ttft_by_reprefill.csv", b_repre,
              ["bucket", "n", "reprefill_tokens_p50", "mean_ms", "p50_ms",
               "p90_ms", "us_per_token_p50"])
    write_csv(args.out / "tpot_by_output_tokens.csv", b_tpot,
              ["bucket", "n", "output_tokens_p50", "mean_ms", "p50_ms",
               "p90_ms"])

    print_table("TTFT (queue-removed) by prompt tokens [ISL = input + cache.read]:",
                b_prompt, "prompt_tokens")
    print_table("TTFT (queue-removed) by RE-PREFILLED tokens [tokens.input]:",
                b_repre, "reprefill_tokens")
    print_table("TPOT by output tokens:", b_tpot, "output_tokens")

    step = args.grid_step
    g_repre = grid_rows(rows, "reprefill_tokens", "ttft_net_ms", step)
    g_prompt = grid_rows(rows, "prompt_tokens", "ttft_net_ms", step)
    g_cached = grid_rows(rows, "cached_tokens", "ttft_net_ms", step)
    g2d = grid2d_rows(rows, step)
    write_csv(args.out / "prefill_by_reprefill_grid.csv", g_repre,
              ["tokens", "label", "n", "reprefill_tokens_p50", "mean_ms",
               "p50_ms", "p90_ms", "us_per_token_p50"])
    write_csv(args.out / "prefill_by_prompt_grid.csv", g_prompt,
              ["tokens", "label", "n", "prompt_tokens_p50", "mean_ms",
               "p50_ms", "p90_ms", "us_per_token_p50"])
    write_csv(args.out / "prefill_by_cached_grid.csv", g_cached,
              ["tokens", "label", "n", "cached_tokens_p50", "mean_ms",
               "p50_ms", "p90_ms", "us_per_token_p50"])
    write_csv(args.out / "prefill_grid_2d.csv", g2d,
              ["reprefill_tokens", "cached_tokens", "n", "mean_ms",
               "p50_ms", "p90_ms"])
    print_grid(f"Prefill time by RE-PREFILLED (computed) tokens, "
               f"{_ktok(step)} grid:", g_repre, "reprefill_tokens")
    print_grid(f"Prefill time by REUSED (cached) tokens, {_ktok(step)} grid:",
               g_cached, "cached_tokens")
    print_grid(f"Prefill time by TOTAL prompt tokens, {_ktok(step)} grid:",
               g_prompt, "prompt_tokens")
    print_grid2d(g2d, step)

    if not args.no_figures:
        try:
            fig_ttft(rows, b_prompt, b_repre, args.out / "fig1_ttft_vs_tokens.pdf")
            fig_prefill_plane(rows, g_repre,
                              args.out / "fig3_prefill_plane.pdf")
            fig_tpot(tpot_rows, b_tpot, args.out / "fig2_tpot_vs_tokens.pdf")
        except ImportError:
            print("matplotlib unavailable -- figures skipped", file=sys.stderr)
    print(f"\noutputs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
