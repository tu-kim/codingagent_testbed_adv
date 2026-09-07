#!/usr/bin/env python3
"""E9 plot: TTFT / TPOT curves per batch size, with SLA limits.

Reads the CSV(s) written by `e9_offline_ttft_tpot_sweep.py` and produces
the tables and figures. Measurement is expensive and happens once on the
GPU host; this side is pure CPU, so the same sweep can be re-analysed
against any SLA, anywhere, as often as needed.

Several --sweep paths can be given and are MERGED on (batch,
prompt_tokens), later files winning. That is how a coarse full-range run
and a targeted long-context tail run (which needs its own, larger
--max-model-len) end up as one curve per batch size.

A --sweep argument may be a DIRECTORY. Runs organised one folder per
batch size -- b1/, b2/, b4/, ... each holding a sweep.csv -- are the
common case: point --sweep at the parent (or list the folders) and each
file's batch size is taken from ITS FOLDER NAME, overriding the CSV
column. A folder whose name does not parse as a batch size leaves the
column in charge.

SLA lines: --ttft-sla-ms / --tpot-sla-ms draw a horizontal threshold on
the matching figure and, per batch-size curve, mark the longest prompt
that still meets it. Two numbers are reported because the sweep grid is
deliberately coarse:
  max_ok_measured -- the largest grid point actually under the SLA
  max_ok_interp   -- linear interpolation of the crossing between that
                     point and the next (failing) one; the measured
                     value alone systematically understates the limit
A curve that never crosses is reported at the top of its grid with
crossed=False, so it is not mistaken for a real ceiling.

Inputs:
  --sweep <path> [...]  sweep.csv files, or directories holding them
                        (incl. a parent of b1/, b2/, b4/, ... folders)
  --ttft-sla-ms <ms>    optional TTFT threshold
  --tpot-sla-ms <ms>    optional TPOT threshold

Outputs (under --out):
  sla_limits.csv        per (metric, batch): longest prompt meeting the SLA
  fig_ttft.pdf          TTFT vs prompt tokens, one line per batch size
  fig_tpot.pdf          TPOT vs prompt tokens, one line per batch size
  stdout                both grid tables, plus an SLA table per threshold

Usage:
  scripts/arm/e9_plot_ttft_tpot_sla.py --sweep results/e9-64k/sweep.csv \
      results/e9-256k/sweep.csv --ttft-sla-ms 2000 --tpot-sla-ms 50 \
      --out results/e9-report
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


_BATCH_DIR_RE = re.compile(r"^b(?:atch)?[-_]?(\d+)$", re.IGNORECASE)


def batch_from_dir(path: Path) -> int | None:
    """Batch size encoded in a directory name: b8, batch8, b_8, B-8 -> 8.

    Returns None for any other name, which leaves the CSV's own `batch`
    column in charge.
    """
    m = _BATCH_DIR_RE.match(path.name)
    return int(m.group(1)) if m else None


def expand_sweep_paths(paths: list[Path]) -> list[tuple[Path, int | None]]:
    """Resolve each --sweep argument to (csv, batch_from_folder) pairs.

    A file is taken as-is. A DIRECTORY is expanded, which is how a run
    laid out one-folder-per-batch-size collapses into a single plot:
      <dir>/sweep.csv          -> that file, batch from <dir>'s name
      <dir>/b1/sweep.csv, ...  -> each file, batch from ITS OWN folder
    A directory holding neither is searched for *.csv as a last resort.
    The folder's batch (when it parses) overrides the CSV column, since
    that is the layout the runs were organised by.
    """
    out: list[tuple[Path, int | None]] = []
    for p in paths:
        if p.is_file():
            out.append((p, batch_from_dir(p.parent)))
            continue
        direct = p / "sweep.csv"
        if direct.is_file():
            out.append((direct, batch_from_dir(p)))
        subs = sorted(d for d in p.iterdir()
                      if d.is_dir() and batch_from_dir(d) is not None)
        for d in subs:
            f = d / "sweep.csv"
            if f.is_file():
                out.append((f, batch_from_dir(d)))
            else:
                out.extend((c, batch_from_dir(d)) for c in sorted(d.glob("*.csv")))
        if not direct.is_file() and not subs:
            out.extend((c, batch_from_dir(p)) for c in sorted(p.glob("*.csv")))
    return out


def merge_sweeps(paths: list[Path]) -> tuple[list[dict], dict]:
    """Merge every resolved CSV into one cell table.

    Later files win on a repeated (batch, prompt_tokens) cell -- a re-run
    is assumed to supersede the earlier attempt (an OOM cell re-measured
    with more memory, say). Returns (rows, report); the report carries
    the resolved file list and how many rows had their `batch` column
    rewritten by the folder name, so a mislabelled folder shows up in the
    output instead of silently relabelling the data.
    """
    cells: dict[tuple[int, int], dict] = {}
    resolved = expand_sweep_paths(paths)
    report = {"files": [str(f) for f, _b in resolved], "overridden": 0,
              "mismatched": 0}
    for f, folder_batch in resolved:
        for r in read_csv(f):
            batch = folder_batch
            if batch is None:
                batch = int(r["batch"])
            else:
                report["overridden"] += 1
                if r.get("batch") not in ("", None) and int(r["batch"]) != batch:
                    report["mismatched"] += 1
                r = dict(r, batch=batch)
            cells[(batch, int(r["prompt_tokens"]))] = r
    return [cells[k] for k in sorted(cells)], report


def write_csv(path: Path, rows: list[dict], cols: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sla_limits(rows: list[dict], metric: str, sla: float) -> list[dict]:
    """Per batch size, the longest prompt whose `metric` still meets `sla`.

    Two numbers, because a coarse grid cannot express the real limit:
      max_ok_measured -- the largest grid point actually under the SLA
      max_ok_interp   -- linear interpolation of the crossing between that
                         point and the next (failing) one
    A curve that never crosses reports the top of the grid with
    crossed=False; one that fails at its very first point reports 0.
    """
    by_batch: dict[int, list[tuple[int, float]]] = {}
    for r in rows:
        if r.get("error") or r.get(metric) in ("", None):
            continue
        by_batch.setdefault(int(r["batch"]), []).append(
            (int(r["prompt_tokens"]), float(r[metric])))
    out = []
    for batch in sorted(by_batch):
        pts = sorted(by_batch[batch])
        ok = [(x, y) for x, y in pts if y <= sla]
        if not ok:
            out.append({"metric": metric, "sla": sla, "batch": batch,
                        "max_ok_measured": 0, "max_ok_interp": 0,
                        "crossed": True})
            continue
        x_ok, y_ok = ok[-1]
        nxt = next(((x, y) for x, y in pts if x > x_ok), None)
        if nxt is None:
            out.append({"metric": metric, "sla": sla, "batch": batch,
                        "max_ok_measured": x_ok, "max_ok_interp": x_ok,
                        "crossed": False})
            continue
        x_bad, y_bad = nxt
        span = y_bad - y_ok
        frac = (sla - y_ok) / span if span > 0 else 0.0
        out.append({"metric": metric, "sla": sla, "batch": batch,
                    "max_ok_measured": x_ok,
                    "max_ok_interp": int(round(x_ok + frac * (x_bad - x_ok))),
                    "crossed": True})
    return out


# ---------- output ----------


# The model's full context; the x axis is pinned here so every figure
# spans the same range regardless of how far a given sweep got.
X_MAX_TOKENS = 262144


def _ktok(n: int) -> str:
    return f"{n // 1024}k" if n >= 1024 and n % 1024 == 0 else str(n)


def print_sweep(rows: list[dict], metric: str, label: str) -> None:
    batches = sorted({int(r["batch"]) for r in rows})
    lengths = sorted({int(r["prompt_tokens"]) for r in rows})
    cell = {(int(r["batch"]), int(r["prompt_tokens"])): r for r in rows}
    print(f"\n{label} (rows = batch size, cols = prompt tokens):")
    print("  batch \\ len " + "".join(f"{_ktok(n):>12}" for n in lengths))
    for b in batches:
        line = f"  {b:>11} "
        for n in lengths:
            r = cell.get((b, n))
            if r is None:
                line += f"{'-':>12}"
            elif r.get("error"):
                line += f"{'x':>12}"
            else:
                line += f"{float(r[metric]):>12.1f}"
        print(line)


def print_sla(limits: list[dict], unit: str) -> None:
    if not limits:
        return
    metric = limits[0]["metric"]
    sla = limits[0]["sla"]
    print(f"\nLongest prompt meeting {metric} <= {sla:g} {unit}:")
    print(f"  {'batch':>7} {'measured':>12} {'interpolated':>14}  note")
    for r in limits:
        note = "" if r["crossed"] else "(never crossed within the grid)"
        print(f"  {r['batch']:>7} {r['max_ok_measured']:>12} "
              f"{r['max_ok_interp']:>14}  {note}")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_metric(rows: list[dict], metric: str, ylabel: str, title: str,
                sla: float | None, limits: list[dict], path: Path) -> None:
    """One line per batch size, plus the SLA line and its per-batch
    crossing markers.

    When the sweep carried repetitions, `<metric>_min`/`_max` columns are
    present and each curve gets a shaded min-max band -- a wide band says
    the point is not reproducible enough to read a limit off. Sweeps run
    at repeat=1 (or written before the columns existed) simply have no
    band.
    """
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    by_batch: dict[int, list[tuple[int, float]]] = {}
    bands: dict[int, list[tuple[int, float, float]]] = {}
    for r in rows:
        if r.get("error") or r.get(metric) in ("", None):
            continue
        x, y = int(r["prompt_tokens"]), float(r[metric])
        by_batch.setdefault(int(r["batch"]), []).append((x, y))
        lo, hi = r.get(f"{metric}_min"), r.get(f"{metric}_max")
        if lo not in ("", None) and hi not in ("", None):
            bands.setdefault(int(r["batch"]), []).append(
                (x, float(lo), float(hi)))
    if not by_batch:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                ha="center", va="center", color="grey")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return
    cmap = plt.get_cmap("viridis")
    batches = sorted(by_batch)
    for i, b in enumerate(batches):
        pts = sorted(by_batch[b])
        colour = cmap(i / max(len(batches) - 1, 1) * 0.85)
        ax.plot([x for x, _y in pts], [y for _x, y in pts], "o-",
                color=colour, lw=1.8, ms=4, label=f"batch {b}")
        band = sorted(bands.get(b, []))
        if band and any(hi > lo for _x, lo, hi in band):
            ax.fill_between([x for x, _lo, _hi in band],
                            [lo for _x, lo, _hi in band],
                            [hi for _x, _lo, hi in band],
                            color=colour, alpha=0.15, linewidth=0)
    if sla is not None:
        # Labelled ON the axes, not in the legend: the legend is for the
        # batch-size curves, and the threshold reads better sitting on the
        # line it describes.
        ax.axhline(sla, color="tab:red", ls="--", lw=1.4)
        ax.text(0.995, sla, f"SLA {sla:g}", transform=ax.get_yaxis_transform(),
                ha="right", va="bottom", fontsize=9, color="tab:red",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none",
                          alpha=0.75))
        lim = {r["batch"]: r for r in limits}
        for i, b in enumerate(batches):
            r = lim.get(b)
            if not r or not r["max_ok_interp"]:
                continue
            colour = cmap(i / max(len(batches) - 1, 1) * 0.85)
            x = r["max_ok_interp"]
            ax.axvline(x, color=colour, ls=":", lw=1.1, alpha=0.8)
            ax.annotate(f"b{b}: {int(x)}", xy=(x, sla),
                        xytext=(0, 8 + 13 * i), textcoords="offset points",
                        ha="center", fontsize=8, color=colour,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white",
                                  ec="none", alpha=0.75))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # Ticks at the powers of two the sweep grid uses, labelled 1k..256k,
    # and the axis pinned to the model's full 256k context so figures from
    # different runs are directly comparable side by side.
    ticks = [1 << e for e in range(10, 19)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_ktok(t) for t in ticks])
    ax.set_xticks([], minor=True)
    xs_all = [x for pts in by_batch.values() for x, _y in pts]
    ax.set_xlim(min(min(xs_all), 1024) * 0.85, X_MAX_TOKENS * 1.05)
    ax.set_xlabel("prompt tokens")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------



def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", required=True, nargs="+", type=Path,
                    help="sweep.csv file(s) OR directories. A directory is "
                         "expanded to its own sweep.csv and to every b<N>/ "
                         "(batch-size) subfolder's, with the folder name "
                         "setting that data's batch size -- so a per-batch "
                         "run layout plots as one figure")
    ap.add_argument("--ttft-sla-ms", type=float, default=None)
    ap.add_argument("--tpot-sla-ms", type=float, default=None)
    ap.add_argument("--out", type=Path, default=Path("e9_report"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    missing = [p for p in args.sweep if not p.exists()]
    if missing:
        print(f"error: not found: {', '.join(str(p) for p in missing)}",
              file=sys.stderr)
        return 2
    rows, rep = merge_sweeps(args.sweep)
    if not rows:
        print("error: no rows in the given sweep file(s)", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    n_err = sum(1 for r in rows if r.get("error"))
    print(f"cells: {len(rows)} from {len(rep['files'])} file(s)"
          + (f"; {n_err} failed (shown as 'x')" if n_err else ""))
    for f in rep["files"]:
        print(f"  {f}")
    if rep["mismatched"]:
        print(f"  note: {rep['mismatched']} rows had a batch column "
              "disagreeing with their folder name; the FOLDER won")
    print_sweep(rows, "ttft_ms", "TTFT (ms)")
    print_sweep(rows, "tpot_ms", "TPOT (ms/token)")

    limits = []
    if args.ttft_sla_ms is not None:
        lt = sla_limits(rows, "ttft_ms", args.ttft_sla_ms)
        print_sla(lt, "ms")
        limits += lt
    if args.tpot_sla_ms is not None:
        lp = sla_limits(rows, "tpot_ms", args.tpot_sla_ms)
        print_sla(lp, "ms/token")
        limits += lp
    if limits:
        write_csv(args.out / "sla_limits.csv", limits,
                  ["metric", "sla", "batch", "max_ok_measured",
                   "max_ok_interp", "crossed"])
    else:
        print("\n(no --ttft-sla-ms / --tpot-sla-ms given: curves are plotted "
              "without a threshold)")

    if not args.no_figures:
        try:
            plot_metric(rows, "ttft_ms", "TTFT (ms)",
                        "TTFT",
                        args.ttft_sla_ms,
                        [r for r in limits if r["metric"] == "ttft_ms"],
                        args.out / "fig_ttft.pdf")
            plot_metric(rows, "tpot_ms", "TPOT (ms/token)",
                        "TPOT",
                        args.tpot_sla_ms,
                        [r for r in limits if r["metric"] == "tpot_ms"],
                        args.out / "fig_tpot.pdf")
        except ImportError:
            print("matplotlib unavailable -- figures skipped", file=sys.stderr)
    print(f"\noutputs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
