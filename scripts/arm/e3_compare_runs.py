#!/usr/bin/env python3
"""E3: cross-run comparison of the canonical turn decomposition.

Give it several runs; for each it reuses analyze_profiles.py's CANONICAL
per-turn decomposition (wall = turn.start -> next turn.start; llm =
dynamo queue+prefill+decode with anchored/stream fallback; scaffold =
wall - llm - tool) and emits:

  comparison.csv           one row per (run, component in llm/tool/
                           scaffold): mean_s, p90_s of the per-turn
                           seconds + mean per-request share.
  fig_share_by_run.pdf     per-request latency-component share
                           distributions (the analyze_profiles violin
                           view) side by side per run, one violin per
                           component, with the MEAN drawn and labeled.

Run entries (positional, repeatable):
  <dir>            a run/workspace dir; profiles input auto-detected as
                   <dir>/profiles/ (session-per-file dir), else
                   <dir>/profiles.jsonl; <dir>/trace.jsonl (when
                   present) filters to MAIN sessions. Label = dir name.
  <label>=<dir>    same, with an explicit label.

Usage:
  scripts/arm/e3_compare_runs.py runA runB lmcache=/path/to/runC \
      [--out <dir>]
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from pathlib import Path

_ARM = Path(__file__).resolve().parent
_AP_PATH = _ARM.parent / "analyze_profiles.py"
_E0_PATH = _ARM / "e0_turn_characterization.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


COMPONENTS = ("llm", "tool", "scaffold")


def resolve_run(entry: str) -> tuple[str, Path, Path | None]:
    """(label, profiles_input, trace_or_None) from a CLI run entry."""
    label, _, rest = entry.partition("=")
    if rest:
        root = Path(rest)
    else:
        root = Path(entry)
        label = root.name
    if (root / "profiles").is_dir():
        inp = root / "profiles"
    elif (root / "profiles.jsonl").is_file():
        inp = root / "profiles.jsonl"
    elif root.is_dir() or root.is_file():
        inp = root                      # already a profiles dir/file
    else:
        raise FileNotFoundError(f"run not found: {entry}")
    trace = root / "trace.jsonl"
    return label, inp, trace if trace.is_file() else None


def run_decomposition(ap_mod, e0, inp: Path,
                      trace: Path | None) -> list[tuple[float, float, float, float]]:
    """[(wall, llm, tool, scaffold)] per turn, canonical values, main
    sessions only when a trace is given."""
    sessions = ap_mod.load_sessions(inp)
    if trace is not None:
        keep = e0.trace_session_ids(trace)
        sessions = {sid: s for sid, s in sessions.items() if sid in keep}
    rows = ap_mod._collect_turn_decomposition(sessions)
    # row = (sid, step, wall, lw_stream, tool, post_stream, llm_canon,
    #        scaffold); canonical positions 2/6/4/7.
    return [(r[2], r[6], r[4], r[7]) for r in rows]


def _percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def summarize(rows) -> dict[str, dict[str, float]]:
    """{component: {mean_s, p90_s, mean_share}} from decomposition rows.
    Shares are per-turn ratios (component / wall), zero-wall turns
    skipped for shares."""
    out: dict[str, dict[str, float]] = {}
    for i, comp in enumerate(COMPONENTS, start=1):
        secs = [r[i] for r in rows]
        shares = [r[i] / r[0] for r in rows if r[0] > 0]
        out[comp] = {
            "mean_s": sum(secs) / len(secs) if secs else 0.0,
            "p90_s": _percentile(secs, 90) if secs else 0.0,
            "mean_share": sum(shares) / len(shares) if shares else 0.0,
        }
    return out


def fig_share_by_run(e0, per_run: dict[str, list], path: Path) -> None:
    """Violin of per-request component shares, grouped by run, mean drawn
    and labeled per violin (the analyze_profiles latency_share_violin
    view extended across runs)."""
    plt = e0._mpl()
    runs = list(per_run)
    colors = {"llm": "C0", "tool": "C2", "scaffold": "0.5"}
    fig, ax = plt.subplots(figsize=(max(8, 2.6 * len(runs)), 5))
    pos = 0
    ticks, tick_labels = [], []
    for run in runs:
        rows = per_run[run]
        group_center = pos + 1
        for i, comp in enumerate(COMPONENTS, start=1):
            shares = [r[i] / r[0] for r in rows if r[0] > 0]
            if not shares:
                pos += 1
                continue
            if len(set(shares)) > 1:
                parts = ax.violinplot([shares], positions=[pos],
                                      showmedians=True, showextrema=False,
                                      widths=0.8)
                for b in parts["bodies"]:
                    b.set_facecolor(colors[comp])
                    b.set_alpha(0.6)
                parts["cmedians"].set_color("black")
            else:
                ax.scatter([pos], shares[:1], color="black", s=12)
            # black vertical min-max line through the violin
            ax.vlines(pos, min(shares), max(shares), color="black",
                      lw=0.9, zorder=2)
            mean = sum(shares) / len(shares)
            ax.scatter([pos], [mean], color="black", marker="o", s=8,
                       zorder=3)
            ax.text(pos, mean, f" {mean:.2f}", color="black", fontsize=7,
                    va="bottom", ha="left", zorder=4)
            pos += 1
        ticks.append(group_center)
        tick_labels.append(run)
        pos += 1                              # gap between run groups
    ax.set_xticks(ticks, tick_labels)
    ax.set_ylabel("per-request share of wall")
    ax.set_ylim(0, 1.05)
    for comp in COMPONENTS:
        ax.plot([], [], color=colors[comp], lw=6, alpha=0.6, label=comp)
    ax.legend(fontsize=8, framealpha=0.7, ncol=3, loc="upper right")
    ax.set_title("Per-request latency-component share by run "
                 "(llm / tool / scaffold)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+",
                    help="run dirs (or label=dir); see module docstring")
    ap.add_argument("--out", type=Path, default=Path("e3_compare"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    ap_mod = _load("analyze_profiles", _AP_PATH)
    e0 = _load("e0_turn_characterization", _E0_PATH)

    per_run: dict[str, list] = {}
    for entry in args.runs:
        try:
            label, inp, trace = resolve_run(entry)
        except FileNotFoundError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        rows = run_decomposition(ap_mod, e0, inp, trace)
        if not rows:
            print(f"warning: no turns in {entry}; skipped", file=sys.stderr)
            continue
        per_run[label] = rows
        print(f"{label}: {len(rows)} turns "
              f"({'trace-filtered' if trace else 'NO trace filter'})")
    if not per_run:
        print("error: no runs with data", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / "comparison.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "component", "mean_s", "p90_s", "mean_share",
                    "n_turns"])
        for run, rows in per_run.items():
            summ = summarize(rows)
            for comp in COMPONENTS:
                s = summ[comp]
                w.writerow([run, comp, f"{s['mean_s']:.4f}",
                            f"{s['p90_s']:.4f}", f"{s['mean_share']:.4f}",
                            len(rows)])

    # stdout mirror
    print(f"\n{'run':<20} {'component':<10} {'mean_s':>10} {'p90_s':>10} "
          f"{'mean_share':>11}")
    for run, rows in per_run.items():
        summ = summarize(rows)
        for comp in COMPONENTS:
            s = summ[comp]
            print(f"{run:<20} {comp:<10} {s['mean_s']:>10.3f} "
                  f"{s['p90_s']:>10.3f} {s['mean_share']:>11.3f}")

    if not args.no_figures:
        try:
            fig_share_by_run(e0, per_run, args.out / "fig_share_by_run.pdf")
        except ImportError:
            print("matplotlib unavailable -- figure skipped",
                  file=sys.stderr)
    print(f"\noutputs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
