#!/usr/bin/env python3
"""E5: does CPU contention become a major overhead as max-in-flight grows?

Compares two (or more) runs — canonically mif=1 vs mif=256 — on the
signatures that separate "CPU/sandbox contention" from "GPU/queue
pressure":

  1. llm / tool / scaffold decomposition (canonical, reused from
     analyze_profiles via e3): mean seconds + per-request shares. Under
     CPU contention the TOOL and SCAFFOLD seconds inflate; under pure
     GPU/queue pressure only LLM inflates.
  2. host resources (resource.ndjson): cpu_util_pct, load_1min vs
     n_cores (>1x cores = runnable backlog = contention), mem_used_gib,
     and the opencode process-tree CPU%.
  3. per-TOOL wall inflation: same tool name across runs, mean/p50
     duration ratio (task tool excluded — nested agent, not a leaf
     tool). The tool binary does the same work regardless of mif, so
     inflation here is scheduling/IO contention, not workload change.

Outputs (--out): cpu_contention.csv (+ per_tool.csv) and
fig_contention.pdf (component seconds + per-tool p50 scatter),
plus a stdout digest with ratios of the LAST run vs the FIRST.

Usage:
  scripts/arm/e5_cpu_contention.py mif1=<run_dir> mif256=<run_dir> \
      [--out <dir>] [--no-figures]

Run-entry resolution is e3's (profiles/ or profiles.jsonl auto-detect,
trace.jsonl main-session filter, resource.ndjson at <root>/ or
<root>/logs/).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

_ARM = Path(__file__).resolve().parent


def _load(name: str, fname: str):
    spec = importlib.util.spec_from_file_location(name, _ARM / fname
                                                  if not fname.startswith("..")
                                                  else _ARM.parent / fname[3:])
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    k = (len(s) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def tool_durations(ap_mod, e0, inp: Path,
                   trace: Path | None) -> dict[str, list[float]]:
    """{tool_name: [duration_s]} across all main-session turns; the task
    tool is excluded (its duration is a nested agent loop)."""
    sessions = ap_mod.load_sessions(inp)
    if trace is not None:
        keep = e0.trace_session_ids(trace)
        sessions = {sid: s for sid, s in sessions.items() if sid in keep}
    out: dict[str, list[float]] = {}
    for s in sessions.values():
        for t in s.turns.values():
            for tc in t.tools:
                if tc.name == "task":
                    continue
                out.setdefault(tc.name, []).append(tc.duration_s)
    return out


def host_resources(entry: str, e3) -> dict[str, float]:
    """Host + opencode-tree resource summary for a run entry. Empty dict
    when resource.ndjson is absent."""
    rpath = e3.resolve_resource_ndjson(entry)
    if rpath is None:
        return {}
    cpu: list[float] = []
    mem: list[float] = []
    load: list[float] = []
    oc_cpu: list[float] = []
    n_cores = None
    with rpath.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = rec.get("host")
            if isinstance(h, dict):
                if isinstance(h.get("cpu_util_pct"), (int, float)):
                    cpu.append(float(h["cpu_util_pct"]))
                if isinstance(h.get("mem_used_bytes"), (int, float)):
                    mem.append(float(h["mem_used_bytes"]) / 2 ** 30)
                if isinstance(h.get("load_1min"), (int, float)):
                    load.append(float(h["load_1min"]))
                if isinstance(h.get("n_cores"), (int, float)):
                    n_cores = float(h["n_cores"])
            for p in rec.get("processes") or []:
                if p.get("name") == "opencode" \
                        and isinstance(p.get("cpu_util_pct"), (int, float)):
                    oc_cpu.append(float(p["cpu_util_pct"]))
    out: dict[str, float] = {}
    for name, vals in (("host_cpu_pct", cpu), ("mem_used_gib", mem),
                       ("load_1min", load), ("opencode_cpu_pct", oc_cpu)):
        if vals:
            out[f"{name}_mean"] = sum(vals) / len(vals)
            out[f"{name}_p90"] = _pct(vals, 0.9)
    if n_cores:
        out["n_cores"] = n_cores
        if load:
            out["load_per_core_p90"] = _pct(load, 0.9) / n_cores
    return out


def fig_contention(e0, comp: dict[str, dict], tools: dict[str, dict],
                   path: Path) -> None:
    """Left: llm/tool/scaffold mean seconds grouped by run. Right:
    per-tool p50 duration, run A (x) vs run B (y), y=x guide — points
    above the line = tool inflation under the higher mif."""
    plt = e0._mpl()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
    runs = list(comp)
    comps = ("llm", "tool", "scaffold")
    colors = {"llm": "C0", "tool": "C2", "scaffold": "0.5"}
    w = 0.8 / max(len(runs), 1)
    for ri, run in enumerate(runs):
        for ci, c in enumerate(comps):
            v = comp[run][c]["mean_s"]
            axL.bar(ci + ri * w - 0.4 + w / 2, v, width=w,
                    color=colors[c], alpha=0.5 + 0.5 * ri / max(len(runs) - 1, 1),
                    edgecolor="black", linewidth=0.5)
            axL.text(ci + ri * w - 0.4 + w / 2, v, f" {v:.1f}",
                     rotation=90, fontsize=7, va="bottom", ha="center")
    axL.set_xticks(range(len(comps)), comps)
    axL.set_ylabel("mean seconds per turn")
    axL.set_title(f"Turn components ({' vs '.join(runs)}; "
                  "darker = later run)")

    if len(runs) >= 2 and tools:
        a, b = runs[0], runs[-1]
        xs, ys, names = [], [], []
        for name, d in tools.items():
            if a in d and b in d:
                xs.append(d[a])
                ys.append(d[b])
                names.append(name)
        if xs:
            axR.scatter(xs, ys, s=20, color="tab:red", zorder=3)
            for x, y, nm in zip(xs, ys, names):
                axR.annotate(nm, (x, y), fontsize=7,
                             textcoords="offset points", xytext=(4, 2))
            lim = max(max(xs), max(ys)) * 1.1
            axR.plot([0, lim], [0, lim], color="grey", lw=0.8, ls="--",
                     label="y = x (no inflation)")
            axR.set_xlim(0, lim)
            axR.set_ylim(0, lim)
            axR.set_xlabel(f"{a} tool p50 (s)")
            axR.set_ylabel(f"{b} tool p50 (s)")
            axR.legend(fontsize=8, framealpha=0.7)
    axR.set_title("Per-tool p50 duration: inflation above y=x")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="label=dir run entries (2+)")
    ap.add_argument("--out", type=Path, default=Path("e5_cpu_contention"))
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    if len(args.runs) < 2:
        print("error: need at least 2 runs to compare", file=sys.stderr)
        return 2

    e3 = _load("e3_compare_runs", "e3_compare_runs.py")
    spec = importlib.util.spec_from_file_location(
        "analyze_profiles", _ARM.parent / "analyze_profiles.py")
    assert spec and spec.loader
    ap_mod = importlib.util.module_from_spec(spec)
    sys.modules["analyze_profiles"] = ap_mod
    spec.loader.exec_module(ap_mod)
    e0 = _load("e0_turn_characterization", "e0_turn_characterization.py")

    comp: dict[str, dict] = {}          # run -> {comp: {mean_s, p90_s, ...}}
    tool_p50: dict[str, dict] = {}      # tool -> {run: p50}
    tool_n: dict[str, dict] = {}
    res: dict[str, dict] = {}
    for entry in args.runs:
        label, inp, trace = e3.resolve_run(entry)
        rows = e3.run_decomposition(ap_mod, e0, inp, trace)
        if not rows:
            print(f"warning: no turns in {entry}; skipped", file=sys.stderr)
            continue
        comp[label] = e3.summarize(rows)
        durs = tool_durations(ap_mod, e0, inp, trace)
        for name, vals in durs.items():
            tool_p50.setdefault(name, {})[label] = _pct(vals, 0.5)
            tool_n.setdefault(name, {})[label] = len(vals)
        res[label] = host_resources(entry, e3)
        print(f"{label}: {len(rows)} turns, {len(durs)} tools, "
              f"resource fields {len(res[label])}")
    if len(comp) < 2:
        print("error: fewer than 2 runs with data", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    runs = list(comp)
    a, b = runs[0], runs[-1]

    # ----- main CSV + digest -----
    rows_csv: list[list] = []
    print(f"\n{'metric':<28} " + " ".join(f"{r:>12}" for r in runs)
          + f" {'ratio(' + b + '/' + a + ')':>16}")

    def emit(metric: str, vals: dict[str, float], fmt="{:.3f}"):
        ratio = (vals[b] / vals[a]) if vals.get(a) else float("nan")
        print(f"{metric:<28} "
              + " ".join(fmt.format(vals.get(r, float('nan'))).rjust(12)
                         for r in runs)
              + f" {ratio:>16.2f}")
        rows_csv.append([metric] + [f"{vals.get(r, float('nan')):.4f}"
                                    for r in runs] + [f"{ratio:.4f}"])

    for c in ("llm", "tool", "scaffold"):
        emit(f"{c}_mean_s", {r: comp[r][c]["mean_s"] for r in runs})
    for c in ("llm", "tool", "scaffold"):
        emit(f"{c}_mean_share", {r: comp[r][c]["mean_share"] for r in runs})
    res_keys = sorted({k for d in res.values() for k in d})
    for k in res_keys:
        emit(k, {r: res[r].get(k, float("nan")) for r in runs}, "{:.2f}")

    with (args.out / "cpu_contention.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric"] + runs + [f"ratio_{b}_over_{a}"])
        w.writerows(rows_csv)

    # ----- per-tool CSV -----
    with (args.out / "per_tool.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool"] + [f"{r}_p50_s" for r in runs]
                   + [f"{r}_n" for r in runs] + ["p50_ratio"])
        print(f"\nper-tool p50 duration (s):")
        for name in sorted(tool_p50, key=lambda n: -max(
                tool_n[n].get(r, 0) for r in runs)):
            d = tool_p50[name]
            ratio = (d[b] / d[a]) if a in d and b in d and d[a] else \
                float("nan")
            w.writerow([name]
                       + [f"{d.get(r, float('nan')):.4f}" for r in runs]
                       + [tool_n[name].get(r, 0) for r in runs]
                       + [f"{ratio:.4f}"])
            print(f"  {name:<16} "
                  + " ".join(f"{d.get(r, float('nan')):>8.2f}" for r in runs)
                  + f"  ratio {ratio:.2f}")

    # ----- verdict hint -----
    t_ratio = (comp[b]["tool"]["mean_s"] / comp[a]["tool"]["mean_s"]) \
        if comp[a]["tool"]["mean_s"] else float("nan")
    s_ratio = (comp[b]["scaffold"]["mean_s"] / comp[a]["scaffold"]["mean_s"]) \
        if comp[a]["scaffold"]["mean_s"] else float("nan")
    l_ratio = (comp[b]["llm"]["mean_s"] / comp[a]["llm"]["mean_s"]) \
        if comp[a]["llm"]["mean_s"] else float("nan")
    print(f"\ncontention signature: tool x{t_ratio:.2f}, "
          f"scaffold x{s_ratio:.2f} vs llm x{l_ratio:.2f} "
          f"({b} over {a})")
    print("  interpretation: tool/scaffold inflating much more than llm "
          "=> CPU-side contention; llm dominating the inflation => "
          "GPU/queue pressure, CPU not the bottleneck")

    if not args.no_figures:
        try:
            fig_contention(e0, comp, tool_p50,
                           args.out / "fig_contention.pdf")
        except ImportError:
            print("matplotlib unavailable -- figure skipped",
                  file=sys.stderr)
    print(f"\noutputs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
