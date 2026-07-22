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


def resolve_logs(entry: str) -> Path | None:
    """<root>/logs (or <root> itself) when it contains vllm-*.log."""
    _, _, root_str = entry.partition("=")
    root = Path(root_str) if root_str else Path(entry)
    for cand in (root / "logs", root):
        if cand.is_dir() and any(cand.glob("vllm-*.log")):
            return cand
    return None


def turn_queue_map(ats, inp: Path, logs: Path | None,
                   ) -> dict[tuple[str, int], float]:
    """{(session_id, step): engine queue seconds} via the profile
    request_id -> SCHED_DELAY join. Empty when logs are absent or the
    profiles input is an aggregated file (load_turns needs the
    per-session dir)."""
    if logs is None or not inp.is_dir():
        return {}
    sched = ats.load_sched(logs)
    if not sched:
        return {}
    out: dict[tuple[str, int], float] = {}
    for t in ats.load_turns(inp):
        rid = getattr(t, "request_id", None)
        if rid and rid in sched:
            out[(t.session_id, t.step)] = sched[rid].total_queue_ms / 1000.0
    return out


def split_queue(ap_rows, qmap: dict[tuple[str, int], float],
                ) -> list[tuple[float, float, float, float, float]]:
    """(wall, llm_compute, queue, tool, scaffold) per turn from the
    analyze_profiles rows (sid, step, wall, lw, tool, po, llm_canon,
    scaffold): the canonical llm (queue+prefill+decode) minus the joined
    engine queue wait, queue capped at llm so the components still
    tile the wall."""
    out = []
    for sid, step, wall, _lw, tool, _po, llm, scaffold in ap_rows:
        q = min(qmap.get((sid, step), 0.0), llm)
        out.append((wall, llm - q, q, tool, scaffold))
    return out


def summarize5(rows5) -> dict[str, dict[str, float]]:
    """{component: {mean_s, mean_share}} over 5-tuples
    (wall, llm, queue, tool, scaffold)."""
    comps = ("llm", "queue", "tool", "scaffold")
    out: dict[str, dict[str, float]] = {}
    for i, c in enumerate(comps, start=1):
        secs = [r[i] for r in rows5]
        shares = [r[i] / r[0] for r in rows5 if r[0] > 0]
        out[c] = {"mean_s": sum(secs) / len(secs) if secs else 0.0,
                  "mean_share": sum(shares) / len(shares) if shares else 0.0}
    return out


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
    comps = ("llm", "queue", "tool", "scaffold")
    colors = {"llm": "C0", "queue": "C1", "tool": "C2", "scaffold": "0.5"}
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
    ats = e0._load_ats()

    comp: dict[str, dict] = {}          # run -> {comp: {mean_s, mean_share}}
    tool_p50: dict[str, dict] = {}      # tool -> {run: p50}
    tool_mean: dict[str, dict] = {}     # tool -> {run: mean}
    tool_n: dict[str, dict] = {}
    res: dict[str, dict] = {}
    for entry in args.runs:
        label, inp, trace = e3.resolve_run(entry)
        sessions = ap_mod.load_sessions(inp)
        if trace is not None:
            keep = e0.trace_session_ids(trace)
            sessions = {sid: s for sid, s in sessions.items() if sid in keep}
        ap_rows = ap_mod._collect_turn_decomposition(sessions)
        if not ap_rows:
            print(f"warning: no turns in {entry}; skipped", file=sys.stderr)
            continue
        qmap = turn_queue_map(ats, inp, resolve_logs(entry))
        rows = split_queue(ap_rows, qmap)
        comp[label] = summarize5(rows)
        n_q = sum(1 for r in ap_rows if (r[0], r[1]) in qmap)
        durs = tool_durations(ap_mod, e0, inp, trace)
        for name, vals in durs.items():
            tool_p50.setdefault(name, {})[label] = _pct(vals, 0.5)
            tool_mean.setdefault(name, {})[label] = sum(vals) / len(vals)
            tool_n.setdefault(name, {})[label] = len(vals)
        res[label] = host_resources(entry, e3)
        print(f"{label}: {len(rows)} turns ({n_q} queue-joined), "
              f"{len(durs)} tools, resource fields {len(res[label])}")
        if not n_q:
            print(f"  [no SCHED_DELAY join for {label}: queue stays "
                  f"inside llm]", file=sys.stderr)
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

    for c in ("llm", "queue", "tool", "scaffold"):
        emit(f"{c}_mean_s", {r: comp[r][c]["mean_s"] for r in runs})
    for c in ("llm", "queue", "tool", "scaffold"):
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
        w.writerow(["tool"]
                   + [f"{r}_p50_s" for r in runs]
                   + [f"{r}_mean_s" for r in runs]
                   + [f"{r}_n" for r in runs]
                   + ["p50_ratio", "mean_ratio"])
        print(f"\nper-tool duration (s):")
        print(f"  {'tool':<16} "
              + " ".join(f"{r + '_p50':>12}" for r in runs)
              + " " + " ".join(f"{r + '_mean':>12}" for r in runs)
              + f" {'p50_r':>7} {'mean_r':>7}")
        for name in sorted(tool_p50, key=lambda n: -max(
                tool_n[n].get(r, 0) for r in runs)):
            d = tool_p50[name]
            dm = tool_mean[name]
            ratio = (d[b] / d[a]) if a in d and b in d and d[a] else \
                float("nan")
            m_ratio = (dm[b] / dm[a]) if a in dm and b in dm and dm[a] \
                else float("nan")
            w.writerow([name]
                       + [f"{d.get(r, float('nan')):.3f}" for r in runs]
                       + [f"{dm.get(r, float('nan')):.3f}" for r in runs]
                       + [tool_n[name].get(r, 0) for r in runs]
                       + [f"{ratio:.3f}", f"{m_ratio:.3f}"])
            print(f"  {name:<16} "
                  + " ".join(f"{d.get(r, float('nan')):>12.3f}"
                             for r in runs)
                  + " " + " ".join(f"{dm.get(r, float('nan')):>12.3f}"
                                   for r in runs)
                  + f" {ratio:>7.3f} {m_ratio:>7.3f}")

    # ----- verdict hint -----
    t_ratio = (comp[b]["tool"]["mean_s"] / comp[a]["tool"]["mean_s"]) \
        if comp[a]["tool"]["mean_s"] else float("nan")
    s_ratio = (comp[b]["scaffold"]["mean_s"] / comp[a]["scaffold"]["mean_s"]) \
        if comp[a]["scaffold"]["mean_s"] else float("nan")
    l_ratio = (comp[b]["llm"]["mean_s"] / comp[a]["llm"]["mean_s"]) \
        if comp[a]["llm"]["mean_s"] else float("nan")
    q_ratio = (comp[b]["queue"]["mean_s"] / comp[a]["queue"]["mean_s"]) \
        if comp[a]["queue"]["mean_s"] else float("nan")
    print(f"\ncontention signature: tool x{t_ratio:.2f}, "
          f"scaffold x{s_ratio:.2f} vs llm(compute) x{l_ratio:.2f}, "
          f"queue x{q_ratio:.2f} ({b} over {a})")
    print("  interpretation: tool/scaffold inflating much more than "
          "llm compute => CPU-side contention; queue dominating => "
          "engine scheduler pressure; llm compute inflating => batch-"
          "induced slowdown on the GPU itself")

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
