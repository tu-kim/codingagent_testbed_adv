#!/usr/bin/env python3
"""E2: workload-shape statistics of an agent run, from OpenCode profiles.

Answers "what does the agent workflow look like to the serving stack":

  fig1_isl_osl.pdf        ISL (effective input = tokens.input + cache.read,
                          the engine-side prompt length) and OSL (output
                          tokens) histograms side by side, with mean / p50
                          / p90 vertical lines.
  tool_stats.csv          Per tool: invocation count, mean/std of the
                          CURRENT turn's output tokens, mean/std of the
                          tokens ADDED to the NEXT turn's prompt.
                          added(N) = eff(N+1) - eff(N) - kept_output(N),
                          kept_output = output - reasoning (opencode
                          drops prior-turn reasoning from the next
                          prompt) -- i.e. the tool responses + scaffold
                          framing the tool injected.
  fig4_tool_transition.pdf Heat map of tool(N) -> tool(N+1) transitions
                          within a session (first tool of each turn),
                          row-normalized %, toolless turns excluded.

Usage:
  scripts/arm/e2_workload_stats.py \
      --profiles <workspace_root>/profiles --trace results/<run>/trace.jsonl \
      [--out <dir>] [--no-figures]

Same --profiles/--trace contract as e1: --profiles is the per-session
NDJSON DIRECTORY (file stem = session id), --trace keeps main sessions
only. Multi-tool turns: fig2 counts every invocation; fig3/fig4 use the
turn's FIRST tool as its representative label.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_E0_PATH = Path(__file__).resolve().parent / "e0_turn_characterization.py"


def _load_e0():
    spec = importlib.util.spec_from_file_location("e0_turn_characterization",
                                                  _E0_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["e0_turn_characterization"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- stats helpers ----------


def _percentile(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def _turn_tool(t) -> str:
    return str(t.tool_names[0]) if t.tool_names else "(none)"


# ---------- extraction ----------


def isl_osl(turns: list) -> tuple[list[float], list[float]]:
    """(ISL list, OSL list) across turns. ISL = effective_input (engine
    prompt length: tokens.input + cache.read); OSL = output tokens.
    Turns missing the field are skipped per-series."""
    isl = [float(t.effective_input) for t in turns
           if t.effective_input is not None]
    osl = [float(t.output_tokens) for t in turns
           if t.output_tokens is not None]
    return isl, osl


def tool_counts(turns: list) -> Counter:
    """Tool name -> total invocation count (every call in every turn)."""
    c: Counter = Counter()
    for t in turns:
        for name in t.tool_names:
            c[str(name)] += 1
    return c


def tool_token_effects(turns: list) -> dict[str, dict[str, list[float]]]:
    """Per representative tool of turn N: {out: [output tokens of N],
    added: [tokens added to turn N+1's prompt]}. added = eff(N+1) -
    eff(N) - max(0, output - reasoning); negative deltas (compaction)
    are kept -- they are real prompt shrinks. Consecutive (session,
    step) pairs only."""
    by_key = {(t.session_id, t.step): t for t in turns}
    out: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"out": [], "added": []})
    for t in turns:
        tool = _turn_tool(t)
        if t.output_tokens is not None:
            out[tool]["out"].append(float(t.output_tokens))
        nxt = by_key.get((t.session_id, t.step + 1))
        if nxt is None or t.effective_input is None \
                or nxt.effective_input is None or t.output_tokens is None:
            continue
        kept = max(0, t.output_tokens - getattr(t, "reasoning_tokens", 0))
        out[tool]["added"].append(
            float(nxt.effective_input - t.effective_input - kept))
    return dict(out)


def tool_transitions(turns: list) -> dict[tuple[str, str], int]:
    """(tool(N), tool(N+1)) -> count within each session. The session's
    last turn transitions to nothing and is not counted as a pair; a
    next turn with no tool lands in the "(none)" column."""
    by_sess: dict[str, list] = defaultdict(list)
    for t in turns:
        by_sess[t.session_id].append(t)
    trans: dict[tuple[str, str], int] = Counter()
    for sess_turns in by_sess.values():
        sess_turns.sort(key=lambda t: t.step)
        for cur, nxt in zip(sess_turns, sess_turns[1:]):
            trans[(_turn_tool(cur), _turn_tool(nxt))] += 1
    return dict(trans)


# ---------- figures ----------


def fig_isl_osl(e0, isl: list[float], osl: list[float], path: Path) -> None:
    plt = e0._mpl()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, vals, name, color in ((axes[0], isl, "ISL", "tab:blue"),
                                  (axes[1], osl, "OSL", "tab:green")):
        if not vals:
            ax.text(0.5, 0.5, f"no {name} data", transform=ax.transAxes,
                    ha="center", va="center", color="grey")
            continue
        ax.hist(vals, bins=50, color=color, alpha=0.75)
        mean = sum(vals) / len(vals)
        for v, lab, ls, c in ((mean, "mean", "--", "tab:red"),
                              (_percentile(vals, 50), "p50", ":",
                               "tab:purple"),
                              (_percentile(vals, 90), "p90", "-.",
                               "tab:orange"),
                              (max(vals), "max", "-", "tab:brown")):
            ax.axvline(v, color=c, ls=ls, lw=1.2,
                       label=f"{lab} {v:,.0f}")
        ax.set_xlabel(f"{name} (tokens)")
        ax.set_ylabel("turns")
        ax.set_title(f"{name} distribution (n={len(vals)})")
        ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_tool_stats_csv(counts: Counter, effects: dict, path: Path) -> None:
    """tool_stats.csv: tool, invocations, output_mean/std (turn N),
    added_mean/std (tokens added to turn N+1's prompt), sample sizes.

    NEGATIVE added values (next prompt SMALLER than expected -- almost
    always compaction shrinking the history) are EXCLUDED from the added
    mean/std; n_added_neg reports how many were dropped per tool."""
    import csv
    tools = sorted(set(counts) | set(effects),
                   key=lambda k: -counts.get(k, 0))
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tool", "invocations",
                    "output_tokens_mean", "output_tokens_std", "n_out",
                    "added_next_prompt_mean", "added_next_prompt_std",
                    "n_added", "n_added_neg"])
        for tool in tools:
            o = effects.get(tool, {}).get("out", [])
            a_all = effects.get(tool, {}).get("added", [])
            a = [v for v in a_all if v >= 0]
            n_neg = len(a_all) - len(a)
            om, os_ = _mean_std(o) if o else (0.0, 0.0)
            am, as_ = _mean_std(a) if a else (0.0, 0.0)
            w.writerow([tool, counts.get(tool, 0),
                        f"{om:.1f}", f"{os_:.1f}", len(o),
                        f"{am:.1f}", f"{as_:.1f}", len(a), n_neg])


def fig_tool_transition(e0, trans: dict, path: Path) -> None:
    """Row-normalized heat map of tool(N) -> tool(N+1), % only.
    Toolless "(none)" turns are excluded entirely."""
    plt = e0._mpl()
    trans = {k: v for k, v in trans.items()
             if k[0] != "(none)" and k[1] != "(none)"}
    tools = sorted({k[0] for k in trans} | {k[1] for k in trans})
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(tools) + 2),
                                    max(5, 0.8 * len(tools) + 1)))
    if tools:
        n = len(tools)
        idx = {t: i for i, t in enumerate(tools)}
        mat = [[0.0] * n for _ in range(n)]
        for (a, b), c in trans.items():
            mat[idx[a]][idx[b]] = float(c)
        row_tot = [sum(row) or 1.0 for row in mat]
        norm = [[mat[i][j] / row_tot[i] for j in range(n)] for i in range(n)]
        im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(n):
            for j in range(n):
                if mat[i][j]:
                    ax.text(j, i, f"{norm[i][j]:.0%}",
                            ha="center", va="center", fontsize=8,
                            color="black" if norm[i][j] < 0.6 else "white")
        ax.set_xticks(range(n), tools, rotation=30, ha="right")
        ax.set_yticks(range(n), tools)
        ax.set_xlabel("next turn's tool")
        ax.set_ylabel("current turn's tool")
        fig.colorbar(im, ax=ax, label="row share")
    else:
        ax.text(0.5, 0.5, "no transitions", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    ax.set_title("Tool transition heat map (row %)")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--trace", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    if not args.profiles.is_dir():
        print(f"error: profiles dir not found: {args.profiles}",
              file=sys.stderr)
        return 2
    e0 = _load_e0()
    ats = e0._load_ats()
    turns = ats.load_turns(args.profiles)
    if not turns:
        print("error: no turns parsed", file=sys.stderr)
        return 2
    main_ids = e0.trace_session_ids(args.trace)
    if not main_ids:
        print("error: no session_id in trace.jsonl", file=sys.stderr)
        return 2
    turns = [t for t in turns if t.session_id in main_ids]
    if not turns:
        print("error: no turns left after trace filter", file=sys.stderr)
        return 2

    out_dir = args.out or (args.profiles / "e2")
    out_dir.mkdir(parents=True, exist_ok=True)

    isl, osl = isl_osl(turns)
    print(f"turns: {len(turns)} across {len({t.session_id for t in turns})} "
          f"sessions")
    for name, vals in (("ISL", isl), ("OSL", osl)):
        if vals:
            print(f"  {name:<4} mean {sum(vals)/len(vals):>10,.0f}  "
                  f"p50 {_percentile(vals, 50):>10,.0f}  "
                  f"p90 {_percentile(vals, 90):>10,.0f}  "
                  f"max {max(vals):>10,.0f}  n={len(vals)}")

    counts = tool_counts(turns)
    print("tool invocation counts:")
    for name, c in counts.most_common():
        print(f"  {name:<16} {c}")

    effects = tool_token_effects(turns)
    print("per-tool token effect (mean +- std; negative added excluded):")
    print(f"  {'tool':<16} {'output@N':>18} {'added->N+1':>18}")
    for tool in sorted(effects, key=lambda k: -len(effects[k]["out"])):
        o, a_all = effects[tool]["out"], effects[tool]["added"]
        a = [v for v in a_all if v >= 0]
        n_neg = len(a_all) - len(a)
        om, os_ = _mean_std(o) if o else (0.0, 0.0)
        am, as_ = _mean_std(a) if a else (0.0, 0.0)
        print(f"  {tool:<16} {om:>9,.0f} +-{os_:>6,.0f} "
              f"{am:>9,.0f} +-{as_:>6,.0f}  "
              f"(n={len(o)}/{len(a)}, {n_neg} neg excluded)")

    trans = tool_transitions(turns)
    write_tool_stats_csv(counts, effects, out_dir / "tool_stats.csv")

    if not args.no_figures:
        try:
            fig_isl_osl(e0, isl, osl, out_dir / "fig1_isl_osl.pdf")
            fig_tool_transition(e0, trans,
                                out_dir / "fig4_tool_transition.pdf")
        except ImportError:
            print("matplotlib unavailable -- figures skipped",
                  file=sys.stderr)
    print(f"outputs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
