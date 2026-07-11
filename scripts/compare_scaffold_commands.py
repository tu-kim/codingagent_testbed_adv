#!/usr/bin/env python3
"""Qualitatively compare the ACTUAL executed commands of an OpenCode run
vs a SWE-agent run on the SAME APPS samples.

Both scaffolds solve the identical deterministic APPS instances against the
identical Dynamo backend; only the scaffold differs. OpenCode finishes far
faster, and the "why" is visible in the command stream: how many actions it
takes, of what kind, at what granularity. This script pulls the per-instance
command sequence out of each trace, matches by instance_id, and prints them
side by side plus an aggregate contrast.

Where the commands live (the two trace shapes are different):
  OpenCode  results/<oc>/trace.jsonl -- each record's `messages` is the raw
            list_messages dump; tool calls are parts with type=="tool",
            carrying `tool` (bash|read|edit|write|...) and
            `state.input`. The executed thing is summarized per tool
            (bash -> the shell command; read/write/edit -> the file; etc.).
  SWE-agent results/<sa>/trace.jsonl -- each record points at `traj_dir`;
            the real actions are in <traj_dir>/<id>.traj under
            `trajectory[].action` (every step is a shell action). The
            command TYPE is the first token of the action.

Comparison notes:
  - Default shows ALL tool calls/actions -- that captures the granularity
    difference (OpenCode's structured read/edit vs SWE-agent's many bash
    round-trips that do the same work). Use --bash-only for an
    apples-to-apples SHELL-command comparison (OpenCode `bash` tool vs every
    SWE-agent action).
  - The SWE-agent command-type is a heuristic (first whitespace token of the
    action's first non-empty line): `python`, `cat`, `str_replace_editor`,
    `submit`, ... Good enough for a qualitative histogram.

Outputs:
  stdout                       aggregate contrast table + tool/command-type
                               histograms + per-instance side-by-side
                               command lists (matched instances first).
  <out>/scaffold_commands.csv  one row per command:
                               {scaffold, instance_id, seq, tool, command}
  <out>/scaffold_summary.csv   one row per (scaffold, instance_id):
                               {n_commands, rtt_s, success} + per-tool counts

Usage:
  scripts/compare_scaffold_commands.py \
      --opencode-run results/apps-opencode \
      --sweagent-run results/sweagent-apps1 \
      [--out results/scaffold_cmp] [--bash-only] \
      [--instance apps-03654] [--preview-chars 160] [--max-instances 20]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

# SWE-agent .traj per-step action field candidates (probed in order).
_ACTION_KEYS = ("action", "command", "thought_action")
# OpenCode tool names whose call is itself a shell command.
_OC_SHELL_TOOLS = {"bash"}


# ---------- trace loading ----------


def load_trace(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "trace.jsonl"
    if not path.is_file():
        print(f"trace not found: {path}", file=sys.stderr)
        return []
    out = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: {path} line {lineno} malformed ({e})",
                      file=sys.stderr)
    return out


# ---------- OpenCode command extraction ----------


def _iter_tool_parts(record: dict[str, Any]) -> Iterator[dict]:
    for msg in record.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        for part in msg.get("parts") or []:
            if isinstance(part, dict) and part.get("type") == "tool":
                yield part


def _oc_summary(tool: str, inp: dict[str, Any]) -> str:
    """Human-readable one-liner for what an OpenCode tool call executed."""
    inp = inp or {}
    if tool == "bash":
        return str(inp.get("command", ""))
    if tool in ("read", "write"):
        return str(inp.get("filePath") or inp.get("path", ""))
    if tool == "edit":
        return str(inp.get("filePath") or inp.get("path", ""))
    if tool in ("task",):
        return str(inp.get("description") or inp.get("prompt", ""))
    if tool in ("glob", "grep"):
        p = inp.get("pattern", "")
        loc = inp.get("path")
        return f"{p}" + (f"  (in {loc})" if loc else "")
    if tool == "list":
        return str(inp.get("path", ""))
    if tool == "webfetch":
        return str(inp.get("url", ""))
    # unknown tool: compact JSON of the input
    try:
        return json.dumps(inp, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(inp)


def extract_opencode(records: list[dict[str, Any]], *, bash_only: bool
                     ) -> dict[str, list[dict[str, Any]]]:
    """instance_id -> ordered list of {seq, tool, command}."""
    per: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        iid = rec.get("instance_id")
        seq = 0
        for part in _iter_tool_parts(rec):
            tool = part.get("tool") or "?"
            if bash_only and tool not in _OC_SHELL_TOOLS:
                continue
            st = part.get("state") or {}
            seq += 1
            per[iid].append({
                "seq": seq,
                "tool": tool,
                "command": _oc_summary(tool, st.get("input") or {}),
            })
    return per


# ---------- SWE-agent command extraction ----------


def _traj_path(traj_dir: Path, instance_id: str) -> Path | None:
    exact = traj_dir / f"{instance_id}.traj"
    if exact.is_file():
        return exact
    hits = sorted(traj_dir.glob("*.traj")) if traj_dir.is_dir() else []
    return hits[0] if hits else None


def _action_of(step: dict[str, Any]) -> str | None:
    for key in _ACTION_KEYS:
        v = step.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _cmd_type(action: str) -> str:
    """First whitespace token of the action's first non-empty line."""
    for line in action.splitlines():
        line = line.strip()
        if line:
            return line.split()[0] if line.split() else "?"
    return "?"


def extract_sweagent(records: list[dict[str, Any]], *, bash_only: bool
                     ) -> dict[str, list[dict[str, Any]]]:
    """instance_id -> ordered list of {seq, tool, command}. `tool` is the
    heuristic command-type; every SWE-agent action is a shell action, so
    --bash-only is a no-op filter here (kept for symmetry)."""
    per: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        iid = rec.get("instance_id")
        traj_dir = rec.get("traj_dir")
        if not traj_dir:
            continue
        path = _traj_path(Path(traj_dir), iid or "")
        if path is None:
            print(f"warning: no .traj for {iid} under {traj_dir}",
                  file=sys.stderr)
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8",
                                             errors="replace"))
        except json.JSONDecodeError as e:
            print(f"warning: {path} not JSON ({e})", file=sys.stderr)
            continue
        seq = 0
        for step in data.get("trajectory") or []:
            if not isinstance(step, dict):
                continue
            action = _action_of(step)
            if action is None:
                continue
            seq += 1
            per[iid].append({
                "seq": seq,
                "tool": _cmd_type(action),
                "command": action,
            })
    return per


# ---------- reporting ----------


def _rtt_success(records: list[dict[str, Any]]) -> dict[str, tuple]:
    return {r.get("instance_id"): (r.get("rtt_s"), r.get("success"))
            for r in records}


def _preview(text: str, n: int) -> str:
    s = str(text)
    if len(s) > n:
        s = s[: n - 1] + "…"
    return s.replace("\n", "\\n")


def _fmt_rtt(v: Any) -> str:
    return f"{v:.1f}s" if isinstance(v, (int, float)) else "-"


def _median(xs: list[float]) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--opencode-run", required=True, type=Path)
    ap.add_argument("--sweagent-run", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir for CSVs (default: no CSV, stdout only)")
    ap.add_argument("--bash-only", action="store_true",
                    help="Compare only shell commands (OpenCode `bash` tool "
                         "vs every SWE-agent action).")
    ap.add_argument("--instance", default=None,
                    help="Restrict the side-by-side to one instance_id.")
    ap.add_argument("--max-instances", type=int, default=20,
                    help="Max instances to print side-by-side (default 20).")
    ap.add_argument("--preview-chars", type=int, default=160)
    args = ap.parse_args(argv)

    oc_recs = load_trace(args.opencode_run)
    sa_recs = load_trace(args.sweagent_run)
    if not oc_recs and not sa_recs:
        print("both traces empty/missing", file=sys.stderr)
        return 2

    oc = extract_opencode(oc_recs, bash_only=args.bash_only)
    sa = extract_sweagent(sa_recs, bash_only=args.bash_only)
    oc_meta = _rtt_success(oc_recs)
    sa_meta = _rtt_success(sa_recs)

    # ----- aggregate contrast -----
    def agg(per: dict[str, list], meta: dict[str, tuple]) -> dict[str, Any]:
        counts = [len(v) for v in per.values()]
        lens = [len(c["command"]) for v in per.values() for c in v]
        rtts = [m[0] for m in meta.values()
                if isinstance(m[0], (int, float))]
        hist: Counter[str] = Counter(
            c["tool"] for v in per.values() for c in v)
        return {
            "instances": len(per),
            "total_cmds": sum(counts),
            "cmds_mean": (sum(counts) / len(counts)) if counts else 0.0,
            "cmds_median": _median([float(c) for c in counts]),
            "cmd_len_mean": (sum(lens) / len(lens)) if lens else 0.0,
            "rtt_mean": (sum(rtts) / len(rtts)) if rtts else float("nan"),
            "hist": hist,
        }

    a_oc, a_sa = agg(oc, oc_meta), agg(sa, sa_meta)
    matched = sorted(set(oc) & set(sa),
                     key=lambda i: (i is None, i))

    mode = "shell commands only" if args.bash_only else "all tool calls"
    print(f"\n=== scaffold command contrast ({mode}) ===")
    print(f"{'':<26}{'opencode':>14}{'sweagent':>14}")
    rows = [
        ("instances", a_oc["instances"], a_sa["instances"], "d"),
        ("matched instances", len(matched), len(matched), "d"),
        ("total commands", a_oc["total_cmds"], a_sa["total_cmds"], "d"),
        ("cmds/instance mean", a_oc["cmds_mean"], a_sa["cmds_mean"], ".1f"),
        ("cmds/instance median", a_oc["cmds_median"], a_sa["cmds_median"],
         ".1f"),
        ("mean cmd length", a_oc["cmd_len_mean"], a_sa["cmd_len_mean"],
         ".0f"),
        ("mean rtt", a_oc["rtt_mean"], a_sa["rtt_mean"], ".1f"),
    ]
    for label, ov, sv, fmt in rows:
        print(f"{label:<26}{ov:>14{fmt}}{sv:>14{fmt}}")

    print("\nopencode tool histogram:")
    for name, n in a_oc["hist"].most_common():
        print(f"  {name:<22} {n:>5}")
    print("sweagent command-type histogram:")
    for name, n in a_sa["hist"].most_common():
        print(f"  {name:<22} {n:>5}")

    # ----- per-instance side by side -----
    to_show = ([args.instance] if args.instance else matched)[:args.max_instances]
    if args.instance and args.instance not in matched:
        print(f"\n(note: {args.instance} not in both traces; "
              f"oc={args.instance in oc} sa={args.instance in sa})")
    for iid in to_show:
        oc_cmds, sa_cmds = oc.get(iid, []), sa.get(iid, [])
        oc_rtt = _fmt_rtt(oc_meta.get(iid, (None,))[0])
        sa_rtt = _fmt_rtt(sa_meta.get(iid, (None,))[0])
        print(f"\n=== {iid} ===")
        print(f"  opencode (rtt {oc_rtt}, {len(oc_cmds)} cmd):")
        for c in oc_cmds:
            print(f"    {c['seq']:>3}. {c['tool']:<10} "
                  f"{_preview(c['command'], args.preview_chars)}")
        print(f"  sweagent (rtt {sa_rtt}, {len(sa_cmds)} cmd):")
        for c in sa_cmds:
            print(f"    {c['seq']:>3}. {c['tool']:<10} "
                  f"{_preview(c['command'], args.preview_chars)}")
    if len(matched) > len(to_show) and not args.instance:
        print(f"\n(... {len(matched) - len(to_show)} more matched "
              f"instances; raise --max-instances or see CSV)")
    only_oc = sorted(set(oc) - set(sa), key=lambda i: (i is None, i))
    only_sa = sorted(set(sa) - set(oc), key=lambda i: (i is None, i))
    if only_oc:
        print(f"\nonly in opencode ({len(only_oc)}): "
              f"{', '.join(str(i) for i in only_oc[:10])}"
              + (" ..." if len(only_oc) > 10 else ""))
    if only_sa:
        print(f"only in sweagent ({len(only_sa)}): "
              f"{', '.join(str(i) for i in only_sa[:10])}"
              + (" ..." if len(only_sa) > 10 else ""))

    # ----- CSVs -----
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        cmds_csv = args.out / "scaffold_commands.csv"
        with cmds_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scaffold", "instance_id", "seq", "tool", "command"])
            for scaffold, per in (("opencode", oc), ("sweagent", sa)):
                for iid, v in per.items():
                    for c in v:
                        w.writerow([scaffold, iid, c["seq"], c["tool"],
                                    c["command"]])
        summ_csv = args.out / "scaffold_summary.csv"
        with summ_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scaffold", "instance_id", "n_commands", "rtt_s",
                        "success"])
            for scaffold, per, meta in (("opencode", oc, oc_meta),
                                        ("sweagent", sa, sa_meta)):
                for iid, v in per.items():
                    rtt, ok = meta.get(iid, (None, None))
                    w.writerow([scaffold, iid, len(v), rtt, ok])
        print(f"\nwrote {cmds_csv}\nwrote {summ_csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
