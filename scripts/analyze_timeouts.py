#!/usr/bin/env python3
"""Compare WHY tasks time out across N testbed runs (e.g. sequential vs
concurrency_32), using only trace.jsonl.

Motivation: a run can show a wildly different timeout rate (e.g. 2/100
sequential vs 94/100 at max-in-flight=32) and you need to know whether
the stalls are CLIENT-side (an opencode tool hung waiting for a human
reply that never comes -- permission / external_directory / question)
or SERVER-side (the LLM stream is slow because the decode/prefill queue
is saturated under concurrency). Those have opposite fixes.

The smoking gun is in the partial trajectory the runner recovers on
abort (error.stage=="timeout" does a best-effort GET /session/:id/message
so `messages` holds the turns persisted before the wall-clock abort).
We classify each timeout by what was IN FLIGHT at abort:

  tool_pending:<name>        a tool part with state.status in {running,
                             pending} was still open. CLIENT-side stall.
                             Sub-flagged `external` when the tool input
                             references a path outside the workspace
                             (/tmp, /var, $HOME, ...) -- the classic
                             permission/external_directory hang.
  awaiting_llm               last assistant msg has a step-start but no
                             completed text/tool after it (waiting on the
                             model's stream) OR the loop was between steps
                             with no open tool. SERVER-side / throughput.
  no_messages                nothing recovered (list also failed) -- can't
                             attribute; error.partial_messages == 0.

We ALSO compare turn counts (step-start count) and tool-call counts of
timed-out vs successful tasks per run, to tell "stuck early" (few turns)
apart from "slow all the way through" (normal turns, never reached the
end before the cap).

Step 3 (CPU vs GPU utilization) is NOT here -- reuse the existing
`scripts/analyze_session_resources.py` (all-points mode) on each run's
deployment-window resource NDJSON; see the hint printed at the end.

Outputs (under --output):
  timeout_causes.csv      one row per timed-out task: run, instance_id,
                          cause, pending_tool, external, n_turns,
                          n_tool_calls, last_input_hint
  run_status.csv          one row per run: counts per status bucket
  turns_by_status.csv     per run x {success,timeout}: n, turn p50/mean/max,
                          toolcall p50/mean/max
  stdout                  human-readable comparison

Usage:
  scripts/analyze_timeouts.py \\
      --traces results/seq/trace.jsonl results/conc32/trace.jsonl \\
      --labels sequential concurrency_32 \\
      --output results/timeout_cmp
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path


# Path prefixes that mean "outside any workspace" -> permission/external
# suspect. workspace_root (read from config.json) is added dynamically as
# the IN-tree prefix; anything not under it AND matching one of these is
# flagged external.
_EXTERNAL_HINT_PREFIXES = ("/tmp", "/var", "/etc", "/root", "/home", "/usr")


@dataclass
class TaskAttr:
    instance_id: str
    status_bucket: str            # success | timeout | err:<stage> | err:other
    n_turns: int
    n_tool_calls: int
    cause: str | None = None      # only for timeout: tool_pending:<n>|awaiting_llm|no_messages
    pending_tool: str | None = None
    external: bool = False
    last_input_hint: str = ""
    partial_messages: int | None = None


def _status_bucket(rec: dict) -> str:
    if rec.get("success"):
        return "success"
    err = rec.get("error") or {}
    stage = err.get("stage")
    if stage == "timeout":
        return "timeout"
    return f"err:{stage}" if stage else "err:other"


def _input_hint(tool_input) -> str:
    """A short, human-meaningful slice of a tool's input for attribution."""
    if isinstance(tool_input, dict):
        for k in ("filePath", "path", "file", "command", "url", "pattern"):
            v = tool_input.get(k)
            if isinstance(v, str) and v:
                return f"{k}={v[:120]}"
        # fall back to a compact dump
        try:
            return json.dumps(tool_input, sort_keys=True, default=str)[:120]
        except Exception:
            return str(tool_input)[:120]
    if tool_input is None:
        return ""
    return str(tool_input)[:120]


def _looks_external(tool_input, ws_root: str | None) -> bool:
    """True if the tool input references a path clearly outside the
    workspace (a permission/external_directory hang suspect)."""
    if not isinstance(tool_input, dict):
        return False
    for k in ("filePath", "path", "file"):
        v = tool_input.get(k)
        if not isinstance(v, str) or not v.startswith("/"):
            continue
        if ws_root and v.startswith(ws_root):
            return False  # explicitly in-tree
        if any(v.startswith(p) for p in _EXTERNAL_HINT_PREFIXES):
            return True
    return False


def _classify_timeout(messages: list, ws_root: str | None) -> tuple[str, str | None, bool, str]:
    """Return (cause, pending_tool, external, last_input_hint).

    Walk the assistant messages; remember the LAST tool part still open
    (state.status in {running,pending}) -- that's what was in flight when
    the wall-clock abort fired. If none is open, decide between
    awaiting_llm (an assistant step started but produced no completed
    output yet / loop between steps) and no_messages.
    """
    saw_assistant = False
    open_tool: tuple[str, object] | None = None  # (name, input)
    last_open_status = None

    for m in messages or []:
        info = (m or {}).get("info") or {}
        if info.get("role") != "assistant":
            continue
        saw_assistant = True
        for p in (m.get("parts") or []):
            if p.get("type") != "tool":
                continue
            state = p.get("state") or {}
            status = state.get("status")
            if status in ("running", "pending"):
                open_tool = (p.get("tool") or "?", state.get("input"))
                last_open_status = status

    if open_tool is not None:
        name, tin = open_tool
        ext = _looks_external(tin, ws_root)
        cause = f"tool_pending:{name}"
        return cause, name, ext, _input_hint(tin)

    if not (messages):
        return "no_messages", None, False, ""
    if not saw_assistant:
        # messages present but no assistant turn persisted -> never got a
        # model response: server-side / first-token stall.
        return "awaiting_llm", None, False, ""
    return "awaiting_llm", None, False, ""


def _attr_one(rec: dict, ws_root: str | None) -> TaskAttr:
    messages = rec.get("messages") or []
    n_steps = n_assistant = n_tool_calls = 0
    for m in messages:
        info = (m or {}).get("info") or {}
        if info.get("role") != "assistant":
            continue
        n_assistant += 1
        for p in (m.get("parts") or []):
            t = p.get("type")
            if t == "step-start":
                n_steps += 1
            elif t == "tool":
                n_tool_calls += 1

    bucket = _status_bucket(rec)
    attr = TaskAttr(
        instance_id=rec.get("instance_id") or "?",
        status_bucket=bucket,
        n_turns=n_steps if n_steps else n_assistant,
        n_tool_calls=n_tool_calls,
    )
    if bucket == "timeout":
        cause, ptool, ext, hint = _classify_timeout(messages, ws_root)
        attr.cause = cause
        attr.pending_tool = ptool
        attr.external = ext
        attr.last_input_hint = hint
        attr.partial_messages = ((rec.get("error") or {}).get("partial_messages"))
    return attr


def _load_run(trace_path: Path) -> tuple[list[TaskAttr], str | None]:
    ws_root = None
    cfg_path = trace_path.parent / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text())
            ws_root = ((cfg.get("config") or {}).get("workspace_root")) or None
        except Exception:
            ws_root = None
    attrs: list[TaskAttr] = []
    with trace_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            attrs.append(_attr_one(json.loads(line), ws_root))
    return attrs, ws_root


def _q(values: list[float], p: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    # nearest-rank-ish on a 0..100 index
    idx = min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))
    return s[idx]


def _fmt(x) -> str:
    return "-" if x is None else (f"{x:.1f}" if isinstance(x, float) else str(x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", nargs="+", required=True,
                    help="trace.jsonl paths (one per run)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="labels for each run (default: parent dir name)")
    ap.add_argument("--output", required=True, help="output directory")
    args = ap.parse_args()

    traces = [Path(t) for t in args.traces]
    for t in traces:
        if not t.exists():
            print(f"trace not found: {t}", file=sys.stderr)
            return 1
    if args.labels:
        if len(args.labels) != len(traces):
            print("--labels count must match --traces count", file=sys.stderr)
            return 2
        labels = args.labels
    else:
        labels = [t.parent.name for t in traces]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    runs: dict[str, list[TaskAttr]] = {}
    ws_roots: dict[str, str | None] = {}
    for label, t in zip(labels, traces):
        attrs, ws = _load_run(t)
        runs[label] = attrs
        ws_roots[label] = ws

    # ---------- run_status.csv + stdout status table ----------
    all_buckets: list[str] = []
    for attrs in runs.values():
        for a in attrs:
            if a.status_bucket not in all_buckets:
                all_buckets.append(a.status_bucket)
    # stable, readable order
    order = ["success", "timeout"] + sorted(b for b in all_buckets
                                            if b not in ("success", "timeout"))

    with (out / "run_status.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "total"] + order)
        for label, attrs in runs.items():
            c = Counter(a.status_bucket for a in attrs)
            w.writerow([label, len(attrs)] + [c.get(b, 0) for b in order])

    print("=" * 70)
    print("STATUS BREAKDOWN")
    print("=" * 70)
    hdr = f"{'run':<18}{'total':>7}" + "".join(f"{b:>14}" for b in order)
    print(hdr)
    for label, attrs in runs.items():
        c = Counter(a.status_bucket for a in attrs)
        row = f"{label:<18}{len(attrs):>7}" + "".join(f"{c.get(b,0):>14}" for b in order)
        print(row)
        n = len(attrs) or 1
        to = c.get("timeout", 0)
        print(f"{'  timeout rate':<18}{'':>7}{to/n*100:>13.0f}%")

    # ---------- timeout_causes.csv + cause tally ----------
    with (out / "timeout_causes.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "instance_id", "cause", "pending_tool", "external",
                    "n_turns", "n_tool_calls", "partial_messages", "last_input_hint"])
        for label, attrs in runs.items():
            for a in attrs:
                if a.status_bucket != "timeout":
                    continue
                w.writerow([label, a.instance_id, a.cause, a.pending_tool or "",
                            int(a.external), a.n_turns, a.n_tool_calls,
                            _fmt(a.partial_messages), a.last_input_hint])

    print()
    print("=" * 70)
    print("TIMEOUT CAUSE ATTRIBUTION  (what was in flight at the wall-clock abort)")
    print("=" * 70)
    for label, attrs in runs.items():
        tos = [a for a in attrs if a.status_bucket == "timeout"]
        print(f"\n[{label}]  {len(tos)} timeouts   (workspace_root={ws_roots[label]})")
        if not tos:
            continue
        cause_c = Counter(a.cause for a in tos)
        for cause, n in cause_c.most_common():
            print(f"    {n:>4}  {cause}")
        ext = sum(1 for a in tos if a.external)
        if ext:
            print(f"    -> {ext} of the tool_pending stalls reference an EXTERNAL "
                  f"path (permission/external_directory suspect)")
        # top pending tools w/ a representative input
        tool_c = Counter(a.pending_tool for a in tos if a.pending_tool)
        if tool_c:
            print("    pending tool breakdown:")
            for tool, n in tool_c.most_common():
                sample = next((a.last_input_hint for a in tos
                               if a.pending_tool == tool and a.last_input_hint), "")
                print(f"        {n:>4}  {tool:<10} e.g. {sample}")

    # ---------- turns_by_status.csv + comparison ----------
    with (out / "turns_by_status.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "status", "n", "turns_p50", "turns_mean", "turns_max",
                    "toolcalls_p50", "toolcalls_mean", "toolcalls_max"])
        for label, attrs in runs.items():
            for status in ("success", "timeout"):
                grp = [a for a in attrs if a.status_bucket == status]
                if not grp:
                    w.writerow([label, status, 0, "", "", "", "", "", ""])
                    continue
                turns = [a.n_turns for a in grp]
                tcs = [a.n_tool_calls for a in grp]
                w.writerow([
                    label, status, len(grp),
                    _q(turns, 50), round(statistics.mean(turns), 2), max(turns),
                    _q(tcs, 50), round(statistics.mean(tcs), 2), max(tcs),
                ])

    print()
    print("=" * 70)
    print("TURNS / TOOL-CALLS  (timed-out vs successful: stuck-early vs slow-throughout)")
    print("=" * 70)
    print(f"{'run':<18}{'status':<9}{'n':>5}{'turns p50':>11}{'turns mean':>12}"
          f"{'turns max':>11}{'tools mean':>12}")
    for label, attrs in runs.items():
        for status in ("success", "timeout"):
            grp = [a for a in attrs if a.status_bucket == status]
            if not grp:
                continue
            turns = [a.n_turns for a in grp]
            tcs = [a.n_tool_calls for a in grp]
            print(f"{label:<18}{status:<9}{len(grp):>5}"
                  f"{_fmt(_q(turns,50)):>11}{statistics.mean(turns):>12.1f}"
                  f"{max(turns):>11}{statistics.mean(tcs):>12.1f}")

    print()
    print("=" * 70)
    print("STEP 3 (CPU vs GPU utilization) -- run separately per deployment window:")
    print("  scripts/analyze_session_resources.py --resource <run>/logs/resource.ndjson")
    print("  (omit --profile for all-points mode = whole-window average; compare")
    print("   host.cpu_util_pct + process.opencode.cpu_util_pct against")
    print("   gpu<N>.DCGM_FI_DEV_GPU_UTIL between the two deployments)")
    print("=" * 70)
    print(f"\nwrote: {out}/run_status.csv, timeout_causes.csv, turns_by_status.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
