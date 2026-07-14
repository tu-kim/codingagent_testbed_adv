#!/usr/bin/env python3
"""Flag server-type / long-running background bash tool calls in an
opencode run's trace.jsonl, and report how much tool wall-time they
inject so they can be EXCLUDED from tool-time aggregation.

Motivation
----------
opencode's `bash` tool runs each command synchronously and blocks on the
child's exit code (`opencode/packages/opencode/src/tool/shell.ts` ->
`raceAll([handle.exitCode, abort, timeout])`). A command that launches a
long-running server -- the classic terminalbench case
`cd ./webroot && python3 -m http.server 8080 > /dev/null 2>&1 &` -- either
returns instantly (correctly backgrounded) OR, if the server keeps the
captured stdout/stderr pipe open, or the `&` is missing, blocks until the
tool's own timeout or `--task-timeout-s` fires. When it blocks, that tool
call shows up as the single biggest consumer of "tool time" in the
profile -- but it is a HANG artifact (opencode waiting on a process that
never exits), NOT model/scheduling load. Left in, it inflates the tool
share and corrupts router/scheduling comparisons.

This filter finds those calls by COMMAND TEXT (which lives only in
trace.jsonl -- the profile NDJSON carries the tool NAME but not its
input) plus timing/status, and emits an exclusion list of callIDs.

Detection (a call is flagged if it matches ANY reason)
------------------------------------------------------
  server      command matches a known long-lived server launcher
              (python -m http.server, flask/uvicorn/gunicorn/hypercorn,
               php -S, ruby -run httpd, `serve`, `http-server`, `nc -l`/
               netcat listen, `npm|yarn|pnpm run? (start|dev|serve)`, ...)
  background  a top-level job-control `&` (backgrounded process launch),
              excluding `&&`, `&>`, `>&`, `2>&1`
  longrun     `tail -f`, `sleep <big>`, `watch`, `while true`, `journalctl
              -f`, `docker run` without `-d`, ...
  duration    the call's wall time >= --min-duration-s (catch-all for a
              call that dominated regardless of command shape)

A flagged call is additionally marked is_hang when its status is not
`completed` (i.e. running/error -- aborted mid-flight) OR its duration
>= --min-duration-s. `background`/`server` calls that returned fast and
short are still LISTED (so you can audit them) but is_hang is false, so
--hangs-only will drop them from the exclusion set.

Timing: opencode tool `state.time.{start,end}` are unix MILLISECONDS
(Date.now()); duration_s = (end - start) / 1000. A `running` call (no
`end`, session aborted before the part finished) has no duration -> its
duration_s is null and it is treated as a hang by status.

Input: trace.jsonl TaskRecords whose `messages` is the raw opencode
list_messages dump (same shape filter_trace_tools.py consumes). Records
with empty/partial messages contribute nothing.

Outputs (under --out, default <run>/hanging_tools):
  flagged_tools.jsonl   one line per flagged call:
                        {instance_id, session_id, call_id, tool, command,
                         description, status, exit, duration_s, reasons,
                         is_hang}
  exclude_calls.txt     one callID per line (is_hang calls only unless
                        --all-flagged) -- feed to a downstream aggregator
                        to drop these calls
  stdout                summary: flagged count, reason histogram, total
                        flagged wall vs total bash wall (the share that
                        would be removed from tool-time aggregation),
                        worst offenders.

Usage:
  scripts/filter_hanging_tools.py --run results/tb-run1
  scripts/filter_hanging_tools.py --run results/tb-run1 \\
      --min-duration-s 30 --hangs-only --preview-chars 160

Then exclude from tool-time aggregation by callID, e.g.:
  scripts/analyze_tool_time.py --profile <profiles> --output <out> \\
      --exclude-calls results/tb-run1/hanging_tools/exclude_calls.txt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

BASH_TOOL = "bash"

# --- command heuristics ---------------------------------------------------
# Each entry: (reason-tag, compiled regex). A command hitting a SERVER or
# LONGRUN pattern is flagged with that tag. Patterns are deliberately
# broad-but-anchored on the launcher token; they are HEURISTICS meant to
# catch the dominant hang shapes, not to be exhaustive.
SERVER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bpython[0-9.]*\s+-m\s+http\.server\b"),
    re.compile(r"\bpython[0-9.]*\s+-m\s+SimpleHTTPServer\b"),
    re.compile(r"\b(?:flask\s+run|uvicorn|gunicorn|hypercorn|waitress-serve|daphne)\b"),
    re.compile(r"\bphp\s+-S\b"),
    re.compile(r"\bruby\s+-run\s+-e\s+httpd\b"),
    re.compile(r"\b(?:http-server|live-server|serve)\b"),
    re.compile(r"\bpython[0-9.]*\s+manage\.py\s+runserver\b"),
    re.compile(r"\bnode\s+[^\n|;&]*\bserver(?:\.js|\.mjs|\.ts)?\b"),
    # netcat listen: nc -l / nc -lk / ncat -l / netcat -l
    re.compile(r"\b(?:nc|ncat|netcat)\s+[^\n|;&]*-l\b"),
    # package-manager long-lived dev/serve scripts
    re.compile(r"\b(?:npm|yarn|pnpm|bun)\s+(?:run\s+)?(?:start|dev|serve|preview)\b"),
]

LONGRUN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\btail\s+[^\n|;&]*-[a-zA-Z]*f\b"),  # tail -f / tail -F
    re.compile(r"\b(?:journalctl|kubectl\s+logs|docker\s+logs)\s+[^\n|;&]*-[a-zA-Z]*f\b"),
    re.compile(r"\bwatch\b"),
    re.compile(r"\bwhile\s+(?:true|:|\[\s*1\s*\])\b"),
    re.compile(r"\bfor\s*\(\(\s*;\s*;\s*\)\)"),  # for ((;;))
    # sleep with a large argument (>= 30s) -- small sleeps are benign
    re.compile(r"\bsleep\s+(?:[3-9][0-9]|[1-9][0-9]{2,})\b"),
    # docker run in the foreground (no -d/--detach) tends to block
    re.compile(r"\bdocker\s+run\b(?![^\n|;&]*(?:\s-d\b|--detach\b))"),
]

# top-level job-control `&` (backgrounded launch), NOT `&&`, `&>`, `>&`,
# `2>&1`, `&1`. A trailing `&` (optionally followed by `;`/newline) or a
# `&` before another command is a backgrounding operator.
BACKGROUND_RE = re.compile(r"(?<![&>0-9])&(?![&>0-9])")


def _has_background(cmd: str) -> bool:
    return BACKGROUND_RE.search(cmd) is not None


def classify_command(cmd: str) -> list[str]:
    """Return the list of command-shape reason tags for a bash command
    (subset of {"server", "background", "longrun"}); empty if none match."""
    reasons: list[str] = []
    if any(p.search(cmd) for p in SERVER_PATTERNS):
        reasons.append("server")
    if _has_background(cmd):
        reasons.append("background")
    if any(p.search(cmd) for p in LONGRUN_PATTERNS):
        reasons.append("longrun")
    return reasons


# --- trace ingest ---------------------------------------------------------


def load_records(trace_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with trace_path.open(encoding="utf-8", errors="replace") as fp:
        for lineno, line in enumerate(fp, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed trace line {lineno} ({e})",
                      file=sys.stderr)
    return out


def iter_tool_parts(record: dict[str, Any]) -> Iterator[tuple[dict, dict]]:
    """Yield (message_info, tool_part) for every tool part in a TaskRecord's
    raw message dump. Tolerates [] messages and missing parts."""
    for msg in record.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        info = msg.get("info") or {}
        for part in msg.get("parts") or []:
            if isinstance(part, dict) and part.get("type") == "tool":
                yield info, part


def _duration_s(time_obj: Any) -> float | None:
    """opencode tool state.time.{start,end} are unix ms; return seconds or
    None when either endpoint is missing (e.g. a still-`running` part)."""
    if not isinstance(time_obj, dict):
        return None
    start, end = time_obj.get("start"), time_obj.get("end")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return None
    d = (end - start) / 1000.0
    return d if d >= 0 else None


# --- core -----------------------------------------------------------------


def flag_calls(records: list[dict[str, Any]], min_duration_s: float) -> list[dict[str, Any]]:
    """Return every bash tool call annotated with reasons + is_hang. Only
    calls with at least one reason are returned (unflagged calls are the
    common case and carry no signal here)."""
    out: list[dict[str, Any]] = []
    for rec in records:
        for info, part in iter_tool_parts(rec):
            if part.get("tool") != BASH_TOOL:
                continue
            st = part.get("state") or {}
            inp = st.get("input") or {}
            cmd = inp.get("command")
            if not isinstance(cmd, str):
                continue
            status = st.get("status")
            dur = _duration_s(st.get("time"))
            meta = st.get("metadata") or {}

            reasons = classify_command(cmd)
            long_by_time = dur is not None and dur >= min_duration_s
            if long_by_time:
                reasons.append("duration")
            if not reasons:
                continue

            # A hang = never completed cleanly, or it ran past the duration
            # threshold. `completed` + short = launched-and-returned (listed
            # for audit, but not an exclusion candidate).
            is_hang = (status != "completed") or long_by_time

            out.append({
                "instance_id": rec.get("instance_id"),
                "session_id": rec.get("session_id"),
                "call_id": part.get("callID"),
                "tool": BASH_TOOL,
                "command": cmd,
                "description": inp.get("description"),
                "status": status,
                "exit": meta.get("exit"),
                "duration_s": dur,
                "reasons": reasons,
                "is_hang": is_hang,
            })
    return out


def total_bash_wall_s(records: list[dict[str, Any]]) -> float:
    """Naive sum of all bash tool durations (the denominator the flagged
    wall is compared against). Missing durations contribute 0."""
    total = 0.0
    for rec in records:
        for _info, part in iter_tool_parts(rec):
            if part.get("tool") != BASH_TOOL:
                continue
            st = part.get("state") or {}
            d = _duration_s(st.get("time"))
            if d:
                total += d
    return total


# --- output ---------------------------------------------------------------


def _preview(text: Any, n: int) -> str:
    s = str(text)
    if len(s) > n:
        s = s[: n - 1] + "…"
    return s.replace("\n", "\\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="run directory containing trace.jsonl")
    ap.add_argument("--out", type=Path, default=None,
                    help="output dir (default <run>/hanging_tools)")
    ap.add_argument("--min-duration-s", type=float, default=30.0,
                    help="a bash call at/over this wall time is flagged "
                         "(reason 'duration') and treated as a hang "
                         "(default 30)")
    ap.add_argument("--hangs-only", action="store_true",
                    help="exclude_calls.txt lists only is_hang calls "
                         "(default). Kept for explicitness.")
    ap.add_argument("--all-flagged", action="store_true",
                    help="exclude_calls.txt lists ALL flagged calls, "
                         "including short launched-and-returned ones")
    ap.add_argument("--preview-chars", type=int, default=160)
    args = ap.parse_args(argv)

    trace_path = args.run / "trace.jsonl"
    if not trace_path.is_file():
        print(f"error: {trace_path} not found", file=sys.stderr)
        return 2

    out_dir = args.out or (args.run / "hanging_tools")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(trace_path)
    flagged = flag_calls(records, args.min_duration_s)
    bash_wall = total_bash_wall_s(records)

    write_jsonl(out_dir / "flagged_tools.jsonl", flagged)

    # exclusion set: is_hang calls unless --all-flagged. Dedup callIDs,
    # preserve first-seen order.
    excl_source = flagged if args.all_flagged else [f for f in flagged if f["is_hang"]]
    seen: set[str] = set()
    exclude_ids: list[str] = []
    for f in excl_source:
        cid = f.get("call_id")
        if cid and cid not in seen:
            seen.add(cid)
            exclude_ids.append(cid)
    (out_dir / "exclude_calls.txt").write_text(
        "".join(cid + "\n" for cid in exclude_ids), encoding="utf-8")

    # --- summary ---
    reason_hist: Counter[str] = Counter()
    for f in flagged:
        reason_hist.update(f["reasons"])
    flagged_wall = sum(f["duration_s"] or 0.0 for f in flagged)
    excl_wall = sum(f["duration_s"] or 0.0 for f in excl_source)

    print(f"records:                {len(records)}")
    print(f"flagged bash calls:     {len(flagged)}")
    print(f"  is_hang:              {sum(1 for f in flagged if f['is_hang'])}")
    print(f"excluded callIDs:       {len(exclude_ids)}  "
          f"({'all flagged' if args.all_flagged else 'hangs only'})")
    print()
    print("reason histogram (a call may hit several):")
    for reason, n in reason_hist.most_common():
        print(f"  {reason:12s} {n}")
    print()
    print(f"total bash wall:        {bash_wall:10.1f} s")
    print(f"flagged wall:           {flagged_wall:10.1f} s"
          + (f"  ({100*flagged_wall/bash_wall:.1f}% of bash wall)" if bash_wall > 0 else ""))
    print(f"would-exclude wall:     {excl_wall:10.1f} s"
          + (f"  ({100*excl_wall/bash_wall:.1f}% of bash wall)" if bash_wall > 0 else ""))
    print()

    worst = sorted(flagged, key=lambda f: f["duration_s"] or -1.0, reverse=True)[:10]
    if worst:
        print("worst offenders (by wall):")
        for f in worst:
            dur = f["duration_s"]
            dur_s = f"{dur:8.1f}s" if dur is not None else "     n/a"
            hang = "HANG" if f["is_hang"] else "ok  "
            print(f"  {dur_s} {hang} [{','.join(f['reasons'])}] "
                  f"{f['instance_id']}: {_preview(f['command'], args.preview_chars)}")

    print()
    print(f"wrote {out_dir}/flagged_tools.jsonl")
    print(f"wrote {out_dir}/exclude_calls.txt  ({len(exclude_ids)} callIDs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
