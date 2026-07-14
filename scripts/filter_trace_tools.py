#!/usr/bin/env python3
"""Filter tool calls out of an opencode run's trace.jsonl.

Two extractions (both run by default; narrow with --only):

  task  every call of the `task` tool (nested-agent spawn): which
        instance_id fired it, plus that call's input (the sub-agent
        prompt/description) and output (the sub-agent's final text).
        Task turns are excluded from turn-decomposition stats
        (analyze_profiles.py), so this is the companion view for
        studying them separately.

  bash  every `bash` tool call's output, annotated with heuristic
        NONDETERMINISM flags -- regex hits for content that can differ
        across byte-identical reruns (timestamps, durations, hex
        addresses, PIDs, /tmp paths, uuids, session-dir names, ...).
        Purpose: audit which bash outputs could inject run-to-run
        variance back into the agent context (a nondeterministic tool
        output changes the next prompt, which can fork the whole
        trajectory even under greedy decoding + pinned seeds).

Input: trace.jsonl TaskRecords whose `messages` field is the raw
opencode list_messages dump: [{ "info": {...}, "parts": [{... "type":
"tool", "tool": <name>, "state": {status,input,output,metadata,time}}]}].
Records with empty/partial messages (error.stage upstream of `list`)
simply contribute nothing.

Outputs (under --out, default <run>/tool_filter):
  task_calls.jsonl   one line per task call:
                     {instance_id, session_id, message_id, call_id,
                      status, input, output}
  bash_calls.jsonl   one line per bash call:
                     {instance_id, call_id, status, command,
                      description, exit, output, nd_flags}
  stdout             summary: per-instance task-call counts, bash
                     nondeterminism-flag histogram, worst offenders.

Usage:
  scripts/filter_trace_tools.py --run results/run1 [--only task|bash]
      [--out results/run1/tool_filter] [--preview-chars 200]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

TASK_TOOL = "task"
BASH_TOOL = "bash"

# Heuristic nondeterminism detectors for bash OUTPUT text. Each hit means
# "this output could differ on a rerun even if the command's work is
# byte-identical". Ordered roughly by how often they burn reproducibility.
ND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # wall-clock: ISO dates/times and unix-epoch-looking numbers
    ("datetime", re.compile(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|\b\d{10}(?:\.\d+)?\b")),
    # timing/durations: "in 0.53s", "took 2.1 seconds", pytest's
    # "== 3 passed in 0.42s ==" summary line
    ("duration", re.compile(
        r"\b(?:in|took|elapsed)\s+\d+(?:\.\d+)?\s*(?:ms|s|sec|seconds)\b"
        r"|=+\s.*\bin \d+\.\d+s\s=+", re.IGNORECASE)),
    # CPython object reprs / memory addresses
    ("hex_address", re.compile(r"0x[0-9a-fA-F]{6,}")),
    ("pid", re.compile(r"\bpid[=:\s]+\d+", re.IGNORECASE)),
    ("uuid", re.compile(
        r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")),
    # tempfile artifacts (mkstemp suffixes, /tmp scratch)
    ("tmp_path", re.compile(r"/tmp/[\w.\-/]{4,}")),
    # workspace dirs carry the uuid suffix in non-reset mode
    ("session_dir", re.compile(r"session-[\w.\-]{4,}")),
    # unordered-collection smells: SET reprs are hash-ordered (dicts are
    # insertion-ordered since 3.7, and JSON objects are deterministic),
    # so only flag colon-free brace groups -- `{'b', 'a'}` yes,
    # `{"key": "val"}` (JSON / dict) no.
    ("set_repr", re.compile(r"(?<!\{)\{[^{}:]*['\"0-9][^{}:]*\}(?!\})")),
    ("random_seed_word", re.compile(r"\brandom\b|\bseed\b", re.IGNORECASE)),
]


def detect_nondeterminism(text: str) -> list[str]:
    """Names of ND_PATTERNS matching anywhere in `text` (order preserved)."""
    return [name for name, pat in ND_PATTERNS if pat.search(text)]


def load_trace(trace_path: Path) -> list[dict[str, Any]]:
    out = []
    with trace_path.open(encoding="utf-8", errors="replace") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"warning: skipping malformed trace line {lineno} "
                      f"({e})", file=sys.stderr)
    return out


def iter_tool_parts(record: dict[str, Any]) -> Iterator[tuple[dict, dict]]:
    """Yield (message_info, tool_part) for every tool part in a TaskRecord's
    raw message dump. Tolerates records whose messages are [] (upstream
    error stages) or whose parts are missing."""
    for msg in record.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        info = msg.get("info") or {}
        for part in msg.get("parts") or []:
            if isinstance(part, dict) and part.get("type") == "tool":
                yield info, part


def _state(part: dict[str, Any]) -> dict[str, Any]:
    return part.get("state") or {}


def extract_task_calls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        for info, part in iter_tool_parts(rec):
            if part.get("tool") != TASK_TOOL:
                continue
            st = _state(part)
            out.append({
                "instance_id": rec.get("instance_id"),
                "session_id": rec.get("session_id"),
                "message_id": part.get("messageID") or info.get("id"),
                "call_id": part.get("callID"),
                "status": st.get("status"),
                "input": st.get("input"),
                "output": st.get("output"),
            })
    return out


def extract_bash_calls(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for rec in records:
        for info, part in iter_tool_parts(rec):
            if part.get("tool") != BASH_TOOL:
                continue
            st = _state(part)
            inp = st.get("input") or {}
            output = st.get("output") or ""
            meta = st.get("metadata") or {}
            out.append({
                "instance_id": rec.get("instance_id"),
                "call_id": part.get("callID"),
                "status": st.get("status"),
                "command": inp.get("command"),
                "description": inp.get("description"),
                "exit": meta.get("exit"),
                "output": output,
                "nd_flags": detect_nondeterminism(str(output)),
            })
    return out


def _preview(text: Any, n: int) -> str:
    # Truncate BEFORE escaping so the cut can't land between the '\' and
    # 'n' of an escaped newline (escaping may push length slightly past n
    # for newline-heavy text -- acceptable for a stdout preview).
    s = str(text)
    if len(s) > n:
        s = s[: n - 1] + "…"
    return s.replace("\n", "\\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {path} ({len(rows)} rows)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, default=None,
                    help="Run directory containing trace.jsonl")
    ap.add_argument("--trace", type=Path, default=None,
                    help="Explicit trace.jsonl path (overrides --run)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: <run>/tool_filter)")
    ap.add_argument("--only", choices=[TASK_TOOL, BASH_TOOL], default=None,
                    help="Extract just one of the two views")
    ap.add_argument("--preview-chars", type=int, default=200,
                    help="Stdout preview truncation (files are never "
                         "truncated). Default 200.")
    args = ap.parse_args(argv)

    if args.trace is None and args.run is None:
        ap.error("pass --run <dir> or --trace <trace.jsonl>")
    trace_path = args.trace or (args.run / "trace.jsonl")
    if not trace_path.is_file():
        print(f"trace not found: {trace_path}", file=sys.stderr)
        return 2
    out_dir = args.out or (trace_path.parent / "tool_filter")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_trace(trace_path)
    print(f"{trace_path}: {len(records)} task records")

    if args.only in (None, TASK_TOOL):
        task_calls = extract_task_calls(records)
        write_jsonl(out_dir / "task_calls.jsonl", task_calls)
        by_inst: dict[str, int] = Counter(c["instance_id"] for c in task_calls)
        print(f"\n=== task tool: {len(task_calls)} call(s) across "
              f"{len(by_inst)} instance(s) ===")
        for iid, n in sorted(by_inst.items()):
            print(f"  {iid}: {n} call(s)")
        for c in task_calls:
            print(f"\n  [{c['instance_id']}] call={c['call_id']} "
                  f"status={c['status']}")
            print(f"    input:  {_preview(c['input'], args.preview_chars)}")
            print(f"    output: {_preview(c['output'], args.preview_chars)}")

    if args.only in (None, BASH_TOOL):
        bash_calls = extract_bash_calls(records)
        write_jsonl(out_dir / "bash_calls.jsonl", bash_calls)
        flag_hist: Counter[str] = Counter()
        flagged = 0
        for c in bash_calls:
            if c["nd_flags"]:
                flagged += 1
                flag_hist.update(c["nd_flags"])
        print(f"\n=== bash tool: {len(bash_calls)} call(s), "
              f"{flagged} with nondeterminism flags ===")
        for name, n in flag_hist.most_common():
            print(f"  {name:<18} {n:>4} output(s)")
        # Worst offenders first: most distinct flags, then output length.
        worst = sorted((c for c in bash_calls if c["nd_flags"]),
                       key=lambda c: (-len(c["nd_flags"]), -len(str(c["output"]))))
        for c in worst[:20]:
            print(f"\n  [{c['instance_id']}] exit={c['exit']} "
                  f"flags={','.join(c['nd_flags'])}")
            print(f"    cmd:    {_preview(c['command'], args.preview_chars)}")
            print(f"    output: {_preview(c['output'], args.preview_chars)}")
        if len(worst) > 20:
            print(f"\n  (... {len(worst) - 20} more flagged calls in "
                  f"bash_calls.jsonl)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
