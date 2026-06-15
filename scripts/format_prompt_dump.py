#!/usr/bin/env python3
"""Pretty-print the per-turn engine prompts captured by dynamo-prompt-dump.patch.

`DYN_PROMPT_DUMP=1` makes each vLLM worker append one NDJSON record per
request to `<workspace_root>/prompts/prompt-<pid>.jsonl`:

    {"ts", "request_id", "role": prefill|decode, "num_prompt_tokens",
     "prompt_text", ["prompt_token_ids"], ["decode_error"]}

Each record is one LLM round-trip = one agent turn/step, and `prompt_text`
is the EXACT prompt the engine saw that turn (frontend chat-template
applied + tokenized, then detokenized back to text). In the raw file the
text is a JSON string with newlines escaped as `\\n`; this script decodes
it so line breaks render as real newlines and the turn reads like the
prompt actually looked.

What it does:
  * reads every prompt-*.jsonl under a dir (or a single file / list),
  * orders turns by timestamp,
  * de-duplicates the prefill/decode pair for one request_id (both carry
    the same prompt under PD disaggregation) -- keep one per turn,
  * prints each turn with a header + the full prompt_text, OR with
    --delta only the NEW text appended since the previous turn (the
    growing-conversation suffix), trimmed to a clean line boundary.

Examples:
  scripts/format_prompt_dump.py --prompts /tmp/testbed-workspaces/prompts
  scripts/format_prompt_dump.py --prompts prompts/ --delta --out turns.txt
  scripts/format_prompt_dump.py --prompts prompts/ --role both
  scripts/format_prompt_dump.py --prompts prompts/ --session-map reqmap.json

Concurrency caveat: the dump carries NO opencode sessionID (Dynamo never
surfaces `x-session-affinity`), so with multiple concurrent sessions the
turns interleave in one global timeline. `--delta` only makes sense within
a single session: capture with a single/sequential run, or pass a
`--session-map` JSON ({"request_id": "session_id", ...}) so turns are
grouped per session and deltas are computed within each.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _iter_files(prompts: list[Path]) -> list[Path]:
    """Expand each path: a dir -> its prompt-*.jsonl, a file -> itself."""
    files: list[Path] = []
    for p in prompts:
        if p.is_dir():
            files.extend(sorted(p.glob("prompt-*.jsonl")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: not found, skipping: {p}", file=sys.stderr)
    return files


def load_records(prompts: list[Path]) -> tuple[list[dict], int]:
    """Read all NDJSON records. Returns (records, n_bad_lines)."""
    records: list[dict] = []
    bad = 0
    for f in _iter_files(prompts):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    bad += 1
                    continue
                rec["_source"] = f.name
                records.append(rec)
    return records, bad


def dedup_turns(records: list[dict], role: str) -> list[dict]:
    """Collapse the prefill/decode copies of one request_id into one turn.

    role:
      auto    -> one record per request_id, preferring prefill (canonical)
      prefill -> only prefill records
      decode  -> only decode records
      both    -> every record, untouched (a request_id may appear twice)
    """
    if role in ("prefill", "decode"):
        return [r for r in records if r.get("role") == role]
    if role == "both":
        return list(records)
    # auto: keep one per request_id, prefer prefill
    chosen: dict[str, dict] = {}
    for r in records:
        rid = r.get("request_id")
        if rid is None:
            # no id to dedup on -- keep it, key on object identity
            chosen[id(r)] = r
            continue
        prev = chosen.get(rid)
        if prev is None or (prev.get("role") != "prefill" and r.get("role") == "prefill"):
            chosen[rid] = r
    return list(chosen.values())


def _fmt_ts(ts) -> str:
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%f"
        )[:-3] + "Z"
    except (TypeError, ValueError):
        return str(ts)


def _session_of(rec: dict, smap: dict[str, str] | None) -> str:
    if smap is None:
        return "all"
    return smap.get(rec.get("request_id", ""), "unknown")


def _new_suffix(prev: str, cur: str) -> tuple[str, int]:
    """Return (new_text, shared_chars). The new text is cur with its common
    prefix vs prev removed, backed up to the last newline so it starts on a
    clean line boundary."""
    common = os.path.commonprefix([prev, cur])
    cut = len(common)
    # back up to the start of the line the divergence falls on
    nl = cur.rfind("\n", 0, cut)
    cut = nl + 1 if nl >= 0 else 0
    return cur[cut:], cut


def render(
    turns: list[dict],
    smap: dict[str, str] | None,
    *,
    delta: bool,
    max_chars: int,
) -> str:
    # group by session, preserving first-seen order; sort each group by ts
    groups: dict[str, list[dict]] = {}
    for r in turns:
        groups.setdefault(_session_of(r, smap), []).append(r)
    for recs in groups.values():
        recs.sort(key=lambda r: (r.get("ts") or 0))

    out: list[str] = []
    bar = "=" * 80
    for sess, recs in groups.items():
        prev_text = ""
        for i, r in enumerate(recs, start=1):
            text = r.get("prompt_text")
            sess_tag = "" if sess == "all" else f"session={sess}  "
            head = (
                f"{sess_tag}TURN {i}  role={r.get('role')}  "
                f"req={r.get('request_id')}"
            )
            meta = (
                f"  tokens={r.get('num_prompt_tokens')}  "
                f"ts={_fmt_ts(r.get('ts'))}  "
                f"chars={len(text) if isinstance(text, str) else 0}"
                f"  src={r.get('_source')}"
            )
            out.append(bar)
            out.append(head)
            out.append(meta)
            out.append(bar)

            if not isinstance(text, str):
                reason = r.get("decode_error") or (
                    "no prompt_text (DYN_PROMPT_DUMP_TEXT was off?)"
                )
                out.append(f"<no prompt_text: {reason}>")
                out.append("")
                continue

            body = text
            if delta:
                if i == 1:
                    out.append("--- (first turn: full prompt) ---")
                else:
                    body, shared = _new_suffix(prev_text, text)
                    out.append(
                        f"--- (new this turn: {len(body)} chars; "
                        f"{shared} shared with previous turn) ---"
                    )
                prev_text = text

            if max_chars > 0 and len(body) > max_chars:
                body = body[:max_chars] + (
                    f"\n... [truncated {len(body) - max_chars} chars; "
                    f"raise --max-chars or drop it for full text]"
                )
            # Print the decoded string verbatim: json.loads already turned
            # the escaped \n into real newlines, so this renders readably.
            out.append(body)
            out.append("")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--prompts",
        required=True,
        nargs="+",
        type=Path,
        help="prompt-*.jsonl dump dir(s) and/or file(s) "
        "(e.g. <workspace_root>/prompts)",
    )
    ap.add_argument(
        "--role",
        choices=["auto", "prefill", "decode", "both"],
        default="auto",
        help="auto (default): one record per request_id, prefer prefill; "
        "prefill/decode: that role only; both: every record",
    )
    ap.add_argument(
        "--delta",
        action="store_true",
        help="show only the text newly appended since the previous turn "
        "(assumes a single session -- see concurrency caveat)",
    )
    ap.add_argument(
        "--session-map",
        type=Path,
        default=None,
        help='JSON {"request_id": "session_id"} to group turns per session',
    )
    ap.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="truncate each turn's printed text to N chars (0 = full text)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write to this file instead of stdout",
    )
    args = ap.parse_args()

    records, bad = load_records(args.prompts)
    if bad:
        print(f"warning: skipped {bad} malformed line(s)", file=sys.stderr)
    if not records:
        print("no prompt-dump records found", file=sys.stderr)
        return 1

    smap = None
    if args.session_map is not None:
        smap = json.loads(args.session_map.read_text())

    turns = dedup_turns(records, args.role)
    text = render(turns, smap, delta=args.delta, max_chars=args.max_chars)

    if args.out is not None:
        args.out.write_text(text)
        print(
            f"wrote {args.out}: {len(turns)} turn(s) "
            f"from {len(records)} record(s)",
            file=sys.stderr,
        )
    else:
        sys.stdout.write(text)

    n_sessions = len({_session_of(r, smap) for r in turns})
    print(
        f"summary: {len(turns)} turn(s), {n_sessions} session-group(s), "
        f"{len(records)} raw record(s)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
