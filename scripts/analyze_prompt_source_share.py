#!/usr/bin/env python3
"""Per-turn share of the prompt coming from the system prompt and from
file reads -- computed from an OpenCode PROFILE directory alone.

What each number is built from (all recorded even at the default
OPENCODE_PROFILE_MESSAGES=head, which truncates TEXT but keeps exact
CHAR COUNTS):

  system_chars  turn.start.system.chars -- the full system prompt length
                for that turn, summed over the system strings.
  read_chars    the cumulative output_chars of every `read` tool that
                finished in an EARLIER step of the same session. A read's
                output is what got appended to the conversation, so turn
                N's prompt carries the reads of steps 1..N-1.
  total_chars   system_chars + the summed chars of the turn's message
                array.

The shares are CHAR shares. They are reported as such, and the token
columns are those shares applied to the turn's REAL ISL (llm.end
tokens.input + tokens.cache.read) -- an estimate, because file content
and prose do not tokenize at the same chars/token ratio (code and
whitespace-heavy text run denser). Treat the shares as solid and the
token counts as approximate.

Two things this CANNOT see, both inherent to the profile:
  * the chat template. The profile is opencode's PRE-template message
    array (CLAUDE.md, "Two different prompt measurement layers"), so
    role headers and tool-call framing are absent from total_chars.
  * a read whose result opencode truncated or summarised before putting
    it in the conversation: output_chars is what the TOOL returned.
For exact, post-template, per-token accounting use the engine prompt
dump instead (DYN_PROMPT_DUMP + scripts/analyze_prompt_kv_composition.py).

Inputs:
  --profiles <dir|file>  profile NDJSON dir (one <sessionID>.jsonl per
                         session) or a single session file
  --read-tools read      comma-separated tool names counted as file
                         reads (default "read")
  --out <csv>            optional per-turn CSV

Output:
  stdout                 per-session table + a run-wide summary
  <out>                  session_id, step, request_id, system_chars,
                         read_chars, other_chars, total_chars,
                         system_share, read_share, isl_tokens,
                         est_system_tokens, est_read_tokens

Usage:
  scripts/analyze_prompt_source_share.py --profiles <workspace_root>/profiles \
      --out results/run1/prompt_source_share.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _files(path: Path) -> list[Path]:
    return [path] if path.is_file() else sorted(path.glob("*.jsonl"))


def _messages_chars(msgs) -> int:
    """Summed message chars across every snapshot fidelity.

    head  -> [{role, parts, chars, head}]        (chars is exact)
    count -> {count, roles, total_chars}
    full  -> the raw message array; fall back to len(json)
    """
    if isinstance(msgs, dict):
        return int(msgs.get("total_chars") or 0)
    if not isinstance(msgs, list):
        return 0
    total = 0
    for m in msgs:
        if isinstance(m, dict) and isinstance(m.get("chars"), (int, float)):
            total += int(m["chars"])
        else:
            content = m.get("content") if isinstance(m, dict) else m
            total += len(content if isinstance(content, str)
                         else json.dumps(content, ensure_ascii=False))
    return total


def load_turns(profiles: Path, read_tools: set[str]) -> list[dict]:
    """One row per turn, with the read output of PRIOR steps accumulated.

    Events are read in file order; `turn.start` opens a turn, `llm.end`
    attaches its token usage, `tool.end` adds to the running read total
    that the NEXT turn's prompt will carry.
    """
    rows: list[dict] = []
    for f in _files(profiles):
        session = f.stem
        read_so_far = 0
        pending: dict[int, dict] = {}      # step -> row awaiting llm.end
        order: list[int] = []
        with f.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("ev")
                if kind == "turn.start":
                    sysinfo = ev.get("system")
                    sys_chars = int((sysinfo or {}).get("chars") or 0) \
                        if isinstance(sysinfo, dict) else 0
                    msg_chars = _messages_chars(ev.get("messages"))
                    step = int(ev.get("step") or 0)
                    pending[step] = {
                        "session_id": ev.get("sessionID") or session,
                        "step": step,
                        "request_id": "",
                        "system_chars": sys_chars,
                        # Reads that had already returned when this prompt
                        # was assembled.
                        "read_chars": read_so_far,
                        "total_chars": sys_chars + msg_chars,
                        "isl_tokens": "",
                    }
                    order.append(step)
                elif kind == "tool.end":
                    if (ev.get("name") in read_tools
                            and isinstance(ev.get("output_chars"), (int, float))):
                        read_so_far += int(ev["output_chars"])
                elif kind == "llm.end":
                    step = int(ev.get("step") or 0)
                    row = pending.get(step)
                    if row is None:
                        continue
                    row["request_id"] = ev.get("request_id") or ""
                    tok = ev.get("tokens")
                    if isinstance(tok, dict):
                        inp = tok.get("input")
                        cache = tok.get("cache")
                        read = (cache.get("read")
                                if isinstance(cache, dict) else 0) or 0
                        if isinstance(inp, (int, float)):
                            row["isl_tokens"] = int(inp) + int(read)
        for step in order:
            row = pending[step]
            total = row["total_chars"]
            # read_chars is bounded by total: a read whose output opencode
            # trimmed before inserting it would otherwise push the share
            # above 1 and quietly corrupt the summary.
            row["read_chars"] = min(row["read_chars"], max(total - row["system_chars"], 0))
            row["other_chars"] = max(total - row["system_chars"] - row["read_chars"], 0)
            row["system_share"] = round(row["system_chars"] / total, 4) if total else ""
            row["read_share"] = round(row["read_chars"] / total, 4) if total else ""
            isl = row["isl_tokens"]
            if isl != "" and total:
                row["est_system_tokens"] = int(round(isl * row["system_chars"] / total))
                row["est_read_tokens"] = int(round(isl * row["read_chars"] / total))
            else:
                row["est_system_tokens"] = ""
                row["est_read_tokens"] = ""
            rows.append(row)
    return rows


def print_sessions(rows: list[dict], limit: int) -> None:
    by_session: dict[str, list[dict]] = {}
    for r in rows:
        by_session.setdefault(r["session_id"], []).append(r)
    shown = 0
    for sid in sorted(by_session):
        if shown >= limit:
            print(f"\n... {len(by_session) - shown} more sessions "
                  "(all of them are in the CSV)")
            break
        shown += 1
        print(f"\nsession {sid}")
        print(f"  {'step':>5} {'system':>9} {'read':>9} {'other':>9} "
              f"{'total':>10} {'sys%':>7} {'read%':>7} {'sys+read%':>10} "
              f"{'ISL tok':>9}")
        for r in sorted(by_session[sid], key=lambda x: x["step"]):
            total = r["total_chars"]
            both = ((r["system_chars"] + r["read_chars"]) / total * 100
                    if total else 0.0)
            print(f"  {r['step']:>5} {r['system_chars']:>9} "
                  f"{r['read_chars']:>9} {r['other_chars']:>9} "
                  f"{total:>10} "
                  f"{(r['system_share'] or 0) * 100:>6.1f}% "
                  f"{(r['read_share'] or 0) * 100:>6.1f}% "
                  f"{both:>9.1f}% "
                  f"{r['isl_tokens'] if r['isl_tokens'] != '' else '-':>9}")


def print_summary(rows: list[dict]) -> None:
    print(f"\nturns: {len(rows)}")
    if not rows:
        return
    tot = sum(r["total_chars"] for r in rows)
    sysc = sum(r["system_chars"] for r in rows)
    readc = sum(r["read_chars"] for r in rows)
    if tot:
        # Char-weighted: what fraction of everything the model was asked
        # to process came from each source. The per-turn mean would
        # over-weight the short early turns.
        print(f"  char-weighted: system {100.0 * sysc / tot:.1f}%  "
              f"read {100.0 * readc / tot:.1f}%  "
              f"together {100.0 * (sysc + readc) / tot:.1f}%")
    shares = sorted((r["system_chars"] + r["read_chars"]) / r["total_chars"]
                    for r in rows if r["total_chars"])
    if shares:
        def _p(q):
            return shares[min(len(shares) - 1, int(round(q * (len(shares) - 1))))]
        print(f"  per-turn system+read share: p10={_p(0.10):.3f} "
              f"p50={_p(0.50):.3f} p90={_p(0.90):.3f}")
    isl = [r for r in rows if r["isl_tokens"] != ""]
    if isl:
        est_s = sum(r["est_system_tokens"] for r in isl)
        est_r = sum(r["est_read_tokens"] for r in isl)
        tot_isl = sum(r["isl_tokens"] for r in isl)
        print(f"  estimated tokens over {len(isl)} turns with usage data: "
              f"system ~{est_s} + read ~{est_r} of {tot_isl} ISL "
              f"({100.0 * (est_s + est_r) / tot_isl:.1f}%)")
    else:
        print("  no llm.end token usage in these profiles -- shares only, "
              "no token estimates")


COLS = ["session_id", "step", "request_id", "system_chars", "read_chars",
        "other_chars", "total_chars", "system_share", "read_share",
        "isl_tokens", "est_system_tokens", "est_read_tokens"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profiles", required=True, type=Path)
    ap.add_argument("--read-tools", default="read",
                    help="comma-separated tool names whose output counts as "
                         "file content (default: read)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-sessions", type=int, default=5,
                    help="sessions printed in full (default 5; the CSV has "
                         "every one)")
    args = ap.parse_args(argv)

    if not args.profiles.exists():
        print(f"error: {args.profiles} not found", file=sys.stderr)
        return 2
    read_tools = {t.strip() for t in args.read_tools.split(",") if t.strip()}
    rows = load_turns(args.profiles, read_tools)
    if not rows:
        print("error: no turn.start events found in the given profiles",
              file=sys.stderr)
        return 2

    print_sessions(rows, args.max_sessions)
    print_summary(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
