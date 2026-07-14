#!/usr/bin/env python3
"""Export DYN_PROMPT_DUMP records as structured per-turn/per-segment JSONL
for import into external simulators.

`DYN_PROMPT_DUMP=1` (dynamo-prompt-dump.patch) captures, per request, the
EXACT engine prompt as flat detokenized text (`prompt-*.jsonl`, fields
{ts, request_id, role, num_prompt_tokens, prompt_text?, ...}). That flat
text is the whole conversation-so-far pushed through the model's chat
template. This script re-structures it: one output JSON line per request,
with the prompt split into chat-template TURNS (system/user/assistant
blocks) and each turn split into SEGMENTS by kind:

    text | think | tool_call | tool_response

so a downstream simulator can reconstruct conversation growth, per-kind
token/char budgets, and shared-prefix structure without re-parsing raw
prompt text. (format_prompt_dump.py stays the human pretty-printer; this
is the machine-readable exporter.)

Output record (one JSON line per request):

    {
      "request_id": "...", "ts": 1712.3, "role": "prefill",
      "source": "prompt-1234.jsonl",
      "session_id": "ses_..",          # only with --session-map
      "request_index": 3,              # 1-based ts-order within the session group
      "num_prompt_tokens": 8123,       # engine-reported, from the dump
      "num_chars": 31871,              # len(prompt_text)
      "template": "qwen3",
      "parse_ok": true, "warnings": [],
      "turns": [
        {"index": 0, "role": "system", "start": 0, "end": 812,
         "num_chars": 812, "generation_prompt": false,
         "segments": [
           {"kind": "text", "start": 20, "end": 800, "num_chars": 780,
            "text": "..."}]},
        {"index": 2, "role": "assistant", ...,
         "segments": [
           {"kind": "think", ...},
           {"kind": "text", ...},
           {"kind": "tool_call", "name": "bash", ...}]},
        ...
      ]
    }

Losslessness contract (what a simulator may rely on):
  * `start`/`end` are absolute char offsets into the dump's prompt_text.
  * The preamble turn (role "_preamble", only if text precedes the first
    turn marker) plus the turns tile [0, len(prompt_text)) exactly.
  * Within a turn, segments tile the turn's CONTENT span exactly; the
    difference `turn.num_chars - sum(seg.num_chars)` is the chat-template
    framing overhead (role header + end-of-turn marker).
  * `text` is the verbatim slice INCLUDING the segment's own tags
    (len(text) == num_chars); pass --no-text for a compact offsets+sizes
    trace (tool_call `name` is kept).
  * The final `<|im_start|>assistant\\n` generation tail is a normal turn
    with "generation_prompt": true and no segments.

Template: --template qwen3 (default) parses ChatML framing
(`<|im_start|>ROLE\\n ... <|im_end|>`) as produced by the Qwen3 /
Qwen3-Coder chat templates. Segment tags are matched sequentially,
earliest-first (a tag inside an open segment, e.g. `<tool_call>` inside a
tool_response body, does NOT start a new segment):
  * think:         <think> ... </think>
  * tool_call:     <tool_call> ... </tool_call>                (Qwen3 / Qwen3-Coder)
                   ]<]minimax[>[<tool_call> ... ]<]minimax[>[</tool_call>   (MiniMax M3)
                   <minimax:tool_call> ... </minimax:tool_call>            (MiniMax M2)
  * tool_response: <tool_response> ... </tool_response>
tool_call tag strings verified against the vendored parser configs
(dynamo/lib/parsers/src/tool_calling/config.rs); <think>/</think> matches the
vendored reasoning parsers (dynamo/lib/parsers/src/reasoning/mod.rs);
<tool_response> is the Qwen chat template's prompt-side wrapper (it never
appears in the output parsers, so there is no vendored source to pin it to).
tool_call `name` is
extracted from `<function=NAME>` (Qwen3-Coder XML), `"name": "..."`
(Qwen3 JSON), or `<invoke name="NAME">` (MiniMax M3). --template raw
skips turn splitting (single "_raw" turn; segments still scanned) — also
the automatic fallback (with a warning) when no ChatML marker is found,
e.g. for models whose outer message framing is not ChatML (MiniMax M3's
outer framing is not implemented; its tool_call segments still split).

Optional --tokenizer <hf-id-or-local-path> re-tokenizes each slice
(transformers AutoTokenizer, add_special_tokens=False) and adds
`num_tokens` to every segment/turn plus a record-level `num_tokens_text`
to sanity-check against the engine-reported num_prompt_tokens. Needs the
tokenizer locally (run on the GPU host); boundary effects make the sum
approximate (~exact for special-token-delimited templates).

Known text-level limitation: detokenized text cannot distinguish a REAL
special token from tool output that happens to contain the same literal
string (e.g. a bash step that cats a chat template). Rare in practice;
a --tokenizer count mismatch is the tell.

Examples:
  scripts/export_prompt_turns.py --prompts /tmp/testbed-workspaces/prompts \
      --out turns.jsonl
  scripts/export_prompt_turns.py --prompts prompts/ --no-text \
      --session-map reqmap.json --out compact.jsonl
  scripts/export_prompt_turns.py --prompts prompts/ \
      --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct --out sized.jsonl

Concurrency caveat (same as format_prompt_dump.py): dumps carry no
opencode sessionID, so `request_index` is only meaningful within a
single-session capture or with a --session-map JSON
({"request_id": "session_id"}).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

TURN_START = "<|im_start|>"
TURN_END = "<|im_end|>"

# (kind, open tag, close tag). Ordered most-specific first: the MiniMax M3
# open contains "<tool_call>" as a substring, and regex alternation prefers
# the earlier-listed pattern at the same match position.
_SEGMENT_TAGS: list[tuple[str, str, str]] = [
    ("tool_call", "]<]minimax[>[<tool_call>", "]<]minimax[>[</tool_call>"),
    ("tool_call", "<minimax:tool_call>", "</minimax:tool_call>"),
    ("tool_call", "<tool_call>", "</tool_call>"),
    ("tool_response", "<tool_response>", "</tool_response>"),
    ("think", "<think>", "</think>"),
]
_OPEN_RE = re.compile("|".join(re.escape(open_) for _, open_, _c in _SEGMENT_TAGS))
_OPEN_TO_SPEC = {open_: (kind, close) for kind, open_, close in _SEGMENT_TAGS}

# tool_call name extraction, tried in order.
_NAME_RES = [
    re.compile(r"<function=([^>\n]+)>"),          # Qwen3-Coder XML style
    re.compile(r'"name"\s*:\s*"([^"]+)"'),        # Qwen3 JSON style
    re.compile(r'<invoke name="([^"]+)">'),       # MiniMax M3 style
]


# ---------- dump loading (mirrors format_prompt_dump.py; scripts are
# self-contained by repo convention, so these helpers are duplicated) ----------


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


def dedup_requests(records: list[dict], role: str) -> list[dict]:
    """Collapse the prefill/decode copies of one request_id into one record.

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
    chosen: dict[Any, dict] = {}
    for r in records:
        rid = r.get("request_id")
        if rid is None:
            chosen[id(r)] = r
            continue
        prev = chosen.get(rid)
        if prev is None or (prev.get("role") != "prefill" and r.get("role") == "prefill"):
            chosen[rid] = r
    return list(chosen.values())


def _session_of(rec: dict, smap: dict[str, str] | None) -> str:
    if smap is None:
        return "all"
    return smap.get(rec.get("request_id", ""), "unknown")


# ---------- prompt text -> turns/segments ----------


def _tool_call_name(slice_text: str) -> str | None:
    for rx in _NAME_RES:
        m = rx.search(slice_text)
        if m:
            return m.group(1).strip()
    return None


def scan_segments(text: str, base: int, warnings: list[str]) -> list[dict]:
    """Split one turn's CONTENT span into kind-tagged segments.

    Sequential earliest-open-tag scan: a tag literal INSIDE an already-open
    segment body never starts a new segment. Segments tile [base,
    base+len(text)) exactly; whitespace-only gaps stay as "text" segments so
    offsets reconstruct the content verbatim. An open tag without its close
    swallows the rest of the content (warning "unclosed:<tag>")."""
    segments: list[dict] = []

    def _emit(kind: str, start: int, end: int) -> None:
        seg: dict[str, Any] = {
            "kind": kind,
            "start": base + start,
            "end": base + end,
            "num_chars": end - start,
        }
        if kind == "tool_call":
            seg["name"] = _tool_call_name(text[start:end])
        segments.append(seg)

    pos = 0
    while True:
        m = _OPEN_RE.search(text, pos)
        if m is None:
            break
        if m.start() > pos:
            _emit("text", pos, m.start())
        kind, close = _OPEN_TO_SPEC[m.group(0)]
        close_at = text.find(close, m.end())
        if close_at < 0:
            warnings.append(f"unclosed:{m.group(0)}")
            _emit(kind, m.start(), len(text))
            pos = len(text)
            break
        end = close_at + len(close)
        _emit(kind, m.start(), end)
        pos = end
    if pos < len(text):
        _emit("text", pos, len(text))
    return segments


def split_turns(text: str, template: str) -> tuple[list[dict], list[str]]:
    """Split a full prompt_text into chat-template turns.

    Returns (turns, warnings). Turns tile [0, len(text)) exactly (a
    "_preamble" pseudo-turn covers any text before the first marker; with
    template="raw" or when no marker exists, a single "_raw" turn covers
    everything). Each turn's segments cover its content span (between the
    role header line and the end-of-turn marker)."""
    warnings: list[str] = []
    starts = []
    if template != "raw":
        at = text.find(TURN_START)
        while at >= 0:
            starts.append(at)
            at = text.find(TURN_START, at + 1)

    turns: list[dict] = []

    def _add(role: str, start: int, end: int, content_start: int,
             content_end: int, generation_prompt: bool) -> None:
        turns.append({
            "index": len(turns),
            "role": role,
            "start": start,
            "end": end,
            "num_chars": end - start,
            "generation_prompt": generation_prompt,
            "segments": scan_segments(
                text[content_start:content_end], content_start, warnings),
        })

    if not starts:
        if template != "raw":
            warnings.append("no_chatml_markers")
        if text:
            _add("_raw", 0, len(text), 0, len(text), False)
        return turns, warnings

    if starts[0] > 0:
        _add("_preamble", 0, starts[0], 0, starts[0], False)

    for i, block_start in enumerate(starts):
        block_end = starts[i + 1] if i + 1 < len(starts) else len(text)
        block = text[block_start:block_end]
        header_end_rel = block.find("\n", len(TURN_START))
        if header_end_rel < 0:
            # Marker + role token and nothing else (truncated tail).
            role = block[len(TURN_START):].strip()
            _add(role or "_unknown", block_start, block_end,
                 block_end, block_end, True)
            continue
        role = block[len(TURN_START):header_end_rel].strip()
        content_start = block_start + header_end_rel + 1
        end_rel = block.find(TURN_END, header_end_rel + 1)
        if end_rel < 0:
            # No end-of-turn marker: the add_generation_prompt tail (or a
            # mid-generation truncation) -- content runs to the block end.
            _add(role, block_start, block_end, content_start, block_end,
                 generation_prompt=(content_start == block_end))
            continue
        _add(role, block_start, block_end, content_start,
             block_start + end_rel, False)
    return turns, warnings


# ---------- record assembly ----------


def _load_tokenizer(spec: str) -> Callable[[str], int]:
    """Indirection so tests can inject a fake counter. Returns text -> ntokens."""
    from transformers import AutoTokenizer  # type: ignore[import-not-found]

    tok = AutoTokenizer.from_pretrained(spec, trust_remote_code=True)
    return lambda s: len(tok.encode(s, add_special_tokens=False))


def export_record(rec: dict, *, template: str, include_text: bool,
                  count_tokens: Callable[[str], int] | None,
                  session_id: str | None, request_index: int) -> dict:
    """Build one output record from one dump record."""
    out: dict[str, Any] = {
        "request_id": rec.get("request_id"),
        "ts": rec.get("ts"),
        "role": rec.get("role"),
        "source": rec.get("_source"),
        "num_prompt_tokens": rec.get("num_prompt_tokens"),
        "template": template,
    }
    if session_id is not None:
        out["session_id"] = session_id
        out["request_index"] = request_index

    text = rec.get("prompt_text")
    if not isinstance(text, str):
        out.update({
            "num_chars": 0,
            "parse_ok": False,
            "warnings": [str(rec.get("decode_error")
                             or "no prompt_text (DYN_PROMPT_DUMP_TEXT off?)")],
            "turns": [],
        })
        return out

    turns, warnings = split_turns(text, template)
    for t in turns:
        for seg in t["segments"]:
            if include_text:
                seg["text"] = text[seg["start"]:seg["end"]]
            if count_tokens is not None:
                seg["num_tokens"] = count_tokens(text[seg["start"]:seg["end"]])
        if count_tokens is not None:
            t["num_tokens"] = count_tokens(text[t["start"]:t["end"]])
    out.update({
        "num_chars": len(text),
        "parse_ok": True,
        "warnings": warnings,
        "turns": turns,
    })
    if count_tokens is not None:
        out["num_tokens_text"] = count_tokens(text)
    return out


def export_records(records: list[dict], smap: dict[str, str] | None, *,
                   template: str, include_text: bool,
                   count_tokens: Callable[[str], int] | None) -> list[dict]:
    """Group by session (first-seen order), sort each group by ts, number
    requests within the group, and export every record."""
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(_session_of(r, smap), []).append(r)
    for recs in groups.values():
        recs.sort(key=lambda r: (r.get("ts") or 0))

    out: list[dict] = []
    for sess, recs in groups.items():
        for i, r in enumerate(recs, start=1):
            out.append(export_record(
                r,
                template=template,
                include_text=include_text,
                count_tokens=count_tokens,
                session_id=None if smap is None else sess,
                request_index=i,
            ))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--prompts", required=True, nargs="+", type=Path,
        help="prompt-*.jsonl dump dir(s) and/or file(s) "
             "(e.g. <workspace_root>/prompts)",
    )
    ap.add_argument(
        "--role", choices=["auto", "prefill", "decode", "both"], default="auto",
        help="auto (default): one record per request_id, prefer prefill; "
             "prefill/decode: that role only; both: every record",
    )
    ap.add_argument(
        "--template", choices=["qwen3", "raw"], default="qwen3",
        help="chat-template framing to parse (qwen3 = ChatML <|im_start|> "
             "blocks; raw = no turn split, segments only)",
    )
    ap.add_argument(
        "--session-map", type=Path, default=None,
        help='JSON {"request_id": "session_id"} to group + number requests '
             "per session",
    )
    ap.add_argument(
        "--no-text", action="store_true",
        help="omit segment text (keep offsets/sizes/kinds/names) for a "
             "compact trace",
    )
    ap.add_argument(
        "--tokenizer", default=None,
        help="HF id or local path; adds num_tokens per segment/turn and "
             "num_tokens_text per record (needs transformers + the "
             "tokenizer locally)",
    )
    ap.add_argument(
        "--out", type=Path, default=None,
        help="write JSONL to this file instead of stdout",
    )
    args = ap.parse_args(argv)

    records, bad = load_records(args.prompts)
    if bad:
        print(f"warning: skipped {bad} malformed line(s)", file=sys.stderr)
    if not records:
        print("no prompt-dump records found", file=sys.stderr)
        return 1

    smap = None
    if args.session_map is not None:
        smap = json.loads(args.session_map.read_text())

    count_tokens = _load_tokenizer(args.tokenizer) if args.tokenizer else None

    requests = dedup_requests(records, args.role)
    exported = export_records(
        requests, smap,
        template=args.template,
        include_text=not args.no_text,
        count_tokens=count_tokens,
    )

    lines = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in exported)
    if args.out is not None:
        args.out.write_text(lines, encoding="utf-8")
    else:
        sys.stdout.write(lines)

    kinds: dict[str, int] = {}
    n_warn = 0
    for r in exported:
        n_warn += len(r.get("warnings") or [])
        for t in r.get("turns") or []:
            for seg in t["segments"]:
                kinds[seg["kind"]] = kinds.get(seg["kind"], 0) + 1
    dest = str(args.out) if args.out is not None else "stdout"
    print(
        f"exported {len(exported)} request(s) from {len(records)} raw "
        f"record(s) -> {dest}; segments: "
        + (", ".join(f"{k}={v}" for k, v in sorted(kinds.items())) or "none")
        + (f"; {n_warn} warning(s)" if n_warn else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
