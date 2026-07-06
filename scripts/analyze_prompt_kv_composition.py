#!/usr/bin/env python3
"""Per-turn KV / prompt-composition table from DYN_PROMPT_DUMP output.

Input: the engine-prompt NDJSON that `dynamo-prompt-dump.patch` writes
(one record per request: {ts, request_id, role, num_prompt_tokens,
prompt_text?, prompt_token_ids?}). Records are deduped per request_id
(role=prefill preferred -- PD sends identical token_ids to both workers)
and chained into sessions by SHARED-PREFIX matching: dynamo carries no
opencode sessionID, but turn N+1's prompt begins with turn N's prompt
(the conversation history grows append-only), so the longest-common-
prefix chain recovers the session structure. Title-agent one-shots and
task-tool child sessions land as their own (usually 1-2 turn) chains.

Output CSV, one row per (session, turn):
  session_id    synthetic chain id (ses-001, ...) -- NOT the opencode id
  turn          1-based index within the chain
  prefix_total  tokens shared with the PREVIOUS turn's prompt = context
                whose KV can already be resident (upper bound: assumes
                no eviction between the two requests; turn 1 -> 0)
  prefix_file   the file-content portion of that shared prefix
  new_file      file-content tokens in this turn's NEW suffix
  new_other     everything else in the new suffix
  (+ request_id, ts, total_tokens, mode for sanity/joins)

"File content" = spans delimited by the literal markers opencode's
tools emit into the conversation (read tool <file>...</file> blocks,
write/edit tool-call bodies) as rendered by the qwen3-coder chat
template. See FILE_SPAN_RES / TOOLCALL_RE below -- marker set is
verified against the vendored opencode source; extend there if the
agent config adds file-bearing tools.

Token accounting modes (column `mode`):
  ids     prompt_token_ids present (DYN_PROMPT_DUMP_TOKENS=1):
          prefix_total is an EXACT token LCP; file spans inside each
          region are converted char->token proportionally.
  tok     --tokenizer <hf-id-or-path> given (needs `transformers` on
          the host): every region/span measured by re-encoding with
          offset mapping -- exact modulo detokenize round-trip.
  chars   neither: all token numbers are char-proportional estimates
          scaled to num_prompt_tokens. Fine for shares, coarse for
          absolute counts.
Recommend DYN_PROMPT_DUMP_TOKENS=1 (+ optionally --tokenizer) for
capture runs feeding this analysis.

Usage:
  scripts/analyze_prompt_kv_composition.py --prompts <workspace_root>/prompts \\
      --out results/run1/kv_composition.csv \\
      [--tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct] \\
      [--min-prefix-frac 0.5] [--min-prefix-chars 200]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# File-content span markers (vendored-source-verified; see module docstring).
# Each regex's ENTIRE match is counted as file content.
# ---------------------------------------------------------------------------

FILE_SPAN_RES: list[re.Pattern[str]] = [
    # opencode read tool output (tool/read.ts:256-274): the file body is
    # wrapped as `<type>file</type>\n<content>\n1: line...\n</content>`
    # (1-indexed `N: ` line prefixes, no zero-pad; a "(End of file...)"/
    # "Showing lines..." notice sits just inside). The directory variant
    # uses <type>directory</type> + <entries> (names only) -- not file
    # content, deliberately NOT matched here.
    re.compile(r"<type>file</type>\s*<content>.*?</content>", re.DOTALL),
]

# File bodies also enter the context through write/edit/apply_patch CALL
# arguments in the assistant turns. qwen3-coder's chat template renders
# tool calls in XML form:
#   <tool_call>\n<function=write>\n<parameter=content>\nBODY\n</parameter>...
# (dynamo parser config config.rs:567). Some templates use the Hermes
# JSON form instead (<tool_call>{"name":...,"arguments":{...}}</tool_call>)
# -- both are supported; grep a real dump for `<function=` vs `{"name":`
# to see which your template emits.
XML_FILE_PARAM_RE = re.compile(
    r"<parameter=(?:content|oldString|newString|patchText)>\n?(.*?)\n?</parameter>",
    re.DOTALL)
TOOLCALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FILE_BEARING_ARGS: dict[str, tuple[str, ...]] = {
    "write": ("content",),
    "edit": ("oldString", "newString"),
    "apply_patch": ("patchText",),
}


def file_char_intervals(text: str) -> list[tuple[int, int]]:
    """Sorted, merged [start, end) char intervals of file content."""
    spans: list[tuple[int, int]] = []
    for pat in FILE_SPAN_RES:
        spans.extend(m.span() for m in pat.finditer(text))
    # qwen3-coder XML tool-call parameters carrying file bodies.
    spans.extend(m.span(1) for m in XML_FILE_PARAM_RE.finditer(text))
    # Hermes/JSON tool-call fallback (template-dependent).
    for m in TOOLCALL_JSON_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name")
        args = payload.get("arguments")
        fields = FILE_BEARING_ARGS.get(str(name), ())
        if not fields or not isinstance(args, dict):
            continue
        body = m.group(1)
        base = m.start(1)
        for f_name in fields:
            val = args.get(f_name)
            if not isinstance(val, str) or not val:
                continue
            # Locate the JSON-encoded value inside the raw payload so the
            # interval lands at real char offsets (encoded form includes
            # escapes; find() on the encoded literal).
            encoded = json.dumps(val)[1:-1]
            pos = body.find(encoded)
            if pos >= 0 and encoded:
                spans.append((base + pos, base + pos + len(encoded)))
    if not spans:
        return []
    spans.sort()
    merged = [spans[0]]
    for s, e in spans[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def chars_in_region(intervals: list[tuple[int, int]],
                    start: int, end: int) -> int:
    """Total chars of `intervals` clipped to [start, end)."""
    total = 0
    for s, e in intervals:
        lo, hi = max(s, start), min(e, end)
        if hi > lo:
            total += hi - lo
    return total


# ---------------------------------------------------------------------------
# Dump ingest + session chaining
# ---------------------------------------------------------------------------


@dataclass
class PromptRec:
    ts: float
    request_id: str
    role: str
    num_tokens: int
    text: str
    token_ids: list[int] | None


@dataclass
class Chain:
    session_id: str
    records: list[PromptRec] = field(default_factory=list)

    @property
    def last(self) -> PromptRec:
        return self.records[-1]


def load_dump(prompts_dir: Path) -> list[PromptRec]:
    by_req: dict[str, PromptRec] = {}
    files = sorted(prompts_dir.glob("prompt-*.jsonl"))
    if not files:
        raise SystemExit(f"no prompt-*.jsonl under {prompts_dir}")
    for path in files:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = r.get("prompt_text")
                if not text:
                    continue  # dumped with DYN_PROMPT_DUMP_TEXT=0 -- unusable
                rec = PromptRec(
                    ts=float(r.get("ts", 0.0)),
                    request_id=str(r.get("request_id")),
                    role=str(r.get("role", "")),
                    num_tokens=int(r.get("num_prompt_tokens") or 0),
                    text=text,
                    token_ids=r.get("prompt_token_ids"),
                )
                prev = by_req.get(rec.request_id)
                # Prefer the prefill copy (canonical); otherwise first wins.
                if prev is None or (prev.role != "prefill" and rec.role == "prefill"):
                    by_req[rec.request_id] = rec
    return sorted(by_req.values(), key=lambda r: r.ts)


def _lcp_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:  # binary search beats char loop on 100k-char prompts
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _lcp_tokens(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def chain_sessions(records: list[PromptRec], *, min_frac: float,
                   min_chars: int) -> list[Chain]:
    """Greedy chaining: attach each prompt (ts order) to the chain whose
    LAST prompt shares the longest char prefix, if that prefix is both
    >= min_chars and >= min_frac of the earlier prompt's length. A turn
    always extends its session's previous prompt (history is append-only),
    so real continuations score near 100%; unrelated sessions share only
    the system prompt, which min_frac filters out."""
    chains: list[Chain] = []
    for rec in records:
        best: Chain | None = None
        best_lcp = 0
        for ch in chains:
            lcp = _lcp_len(ch.last.text, rec.text)
            if lcp > best_lcp:
                best, best_lcp = ch, lcp
        prev_len = len(best.last.text) if best else 0
        if (best is not None and best_lcp >= min_chars
                and prev_len > 0 and best_lcp >= min_frac * prev_len):
            best.records.append(rec)
        else:
            chains.append(Chain(session_id=f"ses-{len(chains) + 1:03d}",
                                records=[rec]))
    return chains


# ---------------------------------------------------------------------------
# Char->token accounting
# ---------------------------------------------------------------------------


class TokenMapper:
    """Converts char intervals to token counts for one prompt."""

    def __init__(self, rec: PromptRec, tokenizer=None):
        self.rec = rec
        self.mode = "chars"
        self._starts: list[int] | None = None  # token start offsets
        if tokenizer is not None:
            enc = tokenizer(rec.text, return_offsets_mapping=True,
                            add_special_tokens=False)
            self._starts = [s for s, _ in enc["offset_mapping"]]
            self.mode = "tok"
        elif rec.token_ids:
            self.mode = "ids"

    def total(self) -> int:
        if self._starts is not None:
            return len(self._starts)
        return self.rec.num_tokens

    def region_tokens(self, start: int, end: int) -> float:
        """Token count for char region [start, end)."""
        if end <= start:
            return 0.0
        if self._starts is not None:
            import bisect
            lo = bisect.bisect_left(self._starts, start)
            hi = bisect.bisect_left(self._starts, end)
            return float(hi - lo)
        # proportional: tokens ~ chars * (total_tokens / total_chars)
        n_chars = max(1, len(self.rec.text))
        return (end - start) * self.total() / n_chars


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", required=True, type=Path,
                    help="DYN_PROMPT_DUMP_DIR (contains prompt-*.jsonl)")
    ap.add_argument("--out", required=True, type=Path,
                    help="Output CSV path")
    ap.add_argument("--tokenizer", default=None,
                    help="Optional HF tokenizer id/path for exact span "
                         "token counts (requires `transformers`)")
    ap.add_argument("--min-prefix-frac", type=float, default=0.5,
                    help="Chain threshold: shared prefix must cover this "
                         "fraction of the previous prompt. Default 0.5.")
    ap.add_argument("--min-prefix-chars", type=int, default=200,
                    help="Chain threshold: absolute minimum shared chars "
                         "(filters system-prompt-only overlap). Default 200.")
    args = ap.parse_args(argv)

    tokenizer = None
    if args.tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except ImportError:
            raise SystemExit("--tokenizer needs `transformers` installed")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    records = load_dump(args.prompts)
    print(f"{len(records)} unique requests from {args.prompts}")
    chains = chain_sessions(records, min_frac=args.min_prefix_frac,
                            min_chars=args.min_prefix_chars)
    n_multi = sum(1 for c in chains if len(c.records) > 1)
    print(f"chained into {len(chains)} session(s) ({n_multi} multi-turn)")

    rows: list[dict[str, Any]] = []
    for ch in chains:
        prev: PromptRec | None = None
        for turn, rec in enumerate(ch.records, 1):
            mapper = TokenMapper(rec, tokenizer)
            intervals = file_char_intervals(rec.text)

            # --- prefix boundary (chars) with the previous turn ---
            if prev is None:
                lcp_chars = 0
            else:
                lcp_chars = _lcp_len(prev.text, rec.text)

            # --- prefix_total in tokens ---
            if prev is None:
                prefix_total = 0.0
            elif (mapper.mode == "ids" and prev.token_ids and rec.token_ids):
                prefix_total = float(_lcp_tokens(prev.token_ids, rec.token_ids))
            else:
                prefix_total = mapper.region_tokens(0, lcp_chars)

            prefix_file = (_region_file_tokens(mapper, intervals, 0, lcp_chars)
                           if lcp_chars else 0.0)
            new_file = _region_file_tokens(mapper, intervals, lcp_chars,
                                           len(rec.text))
            new_total = max(0.0, mapper.total() - prefix_total)
            new_other = max(0.0, new_total - new_file)

            rows.append({
                "session_id": ch.session_id,
                "turn": turn,
                "prefix_total": round(prefix_total, 1),
                "prefix_file": round(prefix_file, 1),
                "new_file": round(new_file, 1),
                "new_other": round(new_other, 1),
                "total_tokens": mapper.total(),
                "request_id": rec.request_id,
                "ts": rec.ts,
                "mode": mapper.mode,
            })
            prev, prev_mapper = rec, mapper

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["session_id", "turn", "prefix_total",
                            "prefix_file", "new_file", "new_other"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    return 0


def _region_file_tokens(mapper: TokenMapper,
                        intervals: list[tuple[int, int]],
                        start: int, end: int) -> float:
    """Token count of file-content intervals clipped to [start, end)."""
    total = 0.0
    for s, e in intervals:
        lo, hi = max(s, start), min(e, end)
        if hi > lo:
            total += mapper.region_tokens(lo, hi)
    return total


if __name__ == "__main__":
    sys.exit(main())
