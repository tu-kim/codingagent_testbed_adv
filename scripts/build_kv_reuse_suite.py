#!/usr/bin/env python3
"""Build the file-read KV-cache-reuse feasibility test suite from prompt dumps.

Research concept (motivation -> what this dataset enables):
  When the agent `read`s a file, the file body lands in the NEXT prefill as
  thousands of extra tokens -> prefill gets expensive. Idea: precompute the
  file's KV cache and REUSE it when the file is read, recomputing only the
  first k tokens of the file body in-context (EPIC-style re-anchoring). To
  test feasibility we compare the text the model generates RIGHT AFTER a read
  under KV-reuse vs the ground-truth full-recompute generation.

This script turns the per-turn engine-prompt dumps (dynamo-prompt-dump.patch,
run with DYN_PROMPT_DUMP_TOKENS=1 so prompt_token_ids is present) into one
test-suite sample per "read-turn".

Pipeline (decisions locked with the researcher):
  * Keep only MAIN-agent turns: a turn whose prompt contains the testbed
    render_prompt wrapper (swebench.py). This drops the title-generation turn
    AND every `task` sub-agent turn (their user message is not our wrapper).
  * Group turns into samples by the embedded SWE-bench problem_statement
    (matched to load_samples(...) for the instance_id when --config is given).
  * Within a sample, order turns by ts (prompts grow monotonically).
  * Walk turns; a "read-turn" is one whose prompt introduces a NEW file-read
    block (opencode read tool: <type>file</type>\\n<content>\\n N: code ... ).
    Assumes ONE new read per turn (parallel reads out of scope).
  * Dedup: a read-turn whose new file was already seen earlier in the sample
    is skipped (no duplicate suite). Dedup key configurable (--dedup-key).
  * Each surviving read-turn = ONE suite. Its prompt is decomposed at EVERY
    file body present (cumulative): seg0, file1, seg1, file2, ..., fileN, segN
    -> file_* segments are the KV-reuse regions, seg_* are recompute regions.
  * Ground truth = that turn's generation (assistant text + tool_call), read
    off the NEXT turn's prompt (delta), up to the first <|im_end|>.
  * Token spans (absolute offsets into the prompt) are recorded per segment
    via the model tokenizer (--model), so the KV experiment knows exactly
    which token ranges are reusable and where each file sits (RoPE position).

  10-turn trace with 4 distinct file reads -> 4 suites.

Output: <out>/<sample_id>/read_<NNN>/{seg_*.txt, file_*.txt, ground_truth.txt,
manifest.json} plus <out>/index.json.

Usage:
  scripts/build_kv_reuse_suite.py --prompts <ws>/prompts --model <hf_or_path> \\
      --config results/test/config.json --out results/test/kv_suite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# From src/testbed/swebench.py _PROMPT_TEMPLATE -- present in every main-agent
# turn's prompt (it is the first user message, which persists in history).
WRAPPER_SENTINEL = "You are working in a git checkout that has already been cloned"
IM_END = "<|im_end|>"

# opencode read tool file output (packages/opencode/src/tool/read.ts:256-269):
#   <path>{abs}</path>\n<type>file</type>\n<content>\n{N: line}\n...[\n\n(note)]\n</content>
_FILE_BLOCK_RE = re.compile(
    r"<path>(?P<path>.*?)</path>\n<type>file</type>\n<content>\n(?P<body>.*?)\n</content>",
    re.DOTALL,
)
# Trailing meta note inside <content>: "(End of file - total N lines)" etc.
_CONTENT_NOTE_RE = re.compile(r"\n\n\([^\n]*\)\s*$")
# A numbered code line: "123: <code>".
_NUM_LINE_RE = re.compile(r"^(\d+): ", re.MULTILINE)


def _sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "surrogatepass")).hexdigest()


def load_records(prompts: list[Path]) -> list[dict]:
    """Read every prompt-*.jsonl record (dir globs prompt-*.jsonl)."""
    files: list[Path] = []
    for p in prompts:
        if p.is_dir():
            files.extend(sorted(p.glob("prompt-*.jsonl")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"warning: not found, skipping: {p}", file=sys.stderr)
    records: list[dict] = []
    for f in files:
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def is_main_turn(rec: dict) -> bool:
    """Main-agent turn: carries the testbed render_prompt wrapper. Drops the
    title-generation call and `task` sub-agent turns."""
    return WRAPPER_SENTINEL in (rec.get("prompt_text") or "")


def extract_problem_statement(prompt_text: str) -> str | None:
    """Pull the SWE-bench problem_statement out of the embedded user message:
    text after '# Issue' up to '# Hints' or the end of the user turn."""
    i = prompt_text.find("# Issue")
    if i == -1:
        return None
    rest = prompt_text[i + len("# Issue"):]
    for stop in ("\n# Hints", IM_END):
        j = rest.find(stop)
        if j != -1:
            rest = rest[:j]
    return rest.strip() or None


def build_instance_map(config_path: Path) -> dict[str, str]:
    """problem_statement(stripped) -> instance_id, by re-deriving the run's
    exact sample set with load_samples (deterministic given split/seed/n)."""
    config = json.loads(config_path.read_text())
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from testbed import swebench  # noqa: PLC0415

    samples = swebench.load_samples(config["split"], config["seed"], config["num_samples"])
    return {s["problem_statement"].strip(): s["instance_id"] for s in samples}


def strip_content_note(body: str) -> str:
    """Drop the trailing '(End of file ...)'/'(Output capped ...)' meta note so
    only the 'N: code' lines remain."""
    return _CONTENT_NOTE_RE.sub("", body)


def find_file_blocks(prompt_text: str) -> list[dict]:
    """All file-read blocks in a prompt, in order. Each: char span of the
    'N: code' BODY (note/tags excluded from the body span), path, line range,
    content hash."""
    blocks: list[dict] = []
    for m in _FILE_BLOCK_RE.finditer(prompt_text):
        raw_body = m.group("body")
        body = strip_content_note(raw_body)
        body_start = m.start("body")
        body_end = body_start + len(body)
        nums = [int(x) for x in _NUM_LINE_RE.findall(body)]
        blocks.append(
            {
                "path": m.group("path"),
                "body": body,
                "char_start": body_start,
                "char_end": body_end,
                "line_range": [nums[0], nums[-1]] if nums else None,
                "content_sha": _sha(body),
            }
        )
    return blocks


def dedup_key(block: dict, mode: str) -> str:
    if mode == "content":
        return block["content_sha"]
    if mode == "path":
        return block["path"]
    # path-range
    return f"{block['path']}@{block['line_range']}"


def ground_truth_from_next(cur_prompt: str, next_prompt: str) -> str | None:
    """The current turn's generation = what the next turn's prompt appends.
    cur_prompt ends at the assistant primer; next_prompt = cur_prompt + <gen>
    + <|im_end|> + .... Return <gen> (assistant text + tool_call)."""
    if not next_prompt.startswith(cur_prompt):
        # Robust fallback: longest common prefix.
        n = min(len(cur_prompt), len(next_prompt))
        k = 0
        while k < n and cur_prompt[k] == next_prompt[k]:
            k += 1
        tail = next_prompt[k:]
    else:
        tail = next_prompt[len(cur_prompt):]
    if not tail:
        return None
    end = tail.find(IM_END)
    return tail[:end] if end != -1 else tail


def chain_links(prompts: list[str]):
    """Reconstruct conversation lineage by prompt-prefix containment.

    The dump can't be trusted in ts order: it carries BOTH the prefill and
    decode copy of each request (identical prompt) and the prompt dir
    accumulates across runs. Instead we link turns by the fact that one
    turn's prompt is a strict prefix of the next turn's prompt.

    Returns (pred, succ) index lists:
      pred[i] = index of the LONGEST prompt that is a strict prefix of
                prompts[i] (the immediate predecessor turn), or None.
      succ[i] = index of the SHORTEST prompt that strictly extends prompts[i]
                (the immediate next turn), or None.
    O(n^2) but n = turns per sample (small); startswith short-circuits."""
    n = len(prompts)
    pred: list[int | None] = [None] * n
    succ: list[int | None] = [None] * n
    for i in range(n):
        pi = prompts[i]
        for j in range(n):
            if i == j:
                continue
            pj = prompts[j]
            if len(pj) < len(pi) and pi.startswith(pj):
                if pred[i] is None or len(pj) > len(prompts[pred[i]]):
                    pred[i] = j
            elif len(pj) > len(pi) and pj.startswith(pi):
                if succ[i] is None or len(pj) < len(prompts[succ[i]]):
                    succ[i] = j
    return pred, succ


def decompose(prompt_text: str, blocks: list[dict]) -> list[dict]:
    """Split the prompt at every file BODY into ordered segments:
    seg0, file1, seg1, file2, ..., fileN, segN."""
    segs: list[dict] = []
    cursor = 0
    for bi, b in enumerate(sorted(blocks, key=lambda x: x["char_start"]), start=1):
        if b["char_start"] > cursor:
            segs.append({"type": "text", "char_span": [cursor, b["char_start"]]})
        segs.append(
            {
                "type": "file",
                "char_span": [b["char_start"], b["char_end"]],
                "path": b["path"],
                "line_range": b["line_range"],
                "content_sha": b["content_sha"],
            }
        )
        cursor = b["char_end"]
    if cursor < len(prompt_text):
        segs.append({"type": "text", "char_span": [cursor, len(prompt_text)]})
    return segs


# ---- tokenizer-dependent (runs on the GPU host with the model) ----

def load_tokenizer(model: str):
    from transformers import AutoTokenizer  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    if not tok.is_fast:
        raise SystemExit(f"need a fast tokenizer for offset mapping: {model}")
    return tok


def char_offsets(prompt_text: str, tok, dumped_ids: list[int] | None,
                 allow_mismatch: bool = False):
    """Per-token (char_start, char_end) for prompt_text using the model
    tokenizer.

    The dump's prompt_text IS tok.decode(prompt_token_ids), so re-encoding it
    must reproduce the dumped ids for the offsets to line up token-for-token
    with the engine's REAL tokenization. For stock Qwen the chat-template
    specials (<|im_start|> etc.) fold back to single ids under
    add_special_tokens=False, so it matches. If it does NOT match, the offsets
    describe a DIFFERENT tokenization than the dump -> every token_span would
    be silently wrong. We refuse rather than emit untrustworthy spans
    (override with --allow-tokenizer-mismatch only if you know why)."""
    enc = tok(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    if dumped_ids is not None and list(enc["input_ids"]) != list(dumped_ids):
        msg = (f"re-encoded ids ({len(enc['input_ids'])}) != dumped "
               f"prompt_token_ids ({len(dumped_ids)}): token spans would not "
               f"match the engine's tokenization")
        if not allow_mismatch:
            raise SystemExit(
                f"error: {msg}. Token offsets are unreliable for this model "
                f"/ tokenizer; investigate before trusting spans, or pass "
                f"--allow-tokenizer-mismatch to proceed with re-encoded offsets."
            )
        print(f"warning: {msg}; proceeding with re-encoded offsets", file=sys.stderr)
    return enc["offset_mapping"]


def char_to_token_span(offsets: list, c0: int, c1: int) -> list[int]:
    """[token_start, token_end) covering char span [c0, c1)."""
    start = None
    end = 0
    for i, (s, e) in enumerate(offsets):
        if s == e:  # zero-width (special token), skip for boundary purposes
            continue
        if e > c0 and start is None:
            start = i
        if s < c1:
            end = i + 1
    return [start if start is not None else 0, end]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--prompts", required=True, nargs="+", type=Path,
                    help="prompt-*.jsonl dump dir(s)/file(s) (DYN_PROMPT_DUMP_TOKENS=1)")
    ap.add_argument("--out", required=True, type=Path, help="output suite dir")
    ap.add_argument("--model", default=None,
                    help="HF id/path for the tokenizer (token spans). Omit for "
                         "text-only spans (no token offsets).")
    ap.add_argument("--config", type=Path, default=None,
                    help="run config.json -> resolve problem_statement to instance_id")
    ap.add_argument("--dedup-key", choices=["content", "path", "path-range"],
                    default="content", help="what counts as 'the same file' (default content)")
    ap.add_argument("--allow-tokenizer-mismatch", action="store_true",
                    help="proceed even if re-encoding prompt_text != dumped "
                         "prompt_token_ids (token spans may be wrong; default errors)")
    args = ap.parse_args()

    records = [r for r in load_records(args.prompts) if is_main_turn(r)]
    if not records:
        print("no main-agent turns found (wrapper sentinel absent)", file=sys.stderr)
        return 1

    inst_map = build_instance_map(args.config) if args.config else {}

    # Group by sample (problem_statement). Within a sample we do NOT trust ts
    # order: the dump carries both the prefill AND decode copy of each request
    # (identical prompt) and the prompt dir accumulates across runs. So we
    # (a) collapse identical prompts (prefer the prefill copy) and (b) link
    # turns by prompt-prefix lineage (chain_links).
    samples: dict[str, list[dict]] = {}
    for r in records:
        ps = extract_problem_statement(r.get("prompt_text") or "")
        key = ps or "_unknown"
        samples.setdefault(key, []).append(r)

    tok = load_tokenizer(args.model) if args.model else None
    args.out.mkdir(parents=True, exist_ok=True)
    index: list[dict] = []
    n_suites = 0

    for ps, group in samples.items():
        sample_id = inst_map.get(ps) or f"sample-{_sha(ps)[:8]}"
        # collapse prefill/decode twins + cross-run identical turns
        uniq: dict[str, dict] = {}
        for r in group:
            pt = r.get("prompt_text") or ""
            prev = uniq.get(pt)
            if prev is None or (prev.get("role") != "prefill" and r.get("role") == "prefill"):
                uniq[pt] = r
        recs = sorted(uniq.values(), key=lambda r: len(r.get("prompt_text") or ""))
        pred, succ = chain_links([r.get("prompt_text") or "" for r in recs])

        seen: set[str] = set()
        read_ord = 0
        for ti, rec in enumerate(recs):
            prompt = rec.get("prompt_text") or ""
            blocks = find_file_blocks(prompt)
            prev_prompt = recs[pred[ti]].get("prompt_text") or "" if pred[ti] is not None else ""
            prev_sha = {b["content_sha"] for b in find_file_blocks(prev_prompt)}
            new_blocks = [b for b in blocks if b["content_sha"] not in prev_sha]
            if not new_blocks:
                continue
            if len(new_blocks) > 1:
                print(f"warning: {sample_id} turn {ti}: {len(new_blocks)} new "
                      f"reads in one turn (parallel read; taking the first)",
                      file=sys.stderr)
            new_block = new_blocks[0]
            dk = dedup_key(new_block, args.dedup_key)
            if dk in seen:
                continue  # re-read of an already-seen file: no duplicate suite
            seen.add(dk)

            if succ[ti] is None:
                print(f"warning: {sample_id} turn {ti}: no successor turn for "
                      f"ground truth (last turn of lineage); skipping", file=sys.stderr)
                continue
            gt = ground_truth_from_next(prompt, recs[succ[ti]].get("prompt_text") or "")
            if not gt:
                print(f"warning: {sample_id} turn {ti}: empty ground truth; skipping",
                      file=sys.stderr)
                continue

            read_ord += 1
            suite_dir = args.out / sample_id / f"read_{read_ord:03d}"
            suite_dir.mkdir(parents=True, exist_ok=True)

            segs = decompose(prompt, blocks)
            offsets = (char_offsets(prompt, tok, rec.get("prompt_token_ids"),
                                    allow_mismatch=args.allow_tokenizer_mismatch)
                       if tok else None)

            manifest_segs = []
            fi = ti_text = 0
            for si, seg in enumerate(segs):
                c0, c1 = seg["char_span"]
                text = prompt[c0:c1]
                if seg["type"] == "file":
                    fi += 1
                    fname = f"file_{fi:02d}.txt"
                else:
                    ti_text += 1
                    fname = f"seg_{si:02d}.txt"
                (suite_dir / fname).write_text(text)
                m = {"i": si, "type": seg["type"], "file": fname,
                     "char_span": [c0, c1], "n_chars": c1 - c0}
                if seg["type"] == "file":
                    m.update(path=seg["path"], line_range=seg["line_range"],
                             content_sha=seg["content_sha"])
                if offsets is not None:
                    tspan = char_to_token_span(offsets, c0, c1)
                    m["token_span"] = tspan
                    m["n_tokens"] = tspan[1] - tspan[0]
                manifest_segs.append(m)

            (suite_dir / "ground_truth.txt").write_text(gt)
            manifest = {
                "sample_id": sample_id,
                "read_index": read_ord,
                "dedup_key": args.dedup_key,
                "turn": {
                    "request_id": rec.get("request_id"),
                    "ts": rec.get("ts"),
                    "num_prompt_tokens": rec.get("num_prompt_tokens"),
                    "n_tokens_encoded": len(offsets) if offsets is not None else None,
                    "lineage_index": ti,
                },
                "new_file": {
                    "path": new_block["path"],
                    "line_range": new_block["line_range"],
                    "content_sha": new_block["content_sha"],
                },
                "n_file_segments": fi,
                "segments": manifest_segs,
                "ground_truth": {"file": "ground_truth.txt", "n_chars": len(gt)},
            }
            (suite_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
            index.append({"sample_id": sample_id, "read_index": read_ord,
                          "dir": str(suite_dir.relative_to(args.out)),
                          "new_file_path": new_block["path"],
                          "n_file_segments": fi})
            n_suites += 1

    (args.out / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {n_suites} suite(s) across {len(samples)} sample(s) -> {args.out}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
