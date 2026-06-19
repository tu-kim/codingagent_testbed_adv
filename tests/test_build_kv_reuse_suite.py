"""Tests for scripts/build_kv_reuse_suite.py.

Pure stdlib, no network, no GPU. Tokenizer-dependent functions (load_tokenizer,
char_offsets) are NOT tested here; they require transformers + a GPU host.
All other public functions are covered.

Conventions mirror test_format_prompt_dump.py / test_extract_predictions.py:
  - module-scope fixture loads the script via importlib
  - tmp_path per test for NDJSON fixture files
  - determinism checked by calling the same function twice with identical input
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_kv_reuse_suite.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("build_kv_reuse_suite", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_kv_reuse_suite"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

WRAPPER = "You are working in a git checkout that has already been cloned"
IM_END = "<|im_end|>"

# Build a minimal opencode file-read block (exactly as read.ts writes it):
# <path>P</path>\n<type>file</type>\n<content>\nBODY\n</content>
def _file_block(path: str, lines: list[str], note: str | None = None) -> str:
    """Build an opencode read-tool <path>/<type>/<content> block.

    The note (e.g. 'End of file - total 2 lines') is wrapped in parentheses
    to match the format that _CONTENT_NOTE_RE expects: '\\n\\n(...)\\s*$'.
    """
    numbered = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
    inner = numbered
    if note:
        inner = inner + f"\n\n({note})"
    return f"<path>{path}</path>\n<type>file</type>\n<content>\n{inner}\n</content>"


def _tool_response(content: str) -> str:
    return f"<tool_response>\n{content}\n</tool_response>"


def _user_msg(body: str) -> str:
    return f"<|im_start|>user\n{body}\n{IM_END}"


def _asst_primer() -> str:
    return "<|im_start|>assistant\n"


def _asst_msg(body: str) -> str:
    """A completed assistant turn (with IM_END)."""
    return f"<|im_start|>assistant\n{body}\n{IM_END}"


def _base_prompt(issue_body: str) -> str:
    """Initial turn prompt: user message with wrapper sentinel + assistant primer."""
    return _user_msg(f"{WRAPPER}\n\n# Issue\n{issue_body}") + "\n" + _asst_primer()


def _extend_prompt(prev_prompt: str, generation: str, tool_response_block: str) -> str:
    """Simulate the engine: append generation+IM_END, tool_response user msg,
    new assistant primer.  prev_prompt IS a strict prefix of the returned value.

    This mirrors the real dynamo dump format where each turn's prompt_text is
    the cumulative chat history up to (and including) the new assistant primer,
    so turn N's prompt_text is always a true prefix of turn N+1's prompt_text.
    """
    return (
        prev_prompt
        + generation
        + IM_END
        + "\n"
        + _user_msg(f"<tool_response>\n{tool_response_block}\n</tool_response>")
        + "\n"
        + _asst_primer()
    )


def _finalize_prompt(prev_prompt: str, generation: str) -> str:
    """Last turn: append generation+IM_END (no next tool response)."""
    return prev_prompt + generation + IM_END


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# ---------------------------------------------------------------------------
# is_main_turn
# ---------------------------------------------------------------------------

def test_is_main_turn_true_with_sentinel(mod):
    """A record whose prompt_text contains WRAPPER_SENTINEL -> True."""
    rec = {"prompt_text": f"preamble\n{WRAPPER}\nmore text"}
    assert mod.is_main_turn(rec) is True


def test_is_main_turn_false_without_sentinel(mod):
    """A title-agent turn without the wrapper -> False."""
    rec = {"prompt_text": "Generate a short title for this task: fix the bug."}
    assert mod.is_main_turn(rec) is False


def test_is_main_turn_false_sub_agent(mod):
    """A sub-agent (task tool) prompt without the wrapper -> False."""
    rec = {"prompt_text": "You are a coding assistant. Do X."}
    assert mod.is_main_turn(rec) is False


def test_is_main_turn_missing_prompt_text(mod):
    """prompt_text absent from dict -> False (no crash)."""
    assert mod.is_main_turn({}) is False


def test_is_main_turn_prompt_text_none(mod):
    """prompt_text=None -> False (no crash)."""
    assert mod.is_main_turn({"prompt_text": None}) is False


def test_is_main_turn_sentinel_exact_match(mod):
    """Sentinel must be a substring; surrounding text is fine."""
    # Sentinel at start
    assert mod.is_main_turn({"prompt_text": WRAPPER}) is True


# ---------------------------------------------------------------------------
# extract_problem_statement
# ---------------------------------------------------------------------------

def test_extract_problem_statement_stops_at_hints(mod):
    """Text after '# Issue' is extracted; '\\n# Hints' is the stop marker."""
    text = f"...preamble...\n# Issue\nThe bug is here.\n# Hints\nhint text"
    result = mod.extract_problem_statement(text)
    assert result == "The bug is here."


def test_extract_problem_statement_stops_at_im_end(mod):
    """Text after '# Issue' is bounded by <|im_end|> when # Hints is absent."""
    text = f"# Issue\nBroken thing.\n{IM_END}\nmore assistant stuff"
    result = mod.extract_problem_statement(text)
    assert result == "Broken thing."


def test_extract_problem_statement_no_issue_returns_none(mod):
    """Prompt with no '# Issue' marker -> None."""
    assert mod.extract_problem_statement("This is a plain prompt.") is None


def test_extract_problem_statement_stripped(mod):
    """Leading/trailing whitespace on the extracted text is stripped."""
    text = "# Issue\n   spaced statement   \n# Hints\nstuff"
    result = mod.extract_problem_statement(text)
    assert result == "spaced statement"


def test_extract_problem_statement_empty_body_returns_none(mod):
    """An '# Issue' followed immediately by the stop token -> None (empty after strip)."""
    text = f"# Issue\n{IM_END}"
    assert mod.extract_problem_statement(text) is None


def test_extract_problem_statement_prefers_hints_over_im_end(mod):
    """When both stop markers exist, the first one encountered wins."""
    text = f"# Issue\nstatement\n# Hints\nhint\n{IM_END}"
    result = mod.extract_problem_statement(text)
    assert result == "statement"


def test_extract_problem_statement_determinism(mod):
    """Same input always returns the same output."""
    text = "# Issue\nfoo bar baz\n# Hints\nhint"
    assert mod.extract_problem_statement(text) == mod.extract_problem_statement(text)


# ---------------------------------------------------------------------------
# strip_content_note
# ---------------------------------------------------------------------------

def test_strip_content_note_removes_end_of_file(mod):
    """'(End of file - total N lines)' trailing note is stripped."""
    body = "1: alpha\n2: beta\n\n(End of file - total 2 lines)"
    result = mod.strip_content_note(body)
    assert result == "1: alpha\n2: beta"


def test_strip_content_note_removes_output_capped(mod):
    """'(Output capped ...)' variant of the note is stripped."""
    body = "1: x\n\n(Output capped at 200 lines)"
    result = mod.strip_content_note(body)
    assert result == "1: x"


def test_strip_content_note_no_note_unchanged(mod):
    """Body without a trailing note is returned unchanged."""
    body = "1: line one\n2: line two"
    assert mod.strip_content_note(body) == body


def test_strip_content_note_only_strips_trailing(mod):
    """A note that is NOT trailing (i.e., embedded mid-body) is not stripped."""
    body = "1: (End of file - total 1 lines)\n2: normal line"
    result = mod.strip_content_note(body)
    # The note pattern requires \n\n before (End ...) and $ after -- mid-line is fine
    assert "normal line" in result


def test_strip_content_note_determinism(mod):
    body = "1: code\n2: more\n\n(End of file - total 2 lines)"
    assert mod.strip_content_note(body) == mod.strip_content_note(body)


# ---------------------------------------------------------------------------
# find_file_blocks
# ---------------------------------------------------------------------------

def test_find_file_blocks_single_block(mod):
    """Single file block: path captured, line range set, body excludes note/tags."""
    block = _file_block("src/foo.py", ["x = 1", "y = 2"],
                        note="End of file - total 2 lines")
    prompt = f"before\n{block}\nafter"

    blocks = mod.find_file_blocks(prompt)

    assert len(blocks) == 1
    b = blocks[0]
    assert b["path"] == "src/foo.py"
    assert b["line_range"] == [1, 2]
    # body extracted must contain the numbered lines
    assert "1: x = 1" in b["body"]
    assert "2: y = 2" in b["body"]
    # body must NOT contain the note text
    assert "End of file" not in b["body"]
    # body must NOT contain the XML tags
    assert "<content>" not in b["body"]
    assert "</content>" not in b["body"]


def test_find_file_blocks_char_span_excludes_tags(mod):
    """char_start/char_end span covers the body, not the wrapping tags."""
    block = _file_block("a.py", ["code"])
    prompt = "PREAMBLE\n" + block + "\nSUFFIX"

    blocks = mod.find_file_blocks(prompt)
    b = blocks[0]
    extracted = prompt[b["char_start"]:b["char_end"]]

    assert "1: code" in extracted
    assert "<path>" not in extracted
    assert "<content>" not in extracted


def test_find_file_blocks_content_sha_set(mod):
    """content_sha is a hex string (SHA1 = 40 chars)."""
    block = _file_block("b.py", ["pass"])
    blocks = mod.find_file_blocks(block)
    assert len(blocks[0]["content_sha"]) == 40


def test_find_file_blocks_multiple_in_order(mod):
    """Multiple file blocks are returned in document order."""
    b1 = _file_block("first.py", ["a = 1"])
    b2 = _file_block("second.py", ["b = 2"])
    prompt = f"intro\n{b1}\nmiddle\n{b2}\nend"

    blocks = mod.find_file_blocks(prompt)

    assert len(blocks) == 2
    assert blocks[0]["path"] == "first.py"
    assert blocks[1]["path"] == "second.py"
    # char_start of second block must be after end of first
    assert blocks[1]["char_start"] > blocks[0]["char_end"]


def test_find_file_blocks_no_blocks_returns_empty(mod):
    """Prompt with no file blocks returns empty list."""
    assert mod.find_file_blocks("just some text, no tool output") == []


def test_find_file_blocks_line_range_from_numbering(mod):
    """line_range reflects the actual N: line numbers in the body."""
    # Simulate a file read starting at line 10
    lines_with_numbers = [f"{i+10}: code {i}" for i in range(3)]
    raw_body = "\n".join(lines_with_numbers)
    # Build the block manually (bypassing _file_block helper which renumbers)
    block = (
        f"<path>partial.py</path>\n<type>file</type>\n<content>\n"
        f"{raw_body}\n</content>"
    )
    blocks = mod.find_file_blocks(block)
    assert blocks[0]["line_range"] == [10, 12]


def test_find_file_blocks_determinism(mod):
    """Same prompt always produces identical block list."""
    block = _file_block("x.py", ["a", "b", "c"])
    prompt = f"text\n{block}\nmore"
    r1 = mod.find_file_blocks(prompt)
    r2 = mod.find_file_blocks(prompt)
    assert len(r1) == len(r2)
    assert r1[0]["path"] == r2[0]["path"]
    assert r1[0]["content_sha"] == r2[0]["content_sha"]


# ---------------------------------------------------------------------------
# dedup_key
# ---------------------------------------------------------------------------

def test_dedup_key_content_mode(mod):
    """content mode returns content_sha."""
    block = {"content_sha": "abc123", "path": "foo.py", "line_range": [1, 10]}
    assert mod.dedup_key(block, "content") == "abc123"


def test_dedup_key_path_mode(mod):
    """path mode returns the path string."""
    block = {"content_sha": "abc123", "path": "foo.py", "line_range": [1, 10]}
    assert mod.dedup_key(block, "path") == "foo.py"


def test_dedup_key_path_range_mode(mod):
    """path-range mode returns 'path@[a, b]'."""
    block = {"content_sha": "abc123", "path": "foo.py", "line_range": [5, 20]}
    result = mod.dedup_key(block, "path-range")
    assert result == "foo.py@[5, 20]"


def test_dedup_key_content_differs_by_hash(mod):
    """Two blocks with different content_sha produce different content keys."""
    b1 = {"content_sha": "aaa", "path": "f.py", "line_range": [1, 1]}
    b2 = {"content_sha": "bbb", "path": "f.py", "line_range": [1, 1]}
    assert mod.dedup_key(b1, "content") != mod.dedup_key(b2, "content")


def test_dedup_key_path_range_same_path_different_range(mod):
    """Same path + different line_range -> different path-range key."""
    b1 = {"content_sha": "x", "path": "f.py", "line_range": [1, 5]}
    b2 = {"content_sha": "x", "path": "f.py", "line_range": [6, 10]}
    assert mod.dedup_key(b1, "path-range") != mod.dedup_key(b2, "path-range")


# ---------------------------------------------------------------------------
# ground_truth_from_next
# ---------------------------------------------------------------------------

def test_ground_truth_from_next_simple_prefix(mod):
    """When next_prompt == cur_prompt + gen + IM_END + rest, returns gen."""
    cur = "SYS\nUSER\nASST_PRIMER"
    gen = "I will read the file first."
    next_prompt = cur + gen + IM_END + "\nmore turns"
    result = mod.ground_truth_from_next(cur, next_prompt)
    assert result == gen


def test_ground_truth_from_next_no_im_end_returns_full_tail(mod):
    """When IM_END is absent in the tail, returns the entire tail."""
    cur = "base"
    gen = "assistant response without end marker"
    next_prompt = cur + gen
    result = mod.ground_truth_from_next(cur, next_prompt)
    assert result == gen


def test_ground_truth_from_next_empty_tail_returns_none(mod):
    """When next_prompt == cur_prompt exactly, tail is empty -> None."""
    prompt = "identical text"
    assert mod.ground_truth_from_next(prompt, prompt) is None


def test_ground_truth_from_next_fallback_longest_common_prefix(mod):
    """Non-prefix case: falls back to longest common prefix, then extracts tail."""
    # cur and next share a prefix but diverge (simulates re-encoding drift)
    cur = "AAABBBCCC"
    # Next diverges at char 6 ('C' vs 'X')
    next_prompt = "AAABBBXXX" + IM_END + "rest"
    result = mod.ground_truth_from_next(cur, next_prompt)
    # tail starts from divergence point; IM_END cuts off "rest"
    assert result is not None
    assert IM_END not in result
    assert "XXX" in result


def test_ground_truth_from_next_gen_contains_tool_call(mod):
    """Generation text can contain tool call XML; extracted correctly up to IM_END."""
    cur = "SYS\n"
    gen = 'I will call read.\n<tool_call>{"name":"read","input":{"path":"a.py"}}</tool_call>'
    next_prompt = cur + gen + IM_END + "\nnext_user_turn"
    result = mod.ground_truth_from_next(cur, next_prompt)
    assert result == gen


def test_ground_truth_from_next_determinism(mod):
    """Same (cur, next) always produces the same result."""
    cur = "prefix\n"
    gen = "generation text"
    nxt = cur + gen + IM_END
    r1 = mod.ground_truth_from_next(cur, nxt)
    r2 = mod.ground_truth_from_next(cur, nxt)
    assert r1 == r2


# ---------------------------------------------------------------------------
# chain_links
# ---------------------------------------------------------------------------

def test_chain_links_empty_list(mod):
    """Empty input -> empty pred/succ lists."""
    pred, succ = mod.chain_links([])
    assert pred == []
    assert succ == []


def test_chain_links_single_prompt(mod):
    """Single prompt -> pred=[None], succ=[None] (no self-link)."""
    pred, succ = mod.chain_links(["ABCDE"])
    assert pred == [None]
    assert succ == [None]


def test_chain_links_linear_chain(mod):
    """p0 < p1 < p2 (each a strict prefix of the next) -> pred=[None,0,1] succ=[1,2,None]."""
    p0 = "ABC"
    p1 = "ABCDE"
    p2 = "ABCDEFGH"
    pred, succ = mod.chain_links([p0, p1, p2])
    assert pred == [None, 0, 1], f"pred={pred}"
    assert succ == [1, 2, None], f"succ={succ}"


def test_chain_links_linear_chain_determinism(mod):
    """Same input always yields identical (pred, succ)."""
    prompts = ["X", "XY", "XYZ"]
    r1 = mod.chain_links(prompts)
    r2 = mod.chain_links(prompts)
    assert r1 == r2


def test_chain_links_branch(mod):
    """p1 base; p2 and p2b both strictly extend p1 but not each other.

    pred: p2->p1 (idx 0), p2b->p1 (idx 0).
    succ: p1 -> the SHORTEST extending prompt = p2 (shorter len).
    p2 and p2b are leaves (succ=None).
    """
    p1 = "ABC"
    p2 = "ABCDE"    # len 5 -- shorter extension
    p2b = "ABCXYZ"  # len 6 -- longer extension, NOT a prefix of p2
    # Order: p1=idx0, p2=idx1, p2b=idx2
    pred, succ = mod.chain_links([p1, p2, p2b])
    # pred
    assert pred[0] is None, f"pred[p1]={pred[0]}"
    assert pred[1] == 0,    f"pred[p2]={pred[1]}"  # p1 is pred of p2
    assert pred[2] == 0,    f"pred[p2b]={pred[2]}" # p1 is pred of p2b
    # succ[p1] = shorter of {p2,p2b} = p2 (idx 1)
    assert succ[0] == 1,    f"succ[p1]={succ[0]}"
    # p2 and p2b have no successors
    assert succ[1] is None, f"succ[p2]={succ[1]}"
    assert succ[2] is None, f"succ[p2b]={succ[2]}"


def test_chain_links_unrelated_prompts(mod):
    """Prompts that share no prefix relationship -> all pred/succ None."""
    pred, succ = mod.chain_links(["ABC", "XYZ", "PQR"])
    assert pred == [None, None, None]
    assert succ == [None, None, None]


def test_chain_links_equal_length_distinct_are_siblings(mod):
    """Equal-length distinct prompts are not linked (neither is a strict prefix of the other)."""
    pred, succ = mod.chain_links(["ABCDE", "VWXYZ"])
    assert pred == [None, None]
    assert succ == [None, None]


def test_chain_links_pred_is_longest_prefix(mod):
    """When multiple shorter prompts are prefixes of p, pred picks the LONGEST one."""
    # p0="A", p1="ABC", p2="ABCDE" -- p2 startswith p0 AND p1; pred[p2] must be p1 (longest)
    p0 = "A"
    p1 = "ABC"
    p2 = "ABCDE"
    pred, succ = mod.chain_links([p0, p1, p2])
    # pred[2] must be 1 (p1, the longest prefix), not 0 (p0)
    assert pred[2] == 1, f"pred[p2]={pred[2]}, expected 1 (p1 is longest prefix)"


def test_chain_links_succ_is_shortest_extension(mod):
    """When multiple longer prompts extend p, succ picks the SHORTEST one."""
    p0 = "AB"
    p1 = "ABCD"    # len 4 -- shorter extension
    p2 = "ABCDEFG" # len 7 -- longer extension
    pred, succ = mod.chain_links([p0, p1, p2])
    # succ[0] must be 1 (p1, shortest extension)
    assert succ[0] == 1, f"succ[p0]={succ[0]}, expected 1 (p1 is shortest extension)"


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------

def _make_blocks_from_prompt(mod, prompt: str) -> list[dict]:
    return mod.find_file_blocks(prompt)


def test_decompose_one_file_three_segments(mod):
    """One file block in the middle -> [text, file, text] with types alternating."""
    file_blk = _file_block("a.py", ["code"])
    prompt = "BEFORE\n" + file_blk + "\nAFTER"
    blocks = mod.find_file_blocks(prompt)

    segs = mod.decompose(prompt, blocks)

    types = [s["type"] for s in segs]
    assert types == ["text", "file", "text"]
    assert segs[1]["path"] == "a.py"


def test_decompose_two_files_five_segments(mod):
    """Two file blocks -> [text, file, text, file, text]."""
    b1 = _file_block("x.py", ["x = 1"])
    b2 = _file_block("y.py", ["y = 2"])
    prompt = "start\n" + b1 + "\nmiddle\n" + b2 + "\nend"
    blocks = mod.find_file_blocks(prompt)

    segs = mod.decompose(prompt, blocks)

    types = [s["type"] for s in segs]
    assert types == ["text", "file", "text", "file", "text"]


def test_decompose_file_body_at_offset_zero_no_empty_text_seg(mod):
    """When the file body char_span starts at 0 (injected directly), no leading
    empty text segment is emitted.  decompose skips leading text segments whose
    span is empty (char_start == cursor == 0)."""
    # Synthesise a block dict whose body char_span starts at position 0 in a
    # prompt that IS exactly the body text (no wrapping tags in the prompt).
    body_text = "1: code"
    # Build the block manually: char_start=0, char_end=len(body_text)
    block = {
        "path": "a.py",
        "body": body_text,
        "char_start": 0,
        "char_end": len(body_text),
        "line_range": [1, 1],
        "content_sha": "a" * 40,
    }
    prompt = body_text + "\nAFTER"

    segs = mod.decompose(prompt, [block])

    # First segment is the file (body at 0), not an empty text segment
    assert segs[0]["type"] == "file"
    # Trailing text follows
    assert len(segs) == 2
    assert segs[1]["type"] == "text"


def test_decompose_char_spans_contiguous_and_cover_prompt(mod):
    """Segment char_spans are contiguous and together cover [0, len(prompt))."""
    b1 = _file_block("f1.py", ["a"])
    b2 = _file_block("f2.py", ["b"])
    prompt = "intro\n" + b1 + "\nmid\n" + b2 + "\nend"
    blocks = mod.find_file_blocks(prompt)

    segs = mod.decompose(prompt, blocks)

    # Contiguous: each seg starts where previous ended
    prev_end = 0
    for seg in segs:
        c0, c1 = seg["char_span"]
        assert c0 == prev_end, f"gap before segment {seg}"
        prev_end = c1
    # Covers entire prompt
    assert prev_end == len(prompt)


def test_decompose_no_blocks_returns_single_text_seg(mod):
    """Prompt with no file blocks -> single text segment covering the whole prompt."""
    prompt = "no files here at all"
    segs = mod.decompose(prompt, [])
    assert len(segs) == 1
    assert segs[0]["type"] == "text"
    assert segs[0]["char_span"] == [0, len(prompt)]


def test_decompose_file_seg_has_path_line_range_sha(mod):
    """File segments carry path, line_range, content_sha."""
    b = _file_block("z.py", ["line one", "line two"])
    prompt = "A\n" + b + "\nB"
    blocks = mod.find_file_blocks(prompt)
    segs = mod.decompose(prompt, blocks)
    file_seg = next(s for s in segs if s["type"] == "file")
    assert file_seg["path"] == "z.py"
    assert file_seg["line_range"] == [1, 2]
    assert len(file_seg["content_sha"]) == 40


def test_decompose_determinism(mod):
    """Same prompt + blocks always yields identical segment list."""
    b = _file_block("d.py", ["x"])
    prompt = "pre\n" + b + "\npost"
    blocks = mod.find_file_blocks(prompt)
    r1 = mod.decompose(prompt, blocks)
    r2 = mod.decompose(prompt, blocks)
    assert [(s["type"], s["char_span"]) for s in r1] == [(s["type"], s["char_span"]) for s in r2]


# ---------------------------------------------------------------------------
# char_to_token_span
# ---------------------------------------------------------------------------

def _make_offsets(pairs: list[tuple[int, int]]) -> list:
    """Build a list matching what HF tokenizer returns as offset_mapping."""
    return list(pairs)


def test_char_to_token_span_basic(mod):
    """Basic span covering a range within the token list."""
    # tokens cover: [0,2) [2,5) [5,9) [9,12)
    offsets = [(0, 2), (2, 5), (5, 9), (9, 12)]
    span = mod.char_to_token_span(offsets, 2, 9)
    # token 1 covers [2,5) -> start; token 2 covers [5,9) -> end=3
    assert span == [1, 3]


def test_char_to_token_span_skips_zero_width(mod):
    """Zero-width (0,0) entries (special tokens) are skipped for boundary purposes."""
    offsets = [(0, 2), (2, 5), (0, 0), (5, 9), (9, 12)]
    span = mod.char_to_token_span(offsets, 2, 9)
    # (0,0) at index 2 is skipped; token 1 [2,5) is start; token 3 [5,9) -> end=4
    assert span[0] == 1
    assert span[1] == 4


def test_char_to_token_span_entire_sequence(mod):
    """Span covering all chars returns [0, n_tokens)."""
    offsets = [(0, 3), (3, 6), (6, 10)]
    span = mod.char_to_token_span(offsets, 0, 10)
    assert span == [0, 3]


def test_char_to_token_span_single_token(mod):
    """Char span that fits within exactly one token -> single-token span."""
    offsets = [(0, 5), (5, 10), (10, 15)]
    span = mod.char_to_token_span(offsets, 5, 10)
    assert span == [1, 2]


def test_char_to_token_span_all_zero_width(mod):
    """When all tokens are zero-width, returns [0, 0]."""
    offsets = [(0, 0), (0, 0)]
    span = mod.char_to_token_span(offsets, 0, 5)
    assert span == [0, 0]


def test_char_to_token_span_determinism(mod):
    """Same inputs always return the same span."""
    offsets = [(0, 2), (2, 5), (5, 9), (9, 12)]
    r1 = mod.char_to_token_span(offsets, 2, 9)
    r2 = mod.char_to_token_span(offsets, 2, 9)
    assert r1 == r2


# ---------------------------------------------------------------------------
# load_records
# ---------------------------------------------------------------------------

def test_load_records_reads_valid_ndjson(mod, tmp_path):
    """Valid NDJSON records are parsed and returned."""
    d = tmp_path / "prompts"
    d.mkdir()
    _write_jsonl(d / "prompt-001.jsonl", [
        {"prompt_text": f"{WRAPPER} turn 1", "ts": 1.0},
        {"prompt_text": f"{WRAPPER} turn 2", "ts": 2.0},
    ])

    records = mod.load_records([d])

    assert len(records) == 2
    assert records[0]["ts"] == 1.0


def test_load_records_globs_only_prompt_files(mod, tmp_path):
    """Only prompt-*.jsonl files are loaded; other files are ignored."""
    d = tmp_path / "prompts"
    d.mkdir()
    _write_jsonl(d / "prompt-001.jsonl", [{"ts": 1.0}])
    _write_jsonl(d / "other.jsonl", [{"ts": 99.0}])
    (d / "not_json.txt").write_text("ignored")

    records = mod.load_records([d])

    assert len(records) == 1
    assert records[0]["ts"] == 1.0


def test_load_records_skips_blank_lines(mod, tmp_path):
    """Blank / whitespace-only lines are silently skipped."""
    f = tmp_path / "prompt-skip.jsonl"
    f.write_text(
        json.dumps({"ts": 1.0}) + "\n"
        "\n"
        "   \n"
        + json.dumps({"ts": 2.0}) + "\n"
    )

    records = mod.load_records([f])
    assert len(records) == 2


def test_load_records_skips_malformed_lines(mod, tmp_path):
    """Malformed JSON lines are skipped (no crash, no count returned publicly)."""
    f = tmp_path / "prompt-bad.jsonl"
    f.write_text(
        "NOT JSON\n"
        + json.dumps({"ts": 1.0}) + "\n"
        + "{broken\n"
        + json.dumps({"ts": 2.0}) + "\n"
    )

    records = mod.load_records([f])
    # Only the two valid records survive
    assert len(records) == 2
    assert records[0]["ts"] == 1.0
    assert records[1]["ts"] == 2.0


def test_load_records_multiple_files(mod, tmp_path):
    """Multiple files passed as a list are all loaded."""
    f1 = tmp_path / "prompt-a.jsonl"
    f2 = tmp_path / "prompt-b.jsonl"
    _write_jsonl(f1, [{"ts": 1.0}])
    _write_jsonl(f2, [{"ts": 2.0}])

    records = mod.load_records([f1, f2])
    assert len(records) == 2


def test_load_records_missing_path_skipped(mod, tmp_path, capsys):
    """A path that does not exist emits a warning to stderr and is skipped."""
    nonexistent = tmp_path / "does_not_exist"
    records = mod.load_records([nonexistent])
    captured = capsys.readouterr()
    assert records == []
    assert "warning" in captured.err.lower() or "not found" in captured.err.lower()


def test_load_records_empty_dir(mod, tmp_path):
    """A directory with no prompt-*.jsonl files -> empty list."""
    d = tmp_path / "empty"
    d.mkdir()
    assert mod.load_records([d]) == []


# ---------------------------------------------------------------------------
# Integration: synthetic trace -> main() end-to-end (text-only, no --model)
#
# Scenario design (reality-based: each turn's prompt = prev + gen + IM_END + ...):
#
# Run 1 (4 turns):
#   turn 0 (t0):      wrapper + issue, NO file reads -- no suite
#   turn 1 (t1):      t0 + gen0 + foo.py tool response + primer
#                     -> new read: foo.py -- SUITE read_001
#   turn 2 (t2_run1): t1 + gen1 + bar.py tool response + primer
#                     -> new read: bar.py (foo already in pred) -- SUITE read_002
#   turn 3 (t3_run1): t2_run1 + gen2 + IM_END  [final, provides gt for read_002]
#
# Run 2 (diverges after t1, re-reads foo.py with same content):
#   turn 0 (t0):      IDENTICAL to run1 t0 -> deduped to one record (prefill kept)
#   turn 1 (t1):      IDENTICAL to run1 t1 -> deduped to one record (prefill kept)
#   turn 2 (t2_run2): t1 + gen1_run2 + foo.py (again, same sha) + primer
#                     -> foo.py's content_sha already in 'seen' -> DEDUP SKIP
#   turn 3 (t3_run2): t2_run2 + gen2_run2 + IM_END  [final]
#
# Each turn has BOTH a prefill and a decode record (same prompt_text, different
# request_id) to exercise twin dedup: decode twins must be dropped, leaving only
# the prefill record.
#
# Expected suites: exactly 2 (foo.py from run1-t1, bar.py from run1-t2).
# read_002 must have n_file_segments == 2 (cumulative: foo + bar in the prompt).
# ground_truth for read_001 = gen1 (what the model said after reading foo.py).
# ground_truth for read_002 = gen2_run1 (what the model said after reading bar.py).
# ---------------------------------------------------------------------------

def _build_twin_dump(tmp_path: Path) -> tuple[Path, str, str, str]:
    """Build a prompt-*.jsonl with prefill+decode twins across two runs.

    Returns (prompts_dir, gt_foo, gt_bar, issue).
    """
    issue = "The frobnicate function raises ValueError on empty input."
    foo_block_str = _file_block(
        "/repo/foo.py", ["def frobnicate(x):", "    return x * 2"],
        note="End of file - total 2 lines"
    )
    bar_block_str = _file_block(
        "/repo/bar.py", ["def helper():", "    pass"],
        note="End of file - total 2 lines"
    )

    # Cumulative prompts (real dynamo dump format):
    t0 = _base_prompt(issue)
    gen0 = (
        "Let me read foo.py.\n"
        '<tool_call>{"name":"read","input":{"path":"/repo/foo.py"}}</tool_call>'
    )
    t1 = _extend_prompt(t0, gen0, foo_block_str)

    # Run 1 turn 2: reads bar.py
    gen1_run1 = (
        "Now let me read bar.py.\n"
        '<tool_call>{"name":"read","input":{"path":"/repo/bar.py"}}</tool_call>'
    )
    t2_run1 = _extend_prompt(t1, gen1_run1, bar_block_str)

    # Run 1 turn 3: final, provides ground truth for bar suite
    gen2_run1 = "The fix is clear. Applying the patch now."
    t3_run1 = _finalize_prompt(t2_run1, gen2_run1)

    # Run 2 turn 2: diverges -- re-reads foo.py (SAME content_sha)
    gen1_run2 = (
        "Let me re-examine foo.py more carefully.\n"
        '<tool_call>{"name":"read","input":{"path":"/repo/foo.py"}}</tool_call>'
    )
    t2_run2 = _extend_prompt(t1, gen1_run2, foo_block_str)

    # Run 2 turn 3: final
    gen2_run2 = "OK, understood now."
    t3_run2 = _finalize_prompt(t2_run2, gen2_run2)

    # ground_truth for each suite (what the script will compute):
    #   foo suite: ground_truth_from_next(t1, t2_run1) -> gen1_run1 (up to IM_END)
    #   bar suite: ground_truth_from_next(t2_run1, t3_run1) -> gen2_run1 (up to IM_END)
    gt_foo = gen1_run1
    gt_bar = gen2_run1

    # Build records: 2 per turn (prefill + decode twin), across both runs.
    # t0 and t1 are shared (identical prompt_text) -- the dedup must collapse them.
    def _twin(prompt_text, request_id_base, ts):
        return [
            {"prompt_text": prompt_text, "request_id": f"{request_id_base}-p",
             "role": "prefill", "ts": ts},
            {"prompt_text": prompt_text, "request_id": f"{request_id_base}-d",
             "role": "decode",   "ts": ts + 0.001},
        ]

    records = []
    # Run 1
    records += _twin(t0,       "r1-t0", ts=1.0)
    records += _twin(t1,       "r1-t1", ts=2.0)
    records += _twin(t2_run1,  "r1-t2", ts=3.0)
    records += _twin(t3_run1,  "r1-t3", ts=4.0)
    # Run 2 (t0 and t1 repeat; t2/t3 are distinct)
    records += _twin(t0,       "r2-t0", ts=5.0)
    records += _twin(t1,       "r2-t1", ts=6.0)
    records += _twin(t2_run2,  "r2-t2", ts=7.0)
    records += _twin(t3_run2,  "r2-t3", ts=8.0)

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    _write_jsonl(prompts_dir / "prompt-001.jsonl", records)
    return prompts_dir, gt_foo, gt_bar, issue


def test_main_integration_exactly_two_suites(mod, tmp_path, monkeypatch):
    """main() produces exactly 2 suites despite prefill/decode twins and a
    cross-run foo.py re-read (deduped by content_sha)."""
    prompts_dir, _gt_foo, _gt_bar, _issue = _build_twin_dump(tmp_path)
    out_dir = tmp_path / "suite_out"

    monkeypatch.setattr(sys, "argv", [
        "build_kv_reuse_suite",
        "--prompts", str(prompts_dir),
        "--out", str(out_dir),
    ])
    rc = mod.main()
    assert rc == 0

    index = json.loads((out_dir / "index.json").read_text())
    assert len(index) == 2, f"expected 2 suites, got {len(index)}: {index}"


def test_main_integration_correct_file_paths(mod, tmp_path, monkeypatch):
    """Suites are for /repo/foo.py (first read) and /repo/bar.py (second);
    the run-2 foo.py re-read does NOT appear as a third suite."""
    prompts_dir, _gt_foo, _gt_bar, _issue = _build_twin_dump(tmp_path)
    out_dir = tmp_path / "suite_out"

    monkeypatch.setattr(sys, "argv", [
        "build_kv_reuse_suite", "--prompts", str(prompts_dir), "--out", str(out_dir),
    ])
    mod.main()

    index = json.loads((out_dir / "index.json").read_text())
    paths = [e["new_file_path"] for e in index]
    assert paths.count("/repo/foo.py") == 1, f"foo.py should appear once: {paths}"
    assert paths.count("/repo/bar.py") == 1, f"bar.py should appear once: {paths}"


def test_main_integration_prefill_decode_twins_dont_break_ground_truth(mod, tmp_path, monkeypatch):
    """With prefill+decode twins in the dump, ground_truth.txt must still be
    non-empty for both suites (the twin collapse must not lose the successor turn)."""
    prompts_dir, _gt_foo, _gt_bar, _issue = _build_twin_dump(tmp_path)
    out_dir = tmp_path / "suite_out"

    monkeypatch.setattr(sys, "argv", [
        "build_kv_reuse_suite", "--prompts", str(prompts_dir), "--out", str(out_dir),
    ])
    mod.main()

    index = json.loads((out_dir / "index.json").read_text())
    for entry in index:
        gt_path = out_dir / entry["dir"] / "ground_truth.txt"
        gt_text = gt_path.read_text()
        assert gt_text.strip(), (
            f"ground_truth.txt is empty for {entry['new_file_path']} -- "
            "twin dedup likely consumed the successor turn"
        )


def test_main_integration_ground_truth_contents(mod, tmp_path, monkeypatch):
    """ground_truth.txt for each suite contains the correct model generation
    (the delta between consecutive unique prompts, up to IM_END)."""
    prompts_dir, gt_foo, gt_bar, _issue = _build_twin_dump(tmp_path)
    out_dir = tmp_path / "suite_out"

    monkeypatch.setattr(sys, "argv", [
        "build_kv_reuse_suite", "--prompts", str(prompts_dir), "--out", str(out_dir),
    ])
    mod.main()

    index = json.loads((out_dir / "index.json").read_text())
    foo_entry = next(e for e in index if e["new_file_path"] == "/repo/foo.py")
    bar_entry = next(e for e in index if e["new_file_path"] == "/repo/bar.py")

    written_gt_foo = (out_dir / foo_entry["dir"] / "ground_truth.txt").read_text()
    written_gt_bar = (out_dir / bar_entry["dir"] / "ground_truth.txt").read_text()

    assert written_gt_foo == gt_foo, (
        f"foo gt mismatch:\n  got: {repr(written_gt_foo[:80])}\n"
        f"  want: {repr(gt_foo[:80])}"
    )
    assert written_gt_bar == gt_bar, (
        f"bar gt mismatch:\n  got: {repr(written_gt_bar[:80])}\n"
        f"  want: {repr(gt_bar[:80])}"
    )


def test_main_integration_read_002_has_two_file_segments(mod, tmp_path, monkeypatch):
    """read_002 (bar.py suite) covers the CUMULATIVE prompt with both foo.py and
    bar.py in it -> n_file_segments == 2 in the index entry."""
    prompts_dir, _gt_foo, _gt_bar, _issue = _build_twin_dump(tmp_path)
    out_dir = tmp_path / "suite_out"

    monkeypatch.setattr(sys, "argv", [
        "build_kv_reuse_suite", "--prompts", str(prompts_dir), "--out", str(out_dir),
    ])
    mod.main()

    index = json.loads((out_dir / "index.json").read_text())
    bar_entry = next(e for e in index if e["new_file_path"] == "/repo/bar.py")
    assert bar_entry["n_file_segments"] == 2, (
        f"bar.py suite should have 2 cumulative file segments, got "
        f"{bar_entry['n_file_segments']}"
    )


def test_main_integration_title_turn_filtered(mod, tmp_path, monkeypatch):
    """A title-agent turn (no wrapper sentinel) in the dump is NOT included in any suite."""
    prompts_dir, _gt_foo, _gt_bar, _issue = _build_twin_dump(tmp_path)
    out_dir = tmp_path / "suite_out"

    # Inject a title-agent record into the existing file
    existing = (prompts_dir / "prompt-001.jsonl").read_text()
    title_rec = json.dumps({"ts": 0.1, "request_id": "title-r",
                            "prompt_text": "Generate a short title for: frobnicate bug."})
    (prompts_dir / "prompt-001.jsonl").write_text(title_rec + "\n" + existing)

    monkeypatch.setattr(sys, "argv", [
        "build_kv_reuse_suite", "--prompts", str(prompts_dir), "--out", str(out_dir),
    ])
    rc = mod.main()
    assert rc == 0

    index = json.loads((out_dir / "index.json").read_text())
    # Still exactly 2 suites (title turn contributes nothing)
    assert len(index) == 2
    # All suites belong to one sample (the issue)
    sample_ids = {e["sample_id"] for e in index}
    assert len(sample_ids) == 1


def test_main_integration_no_main_turns_returns_1(mod, tmp_path, monkeypatch):
    """When all records lack the wrapper sentinel, main() returns 1 (no suites)."""
    prompts_dir = tmp_path / "empty_prompts"
    prompts_dir.mkdir()
    _write_jsonl(prompts_dir / "prompt-001.jsonl", [
        {"ts": 1.0, "prompt_text": "Just a title turn"},
        {"ts": 2.0, "prompt_text": "Another non-main turn"},
    ])
    out_dir = tmp_path / "suite_out"

    monkeypatch.setattr(sys, "argv", [
        "build_kv_reuse_suite",
        "--prompts", str(prompts_dir),
        "--out", str(out_dir),
    ])
    rc = mod.main()
    assert rc == 1


def test_main_integration_manifest_lineage_index_field(mod, tmp_path, monkeypatch):
    """manifest.json uses the 'lineage_index' field (not the old 'ts_index')."""
    prompts_dir, _gt_foo, _gt_bar, _issue = _build_twin_dump(tmp_path)
    out_dir = tmp_path / "suite_out"

    monkeypatch.setattr(sys, "argv", [
        "build_kv_reuse_suite", "--prompts", str(prompts_dir), "--out", str(out_dir),
    ])
    mod.main()

    index = json.loads((out_dir / "index.json").read_text())
    for entry in index:
        manifest = json.loads((out_dir / entry["dir"] / "manifest.json").read_text())
        assert "lineage_index" in manifest["turn"], (
            "manifest.turn must have 'lineage_index' (not the old 'ts_index')"
        )
        assert "ts_index" not in manifest["turn"], (
            "manifest.turn must NOT have the old 'ts_index' field"
        )
