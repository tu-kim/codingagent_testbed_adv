"""Tests for scripts/analyze_prompt_kv_composition.py.

Conventions mirrored from tests/test_sweagent_apps.py (importlib.util module
loader with the module registered in sys.modules BEFORE exec_module --
dataclass introspection breaks on some interpreters otherwise). No network,
no GPU: this script only reads local NDJSON files and does regex/string
math, so every test operates on in-repo tmp_path fixtures.

The --tokenizer CLI path (needs `transformers`) is NOT exercised here per
the module owner's guidance; TokenMapper's "tok" mode IS covered by handing
it a stub tokenizer callable directly (TokenMapper takes any object with a
`tokenizer(text, return_offsets_mapping=..., add_special_tokens=...)`
signature -- no transformers import happens unless main()'s --tokenizer
flag is used).
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (Path(__file__).resolve().parents[1] / "scripts"
                / "analyze_prompt_kv_composition.py")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module("analyze_prompt_kv_composition", _SCRIPT_PATH)


# ===========================================================================
# 1. file_char_intervals
# ===========================================================================

def test_file_char_intervals_read_tool_block_matched(mod):
    text = "<type>file</type>\n<content>\n1: hello\n2: world\n</content>"
    spans = mod.file_char_intervals(text)
    assert spans == [(0, len(text))]


def test_file_char_intervals_directory_variant_not_matched(mod):
    text = ("<type>directory</type>\n<entries>\nfoo.py\nbar.py\n</entries>")
    assert mod.file_char_intervals(text) == []


def test_file_char_intervals_xml_param_matched_filepath_excluded(mod):
    text = (
        "<tool_call>\n<function=write>\n<parameter=filePath>\n/tmp/x.py\n"
        "</parameter>\n<parameter=content>\nprint(1)\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    spans = mod.file_char_intervals(text)
    assert len(spans) == 1
    s, e = spans[0]
    assert text[s:e] == "print(1)"
    # The filePath value must NOT be part of any matched interval.
    assert "/tmp/x.py" not in text[s:e]
    fp_start = text.index("/tmp/x.py")
    fp_end = fp_start + len("/tmp/x.py")
    assert not (s < fp_end and fp_start < e)  # no overlap with filePath span


def test_file_char_intervals_hermes_json_fallback_matched(mod):
    payload = {"name": "write", "arguments": {"content": "line1\nline2"}}
    text = "<tool_call>" + json.dumps(payload) + "</tool_call>"
    spans = mod.file_char_intervals(text)
    assert len(spans) == 1
    s, e = spans[0]
    # The interval covers the JSON-ENCODED value (escapes intact), not the
    # decoded string -- encoded "\n" stays as a literal backslash-n.
    encoded = json.dumps("line1\nline2")[1:-1]
    assert text[s:e] == encoded
    assert "\\n" in text[s:e]


def test_file_char_intervals_hermes_json_ignores_non_file_bearing_tool(mod):
    payload = {"name": "read", "arguments": {"filePath": "/tmp/x.py"}}
    text = "<tool_call>" + json.dumps(payload) + "</tool_call>"
    assert mod.file_char_intervals(text) == []


def test_file_char_intervals_overlapping_spans_merged(mod):
    # A read-tool block whose own file content happens to literally contain
    # a <parameter=content> marker: the inner XML-param span is fully
    # contained inside the outer read-block span, so after sort+merge only
    # ONE interval should remain, covering the entire text.
    text = ("<type>file</type>\n<content>\n1: <parameter=content>\nabc\n"
            "</parameter>\n</content>")
    spans = mod.file_char_intervals(text)
    assert spans == [(0, len(text))]


def test_file_char_intervals_empty_text(mod):
    assert mod.file_char_intervals("") == []


# ===========================================================================
# 2. chars_in_region
# ===========================================================================

def test_chars_in_region_straddling_boundaries(mod):
    intervals = [(5, 15)]
    assert mod.chars_in_region(intervals, 8, 12) == 4    # fully inside
    assert mod.chars_in_region(intervals, 10, 20) == 5   # straddles right end
    assert mod.chars_in_region(intervals, 0, 8) == 3     # straddles left end


def test_chars_in_region_touching_boundary_excluded(mod):
    # [0,5) vs interval [5,15): half-open ranges touch but don't overlap.
    assert mod.chars_in_region([(5, 15)], 0, 5) == 0


def test_chars_in_region_no_overlap(mod):
    assert mod.chars_in_region([(5, 15)], 0, 3) == 0
    assert mod.chars_in_region([(5, 15)], 20, 30) == 0


def test_chars_in_region_multiple_intervals_summed(mod):
    intervals = [(0, 5), (10, 20), (25, 30)]
    assert mod.chars_in_region(intervals, 0, 30) == 5 + 10 + 5


# ===========================================================================
# 3. _lcp_len
# ===========================================================================

def test_lcp_len_identical_strings(mod):
    assert mod._lcp_len("hello", "hello") == 5


def test_lcp_len_empty_inputs(mod):
    assert mod._lcp_len("", "abc") == 0
    assert mod._lcp_len("abc", "") == 0
    assert mod._lcp_len("", "") == 0


def test_lcp_len_no_common_prefix(mod):
    assert mod._lcp_len("abc", "xyz") == 0


def test_lcp_len_binary_search_correctness_on_long_strings(mod):
    # 10k+ char strings, identical up to a known index, then diverge --
    # exercises the binary-search implementation (not a naive char loop).
    a = "P" * 5000 + "Q" * 5000
    b = "P" * 5000 + "R" * 5000
    assert mod._lcp_len(a, b) == 5000


def test_lcp_len_determinism(mod):
    a = "P" * 5000 + "Q" * 5000
    b = "P" * 5000 + "R" * 5000
    assert mod._lcp_len(a, b) == mod._lcp_len(a, b) == 5000


# ===========================================================================
# 4. _lcp_tokens
# ===========================================================================

def test_lcp_tokens_basic(mod):
    assert mod._lcp_tokens([1, 2, 3], [1, 2, 4]) == 2
    assert mod._lcp_tokens([1, 2, 3], [1, 2, 3]) == 3
    assert mod._lcp_tokens([], [1, 2]) == 0
    assert mod._lcp_tokens([1, 2], []) == 0


# ===========================================================================
# 5. load_dump
# ===========================================================================

def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_load_dump_missing_dir_raises_systemexit(mod, tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(SystemExit):
        mod.load_dump(missing)


def test_load_dump_skips_records_without_prompt_text(mod, tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    _write_jsonl(prompts_dir / "prompt-1.jsonl", [
        {"ts": 1.0, "request_id": "a", "role": "decode",
         "num_prompt_tokens": 10, "prompt_text": "hello"},
        {"ts": 2.0, "request_id": "b", "role": "decode",
         "num_prompt_tokens": 10},  # no prompt_text key at all
        {"ts": 3.0, "request_id": "c", "role": "decode",
         "num_prompt_tokens": 10, "prompt_text": ""},  # empty, falsy
    ])

    recs = mod.load_dump(prompts_dir)

    assert len(recs) == 1
    assert recs[0].request_id == "a"


def test_load_dump_dedup_prefers_prefill_decode_then_prefill_order(mod, tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    # File-sort order: prompt-1.jsonl (decode) read BEFORE prompt-2.jsonl
    # (prefill) for the SAME request_id.
    _write_jsonl(prompts_dir / "prompt-1.jsonl", [
        {"ts": 1.0, "request_id": "x", "role": "decode",
         "num_prompt_tokens": 5, "prompt_text": "decode-text"},
    ])
    _write_jsonl(prompts_dir / "prompt-2.jsonl", [
        {"ts": 2.0, "request_id": "x", "role": "prefill",
         "num_prompt_tokens": 5, "prompt_text": "prefill-text"},
    ])

    recs = mod.load_dump(prompts_dir)

    assert len(recs) == 1
    assert recs[0].role == "prefill"
    assert recs[0].text == "prefill-text"


def test_load_dump_dedup_prefers_prefill_prefill_then_decode_order(mod, tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    # Reversed file-sort order: prefill read FIRST, decode SECOND -- prefill
    # must still win (dedup is role-preference, not "first wins").
    _write_jsonl(prompts_dir / "prompt-1.jsonl", [
        {"ts": 1.0, "request_id": "y", "role": "prefill",
         "num_prompt_tokens": 5, "prompt_text": "prefill-text2"},
    ])
    _write_jsonl(prompts_dir / "prompt-2.jsonl", [
        {"ts": 2.0, "request_id": "y", "role": "decode",
         "num_prompt_tokens": 5, "prompt_text": "decode-text2"},
    ])

    recs = mod.load_dump(prompts_dir)

    assert len(recs) == 1
    assert recs[0].role == "prefill"
    assert recs[0].text == "prefill-text2"


def test_load_dump_merges_multiple_files_and_sorts_by_ts(mod, tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    # prompt-1.jsonl (read first, alphabetically) holds the LATER ts;
    # prompt-2.jsonl (read second) holds the EARLIER ts -- proves the
    # final ordering is a genuine ts-sort, not just file/insertion order.
    _write_jsonl(prompts_dir / "prompt-1.jsonl", [
        {"ts": 5.0, "request_id": "later", "role": "decode",
         "num_prompt_tokens": 1, "prompt_text": "t-later"},
    ])
    _write_jsonl(prompts_dir / "prompt-2.jsonl", [
        {"ts": 1.0, "request_id": "earlier", "role": "decode",
         "num_prompt_tokens": 1, "prompt_text": "t-earlier"},
    ])

    recs = mod.load_dump(prompts_dir)

    assert [r.request_id for r in recs] == ["earlier", "later"]
    assert [r.ts for r in recs] == [1.0, 5.0]


# ===========================================================================
# 6. chain_sessions
# ===========================================================================

def test_chain_sessions_growing_prefix_stays_one_session_ordered_by_ts(mod):
    base = "A" * 250
    t2 = base + "B" * 50
    t3 = t2 + "C" * 50
    recs = [
        mod.PromptRec(ts=1.0, request_id="r1", role="decode", num_tokens=10,
                     text=base, token_ids=None),
        mod.PromptRec(ts=2.0, request_id="r2", role="decode", num_tokens=10,
                     text=t2, token_ids=None),
        mod.PromptRec(ts=3.0, request_id="r3", role="decode", num_tokens=10,
                     text=t3, token_ids=None),
    ]

    chains = mod.chain_sessions(recs, min_frac=0.5, min_chars=200)

    assert len(chains) == 1
    assert len(chains[0].records) == 3
    # Turn order within the chain must track ts order.
    assert [r.ts for r in chains[0].records] == [1.0, 2.0, 3.0]
    assert [r.request_id for r in chains[0].records] == ["r1", "r2", "r3"]


def test_chain_sessions_unrelated_prompt_starts_new_session(mod):
    base = "A" * 250
    t2 = base + "B" * 50
    unrelated = "Z" * 300  # shares 0 chars with t2
    recs = [
        mod.PromptRec(ts=1.0, request_id="r1", role="decode", num_tokens=10,
                     text=base, token_ids=None),
        mod.PromptRec(ts=2.0, request_id="r2", role="decode", num_tokens=10,
                     text=t2, token_ids=None),
        mod.PromptRec(ts=3.0, request_id="r3", role="decode", num_tokens=10,
                     text=unrelated, token_ids=None),
    ]

    chains = mod.chain_sessions(recs, min_frac=0.5, min_chars=200)

    assert len(chains) == 2
    sizes = sorted(len(c.records) for c in chains)
    assert sizes == [1, 2]


def test_chain_sessions_shared_system_prompt_only_below_min_frac_splits(mod):
    # Two prompts share a 300-char system preamble (>= min_chars=200) but
    # each total prompt is 1000 chars, so the shared fraction is 0.3 --
    # below min_frac=0.5 -- and they must NOT be chained together.
    sys_prompt = "S" * 300
    rec_a = sys_prompt + "T" * 700
    rec_b = sys_prompt + "U" * 700
    recs = [
        mod.PromptRec(ts=1.0, request_id="a", role="decode", num_tokens=10,
                     text=rec_a, token_ids=None),
        mod.PromptRec(ts=2.0, request_id="b", role="decode", num_tokens=10,
                     text=rec_b, token_ids=None),
    ]

    chains = mod.chain_sessions(recs, min_frac=0.5, min_chars=200)

    assert len(chains) == 2
    assert all(len(c.records) == 1 for c in chains)


def test_chain_sessions_determinism(mod):
    base = "A" * 250
    t2 = base + "B" * 50
    recs = [
        mod.PromptRec(ts=1.0, request_id="r1", role="decode", num_tokens=10,
                     text=base, token_ids=None),
        mod.PromptRec(ts=2.0, request_id="r2", role="decode", num_tokens=10,
                     text=t2, token_ids=None),
    ]

    chains1 = mod.chain_sessions(recs, min_frac=0.5, min_chars=200)
    chains2 = mod.chain_sessions(recs, min_frac=0.5, min_chars=200)

    assert [len(c.records) for c in chains1] == [len(c.records) for c in chains2]
    assert [c.session_id for c in chains1] == [c.session_id for c in chains2]


# ===========================================================================
# 7. TokenMapper
# ===========================================================================

def test_token_mapper_mode_ids_when_token_ids_present(mod):
    rec = mod.PromptRec(ts=0, request_id="r", role="decode", num_tokens=10,
                        text="hello world", token_ids=[1, 2, 3])
    mapper = mod.TokenMapper(rec)
    assert mapper.mode == "ids"


def test_token_mapper_mode_chars_when_token_ids_none(mod):
    rec = mod.PromptRec(ts=0, request_id="r", role="decode", num_tokens=10,
                        text="hello world", token_ids=None)
    mapper = mod.TokenMapper(rec)
    assert mapper.mode == "chars"


def test_token_mapper_mode_chars_when_token_ids_empty_list(mod):
    rec = mod.PromptRec(ts=0, request_id="r", role="decode", num_tokens=10,
                        text="hello world", token_ids=[])
    mapper = mod.TokenMapper(rec)
    assert mapper.mode == "chars"


def test_token_mapper_mode_tok_with_stub_tokenizer(mod):
    def fake_tokenizer(text, return_offsets_mapping=True,
                       add_special_tokens=False):
        offsets = []
        idx = 0
        for word in text.split(" "):
            start = text.index(word, idx)
            offsets.append((start, start + len(word)))
            idx = start + len(word)
        return {"offset_mapping": offsets}

    rec = mod.PromptRec(ts=0, request_id="r", role="decode", num_tokens=999,
                        text="hello world foo", token_ids=None)
    mapper = mod.TokenMapper(rec, tokenizer=fake_tokenizer)

    assert mapper.mode == "tok"
    assert mapper.total() == 3  # 3 whitespace-delimited words
    assert mapper.region_tokens(0, 5) == 1.0    # "hello" only
    assert mapper.region_tokens(0, 11) == 2.0   # "hello world"


def test_token_mapper_chars_mode_region_tokens_proportional(mod):
    rec = mod.PromptRec(ts=0, request_id="r", role="decode", num_tokens=40,
                        text="X" * 100, token_ids=None)
    mapper = mod.TokenMapper(rec)

    assert mapper.mode == "chars"
    assert mapper.region_tokens(0, 50) == pytest.approx(20.0)
    assert mapper.region_tokens(0, 100) == pytest.approx(40.0) == mapper.total()


def test_token_mapper_region_tokens_degenerate_range_is_zero(mod):
    rec = mod.PromptRec(ts=0, request_id="r", role="decode", num_tokens=40,
                        text="X" * 100, token_ids=None)
    mapper = mod.TokenMapper(rec)

    assert mapper.region_tokens(10, 5) == 0.0   # end < start
    assert mapper.region_tokens(5, 5) == 0.0    # end == start


# ===========================================================================
# 8. main() end-to-end
# ===========================================================================

def _read_csv_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def test_main_end_to_end_three_turn_file_content_progression(mod, tmp_path):
    """turn1: plain text, no file content. turn2: adds a read-tool block
    (new file content, none of it in the shared prefix yet). turn3: adds an
    XML write tool-call (both the inherited read-block content AND the new
    write-call content are file content)."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    filler = "SYS " * 100  # long, fixed preamble so LCP thresholds pass
    turn1 = filler + "\nUser: please look at foo.py"
    turn2 = turn1 + (
        "\nAssistant: <type>file</type>\n<content>\n1: print('hi')\n"
        "</content>"
    )
    turn3 = turn2 + (
        "\nAssistant: <tool_call>\n<function=write>\n"
        "<parameter=content>\nprint('new')\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    records = [
        {"ts": 1.0, "request_id": "req-1", "role": "decode",
         "num_prompt_tokens": len(turn1) // 4, "prompt_text": turn1},
        {"ts": 2.0, "request_id": "req-2", "role": "decode",
         "num_prompt_tokens": len(turn2) // 4, "prompt_text": turn2},
        {"ts": 3.0, "request_id": "req-3", "role": "decode",
         "num_prompt_tokens": len(turn3) // 4, "prompt_text": turn3},
    ]
    _write_jsonl(prompts_dir / "prompt-1.jsonl", records)

    out_csv = tmp_path / "out.csv"
    rc = mod.main(["--prompts", str(prompts_dir), "--out", str(out_csv)])

    assert rc == 0
    rows = _read_csv_rows(out_csv)
    assert len(rows) == 3
    # All three turns must chain into a single session (append-only growth).
    assert {r["session_id"] for r in rows} == {"ses-001"}

    row1 = next(r for r in rows if r["turn"] == "1")
    row2 = next(r for r in rows if r["turn"] == "2")
    row3 = next(r for r in rows if r["turn"] == "3")

    assert float(row1["prefix_total"]) == pytest.approx(0.0)

    assert float(row2["prefix_file"]) == pytest.approx(0.0)
    assert float(row2["new_file"]) > 0.0

    assert float(row3["prefix_file"]) > 0.0
    assert float(row3["new_file"]) > 0.0


def test_main_end_to_end_ids_mode_prefix_total_is_exact_lcp(mod, tmp_path):
    """When prompt_token_ids are present, prefix_total must be the EXACT
    token-level LCP, not the chars-proportional estimate (which would give
    a materially different number here: 300/350 * 7 ~= 6.0 vs the true
    token LCP of 3)."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    common = "X" * 300
    text1 = common
    text2 = common + "Y" * 50

    records = [
        {"ts": 1.0, "request_id": "r1", "role": "prefill",
         "num_prompt_tokens": 5, "prompt_text": text1,
         "prompt_token_ids": [1, 2, 3, 4, 5]},
        {"ts": 2.0, "request_id": "r2", "role": "prefill",
         "num_prompt_tokens": 7, "prompt_text": text2,
         "prompt_token_ids": [1, 2, 3, 9, 9, 9, 9]},
    ]
    _write_jsonl(prompts_dir / "prompt-1.jsonl", records)

    out_csv = tmp_path / "out.csv"
    rc = mod.main(["--prompts", str(prompts_dir), "--out", str(out_csv)])

    assert rc == 0
    rows = _read_csv_rows(out_csv)
    assert len(rows) == 2
    row2 = next(r for r in rows if r["turn"] == "2")

    assert row2["mode"] == "ids"
    assert float(row2["prefix_total"]) == pytest.approx(3.0)


# ===========================================================================
# 9. Title-generation filter (opencode's per-session title one-shot;
#    TITLE_MARKER = "Generate a title for this conversation:", injected by
#    opencode/packages/opencode/src/session/prompt.ts:211). Dropped BEFORE
#    chaining by default; --keep-title retains it.
# ===========================================================================

def _title_and_main_chain_dump(prompts_dir: Path) -> None:
    """One title-generation request (ts=0.5, its own unrelated text) plus a
    genuine 2-turn main chain (common 300-char prefix + a 50-char
    extension, well past the default min_chars/min_frac thresholds)."""
    common = "M" * 300
    main1 = common
    main2 = common + "N" * 50
    title_text = "Generate a title for this conversation:\nUser: fix the bug"

    records = [
        {"ts": 0.5, "request_id": "title-1", "role": "decode",
         "num_prompt_tokens": 10, "prompt_text": title_text},
        {"ts": 1.0, "request_id": "main-1", "role": "decode",
         "num_prompt_tokens": 60, "prompt_text": main1},
        {"ts": 2.0, "request_id": "main-2", "role": "decode",
         "num_prompt_tokens": 70, "prompt_text": main2},
    ]
    _write_jsonl(prompts_dir / "prompt-1.jsonl", records)


def test_main_default_drops_title_request_keeps_two_turn_chain(mod, tmp_path,
                                                                capsys):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    _title_and_main_chain_dump(prompts_dir)

    out_csv = tmp_path / "out.csv"
    rc = mod.main(["--prompts", str(prompts_dir), "--out", str(out_csv)])

    assert rc == 0
    rows = _read_csv_rows(out_csv)
    assert len(rows) == 2
    assert {r["session_id"] for r in rows} == {"ses-001"}
    assert [r["turn"] for r in rows] == ["1", "2"]
    assert [r["request_id"] for r in rows] == ["main-1", "main-2"]

    out = capsys.readouterr().out
    assert "dropped 1 title-generation request(s)" in out


def test_main_keep_title_yields_three_rows(mod, tmp_path, capsys):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    _title_and_main_chain_dump(prompts_dir)

    out_csv = tmp_path / "out.csv"
    rc = mod.main(["--prompts", str(prompts_dir), "--out", str(out_csv),
                  "--keep-title"])

    assert rc == 0
    rows = _read_csv_rows(out_csv)
    assert len(rows) == 3
    assert {r["request_id"] for r in rows} == {"title-1", "main-1", "main-2"}

    # Nothing was dropped, so the drop-count line must not print.
    out = capsys.readouterr().out
    assert "dropped" not in out


def test_main_default_drops_title_request_marker_mid_text(mod, tmp_path,
                                                           capsys):
    """The marker is matched by substring containment (`TITLE_MARKER not in
    r.text`), not by position -- a title request with real preamble text
    BEFORE the marker must still be dropped."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    mid_text = ("Some system preamble text here padding padding padding.\n"
               "Generate a title for this conversation:\nUser: xyz")
    _write_jsonl(prompts_dir / "prompt-1.jsonl", [
        {"ts": 1.0, "request_id": "title-mid", "role": "decode",
         "num_prompt_tokens": 10, "prompt_text": mid_text},
    ])

    out_csv = tmp_path / "out.csv"
    rc = mod.main(["--prompts", str(prompts_dir), "--out", str(out_csv)])

    assert rc == 0
    rows = _read_csv_rows(out_csv)
    assert len(rows) == 0

    out = capsys.readouterr().out
    assert "dropped 1 title-generation request(s)" in out
