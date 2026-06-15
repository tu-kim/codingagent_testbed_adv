"""Tests for scripts/format_prompt_dump.py.

Pure stdlib, no network, no GPU. All functions are tested via fixtures in
tmp_path — no subprocess, no real prompt dump files required.

Conventions mirror test_extract_predictions.py / test_analyze_eval_results.py:
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


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "format_prompt_dump.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("format_prompt_dump", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["format_prompt_dump"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as NDJSON to path."""
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _make_record(
    request_id: str = "req-001",
    role: str = "prefill",
    ts: float = 1_700_000_000.0,
    num_prompt_tokens: int = 100,
    prompt_text: str | None = "Hello\nworld",
    decode_error: str | None = None,
) -> dict:
    r: dict = {
        "ts": ts,
        "request_id": request_id,
        "role": role,
        "num_prompt_tokens": num_prompt_tokens,
        "prompt_text": prompt_text,
    }
    if decode_error is not None:
        r["decode_error"] = decode_error
    return r


# ---------------------------------------------------------------------------
# load_records
# ---------------------------------------------------------------------------

def test_load_records_reads_valid_ndjson(mod, tmp_path):
    """load_records parses valid NDJSON lines and tags each record with _source."""
    f = tmp_path / "prompt-123.jsonl"
    _write_jsonl(f, [_make_record("r1"), _make_record("r2")])

    records, n_bad = mod.load_records([f])

    assert n_bad == 0
    assert len(records) == 2
    assert records[0]["request_id"] == "r1"
    assert records[1]["request_id"] == "r2"
    assert records[0]["_source"] == "prompt-123.jsonl"
    assert records[1]["_source"] == "prompt-123.jsonl"


def test_load_records_skips_blank_lines(mod, tmp_path):
    """Blank lines (whitespace-only) are ignored and do not increment n_bad."""
    f = tmp_path / "prompt-1.jsonl"
    f.write_text(
        json.dumps(_make_record("r1")) + "\n"
        "\n"
        "   \n"
        + json.dumps(_make_record("r2")) + "\n"
    )

    records, n_bad = mod.load_records([f])

    assert n_bad == 0
    assert len(records) == 2


def test_load_records_counts_malformed_lines_as_bad(mod, tmp_path):
    """A malformed JSON line increments n_bad and is not included in records."""
    f = tmp_path / "prompt-2.jsonl"
    f.write_text(
        json.dumps(_make_record("r1")) + "\n"
        "NOT JSON {{{\n"
        + json.dumps(_make_record("r2")) + "\n"
    )

    records, n_bad = mod.load_records([f])

    assert n_bad == 1
    assert len(records) == 2
    assert records[0]["request_id"] == "r1"
    assert records[1]["request_id"] == "r2"


def test_load_records_multiple_bad_lines(mod, tmp_path):
    """Each malformed line independently increments n_bad."""
    f = tmp_path / "prompt-3.jsonl"
    f.write_text("BAD\nALSOBAD\n" + json.dumps(_make_record("r1")) + "\n")

    records, n_bad = mod.load_records([f])

    assert n_bad == 2
    assert len(records) == 1


def test_load_records_globs_dir(mod, tmp_path):
    """Passing a directory globs prompt-*.jsonl files within it."""
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    _write_jsonl(prompts_dir / "prompt-100.jsonl", [_make_record("r1")])
    _write_jsonl(prompts_dir / "prompt-200.jsonl", [_make_record("r2")])
    # Should NOT be picked up (wrong name pattern)
    (prompts_dir / "other.jsonl").write_text(json.dumps(_make_record("r3")) + "\n")

    records, n_bad = mod.load_records([prompts_dir])

    assert n_bad == 0
    assert len(records) == 2
    assert {r["request_id"] for r in records} == {"r1", "r2"}


def test_load_records_empty_dir(mod, tmp_path):
    """A directory with no prompt-*.jsonl files returns empty records, zero bad."""
    d = tmp_path / "empty"
    d.mkdir()

    records, n_bad = mod.load_records([d])

    assert records == []
    assert n_bad == 0


def test_load_records_mixed_file_and_dir(mod, tmp_path):
    """Can pass both a directory and an explicit file in the same call."""
    d = tmp_path / "prompts"
    d.mkdir()
    _write_jsonl(d / "prompt-10.jsonl", [_make_record("r1")])
    explicit = tmp_path / "extra.jsonl"
    _write_jsonl(explicit, [_make_record("r2")])

    records, n_bad = mod.load_records([d, explicit])

    assert n_bad == 0
    request_ids = {r["request_id"] for r in records}
    assert "r1" in request_ids
    assert "r2" in request_ids


def test_load_records_tags_source_with_filename(mod, tmp_path):
    """_source is the filename (not the full path) of the originating file."""
    f = tmp_path / "prompt-999.jsonl"
    _write_jsonl(f, [_make_record("r1")])

    records, _ = mod.load_records([f])

    assert records[0]["_source"] == "prompt-999.jsonl"


# ---------------------------------------------------------------------------
# dedup_turns
# ---------------------------------------------------------------------------

def test_dedup_auto_collapses_prefill_decode_pair(mod):
    """auto mode: a prefill+decode pair with same request_id → one turn (prefill)."""
    prefill_rec = _make_record("req-1", role="prefill")
    decode_rec = _make_record("req-1", role="decode")

    result = mod.dedup_turns([prefill_rec, decode_rec], "auto")

    assert len(result) == 1
    assert result[0]["role"] == "prefill"


def test_dedup_auto_prefers_prefill_when_decode_arrives_first(mod):
    """auto: decode record arrives in records list before prefill; prefill wins."""
    decode_rec = _make_record("req-x", role="decode")
    prefill_rec = _make_record("req-x", role="prefill")

    result = mod.dedup_turns([decode_rec, prefill_rec], "auto")

    assert len(result) == 1
    assert result[0]["role"] == "prefill"


def test_dedup_auto_keeps_decode_when_no_prefill(mod):
    """auto: if only a decode record exists for a request_id, it is kept."""
    decode_rec = _make_record("req-only-decode", role="decode")

    result = mod.dedup_turns([decode_rec], "auto")

    assert len(result) == 1
    assert result[0]["role"] == "decode"


def test_dedup_auto_distinct_request_ids_all_kept(mod):
    """auto: two different request_ids produce two output records."""
    r1 = _make_record("req-A", role="prefill")
    r2 = _make_record("req-B", role="prefill")

    result = mod.dedup_turns([r1, r2], "auto")

    assert len(result) == 2


def test_dedup_role_both_keeps_all_records(mod):
    """role='both': prefill+decode pair with same request_id → two records."""
    prefill_rec = _make_record("req-1", role="prefill")
    decode_rec = _make_record("req-1", role="decode")

    result = mod.dedup_turns([prefill_rec, decode_rec], "both")

    assert len(result) == 2
    roles = {r["role"] for r in result}
    assert roles == {"prefill", "decode"}


def test_dedup_role_prefill_filters_to_prefill_only(mod):
    """role='prefill': only records with role='prefill' are returned."""
    records = [
        _make_record("req-1", role="prefill"),
        _make_record("req-1", role="decode"),
        _make_record("req-2", role="decode"),
        _make_record("req-3", role="prefill"),
    ]

    result = mod.dedup_turns(records, "prefill")

    assert all(r["role"] == "prefill" for r in result)
    assert len(result) == 2


def test_dedup_role_decode_filters_to_decode_only(mod):
    """role='decode': only records with role='decode' are returned."""
    records = [
        _make_record("req-1", role="prefill"),
        _make_record("req-1", role="decode"),
        _make_record("req-2", role="prefill"),
    ]

    result = mod.dedup_turns(records, "decode")

    assert all(r["role"] == "decode" for r in result)
    assert len(result) == 1


def test_dedup_determinism(mod):
    """Same input always produces identical output (dedup is deterministic)."""
    records = [
        _make_record("req-1", role="prefill"),
        _make_record("req-1", role="decode"),
        _make_record("req-2", role="decode"),
        _make_record("req-3", role="prefill"),
    ]

    result_a = mod.dedup_turns(records, "auto")
    result_b = mod.dedup_turns(records, "auto")

    assert [r["request_id"] for r in result_a] == [r["request_id"] for r in result_b]
    assert [r["role"] for r in result_a] == [r["role"] for r in result_b]


# ---------------------------------------------------------------------------
# _new_suffix
# ---------------------------------------------------------------------------

def test_new_suffix_empty_prev(mod):
    """When prev is empty, the entire cur string is new (cut=0)."""
    cur = "Hello\nworld\n"
    new_text, shared = mod._new_suffix("", cur)

    assert new_text == cur
    assert shared == 0


def test_new_suffix_identical_strings(mod):
    """When prev == cur, the common prefix is the whole string; backed up to
    the last newline means new_text is empty (cur ends with \n, so cut == len(s))."""
    s = "Hello\nworld\n"
    new_text, shared = mod._new_suffix(s, s)

    # last \n in s is at index 11; cut = 12 = len(s); new_text = ""
    assert new_text == ""
    assert shared == len(s)


def test_new_suffix_appended_tail(mod):
    """When cur is prev plus appended text, the suffix is the appended text."""
    prev = "line1\nline2\n"
    append = "line3\nline4\n"
    cur = prev + append

    new_text, shared = mod._new_suffix(prev, cur)

    # Shared prefix is at least len(prev); backed up to newline means cut is at
    # least at the previous newline — in this case exactly len(prev) because
    # prev ends with \n.
    assert new_text == append
    assert shared == len(prev)


def test_new_suffix_line_boundary_backup(mod):
    """Divergence mid-line backs up to the start of that line."""
    prev = "AAA\nBBB_old"
    cur = "AAA\nBBB_new\nCCC\n"

    new_text, shared = mod._new_suffix(prev, cur)

    # Common prefix is "AAA\nBBB" (7 chars); last \n before cut is at index 3
    # so cut = 4 and new_text starts from "BBB_new\nCCC\n"
    assert new_text.startswith("BBB_new")
    assert shared == 4  # "AAA\n" = 4 chars


def test_new_suffix_no_newline_in_common(mod):
    """When there is no newline in the common prefix, cut falls back to 0."""
    prev = "ABCD"
    cur = "ABXY"

    new_text, shared = mod._new_suffix(prev, cur)

    # common prefix is "AB" (2 chars); no \n in [0:2], so cut=0, full cur is new
    assert new_text == cur
    assert shared == 0


def test_new_suffix_determinism(mod):
    """Same (prev, cur) always yields identical (new_text, shared_chars)."""
    prev = "alpha\nbeta\ngamma\n"
    cur = prev + "delta\nepsilon\n"

    a = mod._new_suffix(prev, cur)
    b = mod._new_suffix(prev, cur)

    assert a == b


# ---------------------------------------------------------------------------
# _fmt_ts
# ---------------------------------------------------------------------------

def test_fmt_ts_float_yields_iso_z(mod):
    """A float timestamp returns an ISO8601 string ending in 'Z'."""
    ts = 1_700_000_000.0
    result = mod._fmt_ts(ts)

    assert result.endswith("Z")
    assert "T" in result
    assert "2023" in result  # 2023-11-14


def test_fmt_ts_int_yields_iso_z(mod):
    """An int (also valid as float) returns an ISO string."""
    result = mod._fmt_ts(1_700_000_000)

    assert result.endswith("Z")


def test_fmt_ts_string_number_works(mod):
    """A numeric string is parsed as float and produces ISO output."""
    result = mod._fmt_ts("1700000000.0")

    assert result.endswith("Z")


def test_fmt_ts_non_numeric_string_passthrough(mod):
    """A non-numeric string falls through to str() and is returned unchanged."""
    result = mod._fmt_ts("not-a-timestamp")

    assert result == "not-a-timestamp"


def test_fmt_ts_none_passthrough(mod):
    """None falls through to str(None) = 'None'."""
    result = mod._fmt_ts(None)

    assert result == "None"


def test_fmt_ts_determinism(mod):
    """Same ts always yields the same string."""
    ts = 1_700_123_456.789

    assert mod._fmt_ts(ts) == mod._fmt_ts(ts)


def test_fmt_ts_has_millisecond_precision(mod):
    """The returned string has millisecond precision (3 decimal digits before Z)."""
    result = mod._fmt_ts(1_700_000_000.123)

    # Format: ...YYYY-MM-DDTHH:MM:SS.mmmZ — last char before Z is the 3rd decimal
    assert result.endswith("Z")
    body = result[:-1]  # strip Z
    frac_part = body.split(".")[-1]
    assert len(frac_part) == 3, f"expected 3 decimal digits, got: {frac_part!r}"


# ---------------------------------------------------------------------------
# render — newline fidelity
# ---------------------------------------------------------------------------

def test_render_newlines_are_real_not_escaped(mod):
    """prompt_text with real newlines renders as real newlines, NOT as the
    two-character literal backslash-n."""
    rec = _make_record("req-1", role="prefill", prompt_text="line one\nline two\nline three")
    rec["_source"] = "prompt-1.jsonl"

    out = mod.render([rec], None, delta=False, max_chars=0)

    # The literal two-character sequence \n must NOT appear in the body
    assert "\\n" not in out
    # Real line content must be present
    assert "line one" in out
    assert "line two" in out
    assert "line three" in out


def test_render_body_lines_are_separated(mod):
    """The rendered output contains separate lines from the prompt_text."""
    text = "alpha\nbeta\ngamma"
    rec = _make_record("req-1", prompt_text=text)
    rec["_source"] = "p.jsonl"

    out = mod.render([rec], None, delta=False, max_chars=0)

    lines = out.splitlines()
    # At least one line must be exactly "alpha", "beta", or "gamma"
    assert "alpha" in lines
    assert "beta" in lines
    assert "gamma" in lines


# ---------------------------------------------------------------------------
# render — delta mode
# ---------------------------------------------------------------------------

def test_render_delta_first_turn_is_full(mod):
    """In delta mode the first turn prints the full prompt (with the label)."""
    rec = _make_record("req-1", ts=1.0, prompt_text="full content here")
    rec["_source"] = "p.jsonl"

    out = mod.render([rec], None, delta=True, max_chars=0)

    assert "first turn: full prompt" in out
    assert "full content here" in out


def test_render_delta_second_turn_shows_suffix_only(mod):
    """In delta mode the second turn shows only the new suffix, not the full prompt."""
    common = "system prompt\nuser turn 1\nassistant turn 1\n"
    appended = "user turn 2\nassistant turn 2\n"
    rec1 = _make_record("req-1", ts=1.0, prompt_text=common)
    rec2 = _make_record("req-2", ts=2.0, prompt_text=common + appended)
    rec1["_source"] = rec2["_source"] = "p.jsonl"

    out = mod.render([rec1, rec2], None, delta=True, max_chars=0)

    # The second turn must show the appended tail
    assert "user turn 2" in out
    # The first turn's common prefix must not be duplicated in the second turn's body
    # (it will appear once in the first turn's full output, but the delta header
    # confirms only new chars are printed for turn 2)
    assert "new this turn" in out


def test_render_no_delta_full_prompt_each_turn(mod):
    """Without delta both turns contain the full prompt_text."""
    common = "shared prefix\n"
    rec1 = _make_record("req-1", ts=1.0, prompt_text=common)
    rec2 = _make_record("req-2", ts=2.0, prompt_text=common + "extra\n")
    rec1["_source"] = rec2["_source"] = "p.jsonl"

    out = mod.render([rec1, rec2], None, delta=False, max_chars=0)

    # "shared prefix" must appear at least twice (once per turn)
    assert out.count("shared prefix") >= 2


# ---------------------------------------------------------------------------
# render — prompt_text=None (no prompt_text / decode_error)
# ---------------------------------------------------------------------------

def test_render_none_prompt_text_with_decode_error(mod):
    """prompt_text=None + decode_error → <no prompt_text: <error>> in output."""
    rec = _make_record(
        "req-1",
        prompt_text=None,
        decode_error="tokenizer unavailable",
    )
    rec["_source"] = "p.jsonl"

    out = mod.render([rec], None, delta=False, max_chars=0)

    assert "<no prompt_text:" in out
    assert "tokenizer unavailable" in out


def test_render_none_prompt_text_no_decode_error(mod):
    """prompt_text=None without decode_error falls back to the default message."""
    rec = _make_record("req-1", prompt_text=None)
    rec.pop("decode_error", None)
    rec["_source"] = "p.jsonl"

    out = mod.render([rec], None, delta=False, max_chars=0)

    assert "<no prompt_text:" in out
    # The default message references the env var name
    assert "DYN_PROMPT_DUMP" in out or "no prompt_text" in out


# ---------------------------------------------------------------------------
# render — max_chars truncation
# ---------------------------------------------------------------------------

def test_render_max_chars_truncates_body(mod):
    """max_chars > 0 truncates the body and appends a truncation notice."""
    long_text = "A" * 500
    rec = _make_record("req-1", prompt_text=long_text)
    rec["_source"] = "p.jsonl"

    out = mod.render([rec], None, delta=False, max_chars=100)

    assert "truncated" in out
    # The body should not contain 500 A's
    assert "A" * 500 not in out


def test_render_max_chars_zero_means_no_truncation(mod):
    """max_chars=0 disables truncation."""
    long_text = "B" * 1000
    rec = _make_record("req-1", prompt_text=long_text)
    rec["_source"] = "p.jsonl"

    out = mod.render([rec], None, delta=False, max_chars=0)

    assert "truncated" not in out
    assert "B" * 1000 in out


# ---------------------------------------------------------------------------
# render — session grouping via smap
# ---------------------------------------------------------------------------

def test_render_smap_groups_by_session(mod):
    """With a session map, records from different sessions are grouped separately."""
    rec1 = _make_record("req-A", ts=1.0, prompt_text="session alpha")
    rec2 = _make_record("req-B", ts=2.0, prompt_text="session beta")
    rec1["_source"] = rec2["_source"] = "p.jsonl"

    smap = {"req-A": "ses_alpha", "req-B": "ses_beta"}
    out = mod.render([rec1, rec2], smap, delta=False, max_chars=0)

    assert "session=ses_alpha" in out
    assert "session=ses_beta" in out


def test_render_no_smap_no_session_prefix(mod):
    """Without a session map, the 'session=...' tag does not appear."""
    rec = _make_record("req-1", prompt_text="content")
    rec["_source"] = "p.jsonl"

    out = mod.render([rec], None, delta=False, max_chars=0)

    assert "session=" not in out


# ---------------------------------------------------------------------------
# render — header fields
# ---------------------------------------------------------------------------

def test_render_header_contains_role_and_request_id(mod):
    """The turn header includes role= and req= fields."""
    rec = _make_record("req-HEADER-TEST", role="prefill", prompt_text="hi")
    rec["_source"] = "p.jsonl"

    out = mod.render([rec], None, delta=False, max_chars=0)

    assert "role=prefill" in out
    assert "req=req-HEADER-TEST" in out


def test_render_header_contains_ts_formatted(mod):
    """The meta line includes a formatted ts ending in Z."""
    rec = _make_record("req-1", ts=1_700_000_000.0, prompt_text="hi")
    rec["_source"] = "p.jsonl"

    out = mod.render([rec], None, delta=False, max_chars=0)

    assert "ts=" in out
    # The ISO timestamp should appear in the output
    assert "2023" in out  # 2023-11-14 from epoch 1700000000


def test_render_header_contains_token_count(mod):
    """The meta line includes tokens= field."""
    rec = _make_record("req-1", num_prompt_tokens=42, prompt_text="hi")
    rec["_source"] = "p.jsonl"

    out = mod.render([rec], None, delta=False, max_chars=0)

    assert "tokens=42" in out


# ---------------------------------------------------------------------------
# render — sort by ts
# ---------------------------------------------------------------------------

def test_render_sorts_turns_by_ts(mod):
    """Turns are sorted by ts (ascending) within a session regardless of input order."""
    rec_early = _make_record("req-early", ts=1.0, prompt_text="first")
    rec_late = _make_record("req-late", ts=99.0, prompt_text="second")
    rec_early["_source"] = rec_late["_source"] = "p.jsonl"

    # Pass in reverse order
    out = mod.render([rec_late, rec_early], None, delta=False, max_chars=0)

    first_pos = out.index("first")
    second_pos = out.index("second")
    assert first_pos < second_pos, "early ts record should be rendered before late ts"


# ---------------------------------------------------------------------------
# render — determinism
# ---------------------------------------------------------------------------

def test_render_determinism(mod):
    """Same turns + smap always produces identical output."""
    records = [
        _make_record("req-1", ts=1.0, prompt_text="alpha\nbeta"),
        _make_record("req-2", ts=2.0, prompt_text="alpha\nbeta\ngamma"),
    ]
    for r in records:
        r["_source"] = "p.jsonl"
    smap = {"req-1": "ses_A", "req-2": "ses_A"}

    out_a = mod.render(records, smap, delta=True, max_chars=0)
    out_b = mod.render(records, smap, delta=True, max_chars=0)

    assert out_a == out_b


# ---------------------------------------------------------------------------
# Integration: load_records + dedup_turns + render (end-to-end no subprocess)
# ---------------------------------------------------------------------------

def test_end_to_end_prefill_decode_pair_renders_once(mod, tmp_path):
    """A real NDJSON file with a prefill+decode pair for one request_id renders
    as a single turn (auto dedup), and the prompt_text with embedded newlines
    appears correctly in the output."""
    f = tmp_path / "prompt-42.jsonl"
    prompt = "system: you are a coder\nuser: fix bug\n"
    records = [
        _make_record("req-XYZ", role="prefill", ts=10.0, prompt_text=prompt),
        _make_record("req-XYZ", role="decode", ts=10.1, prompt_text=prompt),
    ]
    _write_jsonl(f, records)

    loaded, n_bad = mod.load_records([f])
    turns = mod.dedup_turns(loaded, "auto")
    out = mod.render(turns, None, delta=False, max_chars=0)

    assert n_bad == 0
    # Only ONE turn header
    assert out.count("TURN 1") == 1
    assert "TURN 2" not in out
    # Newlines are real
    assert "\\n" not in out
    assert "system: you are a coder" in out
    assert "user: fix bug" in out


def test_end_to_end_bad_lines_skipped(mod, tmp_path):
    """Malformed lines are counted but don't crash; valid records proceed normally."""
    f = tmp_path / "prompt-bad.jsonl"
    f.write_text(
        "BAD JSON LINE\n"
        + json.dumps(_make_record("req-OK", role="prefill", prompt_text="content")) + "\n"
    )

    loaded, n_bad = mod.load_records([f])
    turns = mod.dedup_turns(loaded, "auto")
    out = mod.render(turns, None, delta=False, max_chars=0)

    assert n_bad == 1
    assert len(turns) == 1
    assert "content" in out
