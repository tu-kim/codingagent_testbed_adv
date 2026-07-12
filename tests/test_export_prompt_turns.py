"""Tests for scripts/export_prompt_turns.py.

Pure stdlib, no network, no GPU. Conventions mirror
test_format_prompt_dump.py (module-scope importlib fixture, tmp_path NDJSON
fixtures, determinism via repeated calls). The --tokenizer path is covered by
monkeypatching the module's _load_tokenizer indirection -- transformers is
never imported.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_prompt_turns.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("export_prompt_turns", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_prompt_turns"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# A realistic Qwen3-Coder engine prompt: system + user + assistant
# (think/text/tool_call) + tool_response user turn + generation tail.
QWEN_PROMPT = (
    "<|im_start|>system\n"
    "You are a coding agent.<|im_end|>\n"
    "<|im_start|>user\n"
    "Fix the bug in app.py.<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\nLook at the file first.\n</think>\n"
    "I'll inspect the file.\n"
    "<tool_call>\n<function=bash>\n<parameter=command>\ncat app.py\n"
    "</parameter>\n</function>\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n"
    "<tool_response>\nprint(1)\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _make_record(
    request_id: str = "req-001",
    role: str = "prefill",
    ts: float = 1_700_000_000.0,
    num_prompt_tokens: int = 100,
    prompt_text: str | None = QWEN_PROMPT,
    decode_error: str | None = None,
) -> dict:
    r: dict = {
        "ts": ts,
        "request_id": request_id,
        "role": role,
        "num_prompt_tokens": num_prompt_tokens,
    }
    if prompt_text is not None:
        r["prompt_text"] = prompt_text
    if decode_error is not None:
        r["decode_error"] = decode_error
    return r


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _export_one(mod, text: str, **kw):
    rec = _make_record(prompt_text=text)
    rec["_source"] = "prompt-1.jsonl"
    defaults = dict(template="qwen3", include_text=True, count_tokens=None,
                    session_id=None, request_index=1)
    defaults.update(kw)
    return mod.export_record(rec, **defaults)


# ---------------------------------------------------------------------------
# split_turns: ChatML framing
# ---------------------------------------------------------------------------

def test_turn_roles_and_generation_prompt(mod):
    out = _export_one(mod, QWEN_PROMPT)
    roles = [t["role"] for t in out["turns"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    gen = [t["generation_prompt"] for t in out["turns"]]
    assert gen == [False, False, False, False, True]
    assert out["turns"][-1]["segments"] == []
    assert out["parse_ok"] is True
    assert out["warnings"] == []


def test_turns_tile_prompt_text_exactly(mod):
    out = _export_one(mod, QWEN_PROMPT)
    turns = out["turns"]
    assert turns[0]["start"] == 0
    for prev, cur in zip(turns, turns[1:]):
        assert prev["end"] == cur["start"]
    assert turns[-1]["end"] == len(QWEN_PROMPT)
    assert out["num_chars"] == len(QWEN_PROMPT)
    for t in turns:
        assert t["num_chars"] == t["end"] - t["start"]


def test_segments_tile_content_and_match_offsets(mod):
    out = _export_one(mod, QWEN_PROMPT)
    for t in out["turns"]:
        segs = t["segments"]
        for prev, cur in zip(segs, segs[1:]):
            assert prev["end"] == cur["start"]
        for seg in segs:
            assert seg["text"] == QWEN_PROMPT[seg["start"]:seg["end"]]
            assert seg["num_chars"] == seg["end"] - seg["start"]
            assert len(seg["text"]) == seg["num_chars"]


def test_preamble_text_before_first_marker_is_kept(mod):
    out = _export_one(mod, "BOS-ISH" + QWEN_PROMPT)
    assert out["turns"][0]["role"] == "_preamble"
    assert out["turns"][0]["segments"][0]["text"] == "BOS-ISH"
    assert out["turns"][1]["role"] == "system"


def test_no_chatml_markers_falls_back_to_raw_turn_with_warning(mod):
    out = _export_one(mod, "just some plain text")
    assert [t["role"] for t in out["turns"]] == ["_raw"]
    assert out["warnings"] == ["no_chatml_markers"]
    assert out["parse_ok"] is True
    assert out["turns"][0]["segments"][0]["kind"] == "text"


def test_template_raw_skips_turn_split_without_warning(mod):
    out = _export_one(mod, QWEN_PROMPT, template="raw")
    assert [t["role"] for t in out["turns"]] == ["_raw"]
    assert out["warnings"] == []
    # segments are still scanned inside the single raw turn
    kinds = [s["kind"] for s in out["turns"][0]["segments"]]
    assert "think" in kinds and "tool_call" in kinds and "tool_response" in kinds


# ---------------------------------------------------------------------------
# scan_segments: kinds, names, edge cases
# ---------------------------------------------------------------------------

def test_assistant_turn_segment_kinds_and_tool_name(mod):
    out = _export_one(mod, QWEN_PROMPT)
    assistant = out["turns"][2]
    kinds = [s["kind"] for s in assistant["segments"]]
    assert kinds == ["think", "text", "tool_call"]
    think, text, call = assistant["segments"]
    assert think["text"].startswith("<think>") and think["text"].endswith("</think>")
    assert "I'll inspect the file." in text["text"]
    assert call["name"] == "bash"
    assert call["text"].startswith("<tool_call>")


def test_tool_response_turn_is_single_segment(mod):
    out = _export_one(mod, QWEN_PROMPT)
    tool_turn = out["turns"][3]
    assert [s["kind"] for s in tool_turn["segments"]] == ["tool_response"]
    assert "print(1)" in tool_turn["segments"][0]["text"]


def test_multiple_tool_responses_in_one_turn(mod):
    text = (
        "<|im_start|>user\n"
        "<tool_response>\na\n</tool_response>\n"
        "<tool_response>\nb\n</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    out = _export_one(mod, text)
    kinds = [s["kind"] for s in out["turns"][0]["segments"]]
    assert kinds == ["tool_response", "text", "tool_response"]


def test_tag_inside_open_segment_does_not_split(mod):
    """A literal tag INSIDE a tool_response body must not start a new
    segment (sequential earliest-open scan)."""
    text = (
        "<|im_start|>user\n"
        "<tool_response>\nsaw <think> and <tool_call> in output\n"
        "</tool_response><|im_end|>\n"
    )
    out = _export_one(mod, text)
    segs = out["turns"][0]["segments"]
    assert [s["kind"] for s in segs] == ["tool_response"]


def test_tool_call_name_qwen_json_style(mod):
    text = (
        "<|im_start|>assistant\n"
        '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>'
        "<|im_end|>\n"
    )
    out = _export_one(mod, text)
    seg = out["turns"][0]["segments"][0]
    assert seg["kind"] == "tool_call"
    assert seg["name"] == "get_weather"


def test_tool_call_minimax_m3_namespaced_tags_and_invoke_name(mod):
    body = (
        "]<]minimax[>[<tool_call>\n"
        ']<]minimax[>[<invoke name="run">\n'
        "]<]minimax[>[<param>ls]<]minimax[>[</param>\n"
        "]<]minimax[>[</invoke>\n"
        "]<]minimax[>[</tool_call>"
    )
    out = _export_one(mod, f"<|im_start|>assistant\n{body}<|im_end|>\n")
    segs = out["turns"][0]["segments"]
    assert [s["kind"] for s in segs] == ["tool_call"]
    assert segs[0]["name"] == "run"
    assert segs[0]["text"] == body


def test_tool_call_minimax_m2_tags(mod):
    # Real M2 output is XML invoke-style (dynamo/lib/parsers/src/tool_calling/
    # xml/parser.rs), not JSON.
    text = (
        "<|im_start|>assistant\n"
        '<minimax:tool_call>\n<invoke name="x">\n'
        '<parameter name="a">1</parameter>\n</invoke>\n'
        "</minimax:tool_call><|im_end|>\n"
    )
    out = _export_one(mod, text)
    seg = out["turns"][0]["segments"][0]
    assert seg["kind"] == "tool_call"
    assert seg["name"] == "x"


def test_unclosed_tag_swallows_rest_and_warns(mod):
    text = (
        "<|im_start|>assistant\n"
        "<tool_call>\n<function=bash>\nnever closed<|im_end|>\n"
    )
    out = _export_one(mod, text)
    segs = out["turns"][0]["segments"]
    assert [s["kind"] for s in segs] == ["tool_call"]
    assert out["warnings"] == ["unclosed:<tool_call>"]
    assert segs[0]["name"] == "bash"


def test_tool_call_without_recognizable_name_gets_none(mod):
    text = "<|im_start|>assistant\n<tool_call>\ngibberish\n</tool_call><|im_end|>\n"
    out = _export_one(mod, text)
    assert out["turns"][0]["segments"][0]["name"] is None


# ---------------------------------------------------------------------------
# export_record: text inclusion, missing text, tokenizer
# ---------------------------------------------------------------------------

def test_no_text_mode_omits_text_but_keeps_offsets_and_name(mod):
    out = _export_one(mod, QWEN_PROMPT, include_text=False)
    for t in out["turns"]:
        for seg in t["segments"]:
            assert "text" not in seg
            assert seg["num_chars"] == seg["end"] - seg["start"]
    call = out["turns"][2]["segments"][2]
    assert call["kind"] == "tool_call" and call["name"] == "bash"


def test_missing_prompt_text_yields_parse_not_ok(mod):
    rec = _make_record(prompt_text=None)
    out = mod.export_record(rec, template="qwen3", include_text=True,
                            count_tokens=None, session_id=None, request_index=1)
    assert out["parse_ok"] is False
    assert out["turns"] == []
    assert out["num_chars"] == 0
    assert out["num_prompt_tokens"] == 100  # dump metadata survives


def test_decode_error_is_surfaced_in_warnings(mod):
    rec = _make_record(prompt_text=None, decode_error="tokenizer exploded")
    out = mod.export_record(rec, template="qwen3", include_text=True,
                            count_tokens=None, session_id=None, request_index=1)
    assert out["parse_ok"] is False
    assert out["warnings"] == ["tokenizer exploded"]


def test_count_tokens_adds_token_fields(mod):
    fake = lambda s: len(s)  # noqa: E731 -- 1 token per char makes sums checkable
    out = _export_one(mod, QWEN_PROMPT, count_tokens=fake)
    assert out["num_tokens_text"] == len(QWEN_PROMPT)
    for t in out["turns"]:
        assert t["num_tokens"] == t["num_chars"]
        for seg in t["segments"]:
            assert seg["num_tokens"] == seg["num_chars"]


# ---------------------------------------------------------------------------
# load_records / _iter_files
# ---------------------------------------------------------------------------

def test_load_records_dir_glob_skips_blank_and_counts_malformed(mod, tmp_path: Path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "prompt-1.jsonl").write_text(
        json.dumps(_make_record(request_id="a")) + "\n\n{broken\n")
    _write_jsonl(d / "prompt-2.jsonl", [_make_record(request_id="b")])
    (d / "not-a-dump.txt").write_text("ignored: does not match prompt-*.jsonl")

    records, bad = mod.load_records([d])

    assert {r["request_id"] for r in records} == {"a", "b"}
    assert bad == 1


def test_load_records_source_is_filename_not_full_path(mod, tmp_path: Path):
    d = tmp_path / "prompts"
    d.mkdir()
    _write_jsonl(d / "prompt-42.jsonl", [_make_record()])
    records, _ = mod.load_records([d])
    assert records[0]["_source"] == "prompt-42.jsonl"


def test_load_records_explicit_file_and_missing_path_warns(mod, tmp_path: Path, capsys):
    f = tmp_path / "prompt-9.jsonl"
    _write_jsonl(f, [_make_record(request_id="x")])

    records, bad = mod.load_records([f, tmp_path / "does-not-exist"])

    assert [r["request_id"] for r in records] == ["x"]
    assert bad == 0
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# dedup_requests
# ---------------------------------------------------------------------------

def test_dedup_auto_prefers_prefill(mod):
    recs = [
        _make_record(request_id="r1", role="decode", ts=2.0),
        _make_record(request_id="r1", role="prefill", ts=1.0),
        _make_record(request_id="r2", role="decode", ts=3.0),
    ]
    out = mod.dedup_requests(recs, "auto")
    by_id = {r["request_id"]: r for r in out}
    assert len(out) == 2
    assert by_id["r1"]["role"] == "prefill"
    assert by_id["r2"]["role"] == "decode"  # decode-only request kept


def test_dedup_both_keeps_every_record(mod):
    recs = [
        _make_record(request_id="r1", role="prefill"),
        _make_record(request_id="r1", role="decode"),
    ]
    assert len(mod.dedup_requests(recs, "both")) == 2
    assert len(mod.dedup_requests(recs, "prefill")) == 1


def test_dedup_auto_keeps_records_without_request_id(mod):
    recs = [
        _make_record(request_id="r1"),
        {"ts": 1.0, "role": "prefill", "num_prompt_tokens": 5},  # no request_id
        {"ts": 2.0, "role": "decode", "num_prompt_tokens": 6},   # no request_id
    ]
    out = mod.dedup_requests(recs, "auto")
    assert len(out) == 3  # id-less records are never collapsed together


# ---------------------------------------------------------------------------
# export_records: ordering + session grouping
# ---------------------------------------------------------------------------

def test_export_records_sorted_by_ts(mod):
    recs = [
        _make_record(request_id="late", ts=200.0),
        _make_record(request_id="early", ts=100.0),
    ]
    out = mod.export_records(recs, None, template="qwen3",
                             include_text=False, count_tokens=None)
    assert [r["request_id"] for r in out] == ["early", "late"]
    assert all("session_id" not in r for r in out)


def test_export_records_session_map_groups_and_numbers(mod):
    recs = [
        _make_record(request_id="a1", ts=1.0),
        _make_record(request_id="b1", ts=2.0),
        _make_record(request_id="a2", ts=3.0),
    ]
    smap = {"a1": "ses_A", "a2": "ses_A", "b1": "ses_B"}
    out = mod.export_records(recs, smap, template="qwen3",
                             include_text=False, count_tokens=None)
    by_id = {r["request_id"]: r for r in out}
    assert by_id["a1"]["session_id"] == "ses_A"
    assert (by_id["a1"]["request_index"], by_id["a2"]["request_index"]) == (1, 2)
    assert by_id["b1"]["request_index"] == 1


def test_export_records_unmapped_request_lands_in_unknown_group(mod):
    recs = [
        _make_record(request_id="mapped", ts=1.0),
        _make_record(request_id="stray", ts=2.0),
    ]
    out = mod.export_records(recs, {"mapped": "ses_A"}, template="qwen3",
                             include_text=False, count_tokens=None)
    by_id = {r["request_id"]: r for r in out}
    assert by_id["mapped"]["session_id"] == "ses_A"
    assert by_id["stray"]["session_id"] == "unknown"
    assert by_id["stray"]["request_index"] == 1


def test_export_records_deterministic(mod):
    recs = [
        _make_record(request_id="r1", ts=1.0),
        _make_record(request_id="r2", ts=2.0),
    ]
    out1 = mod.export_records(recs, None, template="qwen3",
                              include_text=True, count_tokens=None)
    out2 = mod.export_records(recs, None, template="qwen3",
                              include_text=True, count_tokens=None)
    assert out1 == out2


# ---------------------------------------------------------------------------
# main(): end-to-end CLI over NDJSON fixtures
# ---------------------------------------------------------------------------

def test_main_writes_jsonl_out(mod, tmp_path: Path):
    dump_dir = tmp_path / "prompts"
    dump_dir.mkdir()
    _write_jsonl(dump_dir / "prompt-100.jsonl", [
        _make_record(request_id="r1", role="prefill", ts=1.0),
        _make_record(request_id="r1", role="decode", ts=1.1),
        _make_record(request_id="r2", role="prefill", ts=2.0),
    ])
    out_path = tmp_path / "turns.jsonl"

    rc = mod.main(["--prompts", str(dump_dir), "--out", str(out_path)])

    assert rc == 0
    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert [r["request_id"] for r in lines] == ["r1", "r2"]  # deduped, ts order
    assert all(r["role"] == "prefill" for r in lines)
    roles = [t["role"] for t in lines[0]["turns"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"]


def test_main_no_text_flag(mod, tmp_path: Path):
    dump_dir = tmp_path / "prompts"
    dump_dir.mkdir()
    _write_jsonl(dump_dir / "prompt-1.jsonl", [_make_record()])
    out_path = tmp_path / "compact.jsonl"

    rc = mod.main(["--prompts", str(dump_dir), "--no-text", "--out", str(out_path)])

    assert rc == 0
    rec = json.loads(out_path.read_text().splitlines()[0])
    for t in rec["turns"]:
        for seg in t["segments"]:
            assert "text" not in seg


def test_main_session_map(mod, tmp_path: Path):
    dump_dir = tmp_path / "prompts"
    dump_dir.mkdir()
    _write_jsonl(dump_dir / "prompt-1.jsonl", [
        _make_record(request_id="r1", ts=1.0),
        _make_record(request_id="r2", ts=2.0),
    ])
    smap_path = tmp_path / "smap.json"
    smap_path.write_text(json.dumps({"r1": "ses_X", "r2": "ses_X"}))
    out_path = tmp_path / "turns.jsonl"

    rc = mod.main(["--prompts", str(dump_dir), "--session-map", str(smap_path),
                   "--out", str(out_path)])

    assert rc == 0
    lines = [json.loads(l) for l in out_path.read_text().splitlines()]
    assert [(r["session_id"], r["request_index"]) for r in lines] == [
        ("ses_X", 1), ("ses_X", 2)]


def test_main_tokenizer_flag_uses_injected_loader(mod, monkeypatch, tmp_path: Path):
    monkeypatch.setattr(mod, "_load_tokenizer",
                        lambda spec: (lambda s: len(s.split())))
    dump_dir = tmp_path / "prompts"
    dump_dir.mkdir()
    _write_jsonl(dump_dir / "prompt-1.jsonl", [_make_record()])
    out_path = tmp_path / "sized.jsonl"

    rc = mod.main(["--prompts", str(dump_dir), "--tokenizer", "fake/tok",
                   "--out", str(out_path)])

    assert rc == 0
    rec = json.loads(out_path.read_text().splitlines()[0])
    assert rec["num_tokens_text"] == len(QWEN_PROMPT.split())
    assert all("num_tokens" in s for t in rec["turns"] for s in t["segments"])


def test_main_default_stdout_and_summary_tallies_segment_kinds(mod, tmp_path: Path, capsys):
    dump_dir = tmp_path / "prompts"
    dump_dir.mkdir()
    _write_jsonl(dump_dir / "prompt-1.jsonl", [_make_record()])

    rc = mod.main(["--prompts", str(dump_dir)])  # no --out -> stdout

    assert rc == 0
    captured = capsys.readouterr()
    recs = [json.loads(l) for l in captured.out.splitlines()]
    assert len(recs) == 1 and recs[0]["request_id"] == "req-001"
    # stderr summary carries the per-kind segment tally for QWEN_PROMPT:
    # 3 plain-text segments (system, user, assistant middle), 1 each of the rest.
    assert "exported 1 request(s)" in captured.err
    assert "text=3" in captured.err
    assert "think=1" in captured.err
    assert "tool_call=1" in captured.err
    assert "tool_response=1" in captured.err


def test_main_returns_1_when_no_records(mod, tmp_path: Path):
    empty = tmp_path / "prompts"
    empty.mkdir()
    assert mod.main(["--prompts", str(empty)]) == 1


def test_main_skips_malformed_lines(mod, tmp_path: Path, capsys):
    dump_dir = tmp_path / "prompts"
    dump_dir.mkdir()
    good = json.dumps(_make_record(request_id="ok"))
    (dump_dir / "prompt-1.jsonl").write_text(good + "\n{not json}\n")
    out_path = tmp_path / "turns.jsonl"

    rc = mod.main(["--prompts", str(dump_dir), "--out", str(out_path)])

    assert rc == 0
    lines = out_path.read_text().splitlines()
    assert len(lines) == 1
    assert "skipped 1 malformed" in capsys.readouterr().err
