"""Tests for scripts/analyze_prompt_source_share.py.

No network, no GPU. Loaded via importlib (script, not a package module),
matching this repo's convention for scripts/ tests (see
tests/test_e8_ttft_tpot.py, tests/test_e4_osl_correction.py).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "analyze_prompt_source_share.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pss():
    return _load_module("analyze_prompt_source_share", _SCRIPT_PATH)


# ---------------------------------------------------------------------------
# Event builders / file writer
# ---------------------------------------------------------------------------

def _turn_start(step, system_chars=0, messages=None, session_id=None):
    ev = {"ev": "turn.start", "step": step, "system": {"chars": system_chars}}
    if messages is not None:
        ev["messages"] = messages
    if session_id is not None:
        ev["sessionID"] = session_id
    return ev


def _llm_end(step, request_id="r1", tokens=None):
    ev = {"ev": "llm.end", "step": step, "request_id": request_id}
    if tokens is not None:
        ev["tokens"] = tokens
    return ev


def _tool_end(name, output_chars):
    return {"ev": "tool.end", "name": name, "output_chars": output_chars}


def _write_jsonl(path: Path, lines: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, str):
                f.write(line + "\n")
            else:
                f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# _messages_chars
# ---------------------------------------------------------------------------

class TestMessagesChars:
    def test_head_fidelity_sums_chars_field(self, pss):
        msgs = [
            {"role": "user", "parts": 1, "chars": 120, "head": "..."},
            {"role": "assistant", "parts": 2, "chars": 45, "head": "..."},
        ]
        assert pss._messages_chars(msgs) == 165

    def test_count_fidelity_reads_total_chars(self, pss):
        msgs = {"count": 7, "roles": {"user": 3, "assistant": 4},
                "total_chars": 999}
        assert pss._messages_chars(msgs) == 999

    def test_count_fidelity_missing_total_chars_defaults_zero(self, pss):
        msgs = {"count": 2, "roles": {}}
        assert pss._messages_chars(msgs) == 0

    def test_full_fidelity_falls_back_to_string_content_length(self, pss):
        msgs = [{"role": "user", "content": "hello world"}]
        assert pss._messages_chars(msgs) == len("hello world")

    def test_full_fidelity_falls_back_to_json_dump_for_non_string_content(
            self, pss):
        content_obj = {"type": "tool_result", "value": [1, 2, 3]}
        msgs = [{"role": "tool", "content": content_obj}]
        expected = len(json.dumps(content_obj, ensure_ascii=False))
        assert pss._messages_chars(msgs) == expected

    def test_full_fidelity_element_missing_content_key_dumps_null(self, pss):
        msgs = [{"role": "user"}]
        expected = len(json.dumps(None, ensure_ascii=False))
        assert pss._messages_chars(msgs) == expected

    def test_full_fidelity_element_not_a_dict_uses_element_itself(self, pss):
        msgs = ["raw string content"]
        assert pss._messages_chars(msgs) == len("raw string content")

    def test_mixed_head_and_full_style_elements_in_one_list(self, pss):
        # One element carries an exact `chars` count (head style); the
        # other has no `chars` key and falls back to content length.
        msgs = [
            {"role": "user", "chars": 10, "head": "..."},
            {"role": "assistant", "content": "abcde"},
        ]
        assert pss._messages_chars(msgs) == 10 + 5

    @pytest.mark.parametrize("value", [None, "just a string", 42, 3.14, True])
    def test_non_list_non_dict_input_returns_zero(self, pss, value):
        assert pss._messages_chars(value) == 0


# ---------------------------------------------------------------------------
# load_turns
# ---------------------------------------------------------------------------

class TestLoadTurns:
    def test_read_output_attributed_to_later_turn_not_current(
            self, pss, tmp_path):
        f = tmp_path / "sess.jsonl"
        _write_jsonl(f, [
            _turn_start(step=1, system_chars=5),
            _llm_end(step=1, tokens={"input": 10}),
            _turn_start(step=2, system_chars=5),
            _llm_end(step=2, tokens={"input": 10}),
            _tool_end("read", 80),  # finishes during step 2's tool phase
            _turn_start(step=3, system_chars=5, messages=[{"chars": 200}]),
        ])
        rows = {r["step"]: r for r in pss.load_turns(tmp_path, {"read"})}
        assert rows[1]["read_chars"] == 0
        assert rows[2]["read_chars"] == 0
        assert rows[3]["read_chars"] == 80

    def test_non_read_tool_end_is_ignored(self, pss, tmp_path):
        f = tmp_path / "sess.jsonl"
        _write_jsonl(f, [
            _tool_end("bash", 500),
            _turn_start(step=1, system_chars=5, messages=[{"chars": 200}]),
        ])
        rows = pss.load_turns(tmp_path, {"read"})
        assert rows[0]["read_chars"] == 0

    def test_custom_read_tools_set_is_honoured(self, pss, tmp_path):
        f = tmp_path / "sess.jsonl"
        _write_jsonl(f, [
            _tool_end("log", 40),
            _turn_start(step=1, system_chars=5, messages=[{"chars": 95}]),
        ])
        rows_custom = pss.load_turns(tmp_path, {"log"})
        assert rows_custom[0]["read_chars"] == 40
        # the default tool name "read" must NOT match under a custom set
        rows_default = pss.load_turns(tmp_path, {"read"})
        assert rows_default[0]["read_chars"] == 0

    def test_turn_without_llm_end_keeps_isl_tokens_blank(self, pss, tmp_path):
        f = tmp_path / "sess.jsonl"
        _write_jsonl(f, [_turn_start(step=1, system_chars=5)])
        rows = pss.load_turns(tmp_path, {"read"})
        assert len(rows) == 1
        assert rows[0]["isl_tokens"] == ""

    def test_isl_tokens_is_input_plus_cache_read(self, pss, tmp_path):
        f = tmp_path / "sess.jsonl"
        _write_jsonl(f, [
            _turn_start(step=1),
            _llm_end(step=1, tokens={"input": 40, "cache": {"read": 15}}),
        ])
        rows = pss.load_turns(tmp_path, {"read"})
        assert rows[0]["isl_tokens"] == 55

    def test_isl_tokens_defaults_cache_read_to_zero_when_absent(
            self, pss, tmp_path):
        f = tmp_path / "sess.jsonl"
        _write_jsonl(f, [
            _turn_start(step=1),
            _llm_end(step=1, tokens={"input": 40}),
        ])
        rows = pss.load_turns(tmp_path, {"read"})
        assert rows[0]["isl_tokens"] == 40

    def test_multiple_session_files_handled_independently(
            self, pss, tmp_path):
        d = tmp_path / "profiles"
        d.mkdir()
        _write_jsonl(d / "sess_a.jsonl", [
            _tool_end("read", 500),
            _turn_start(step=1, system_chars=5, messages=[{"chars": 95}]),
        ])
        _write_jsonl(d / "sess_b.jsonl", [
            _turn_start(step=1, system_chars=5, messages=[{"chars": 95}]),
        ])
        rows = {r["session_id"]: r for r in pss.load_turns(d, {"read"})}
        assert rows["sess_a"]["read_chars"] == 95  # clamped from 500
        assert rows["sess_b"]["read_chars"] == 0   # no leak from sess_a

    def test_malformed_json_and_unknown_event_are_skipped(
            self, pss, tmp_path):
        f = tmp_path / "sess.jsonl"
        _write_jsonl(f, [
            '{"ev": "turn.start", "step": 1, "system": {broken',
            {"ev": "some.unknown.event", "step": 99, "foo": "bar"},
            _turn_start(step=1, system_chars=5),
            _llm_end(step=1, request_id="rX", tokens={"input": 10}),
        ])
        rows = pss.load_turns(tmp_path, {"read"})
        assert len(rows) == 1
        assert rows[0]["request_id"] == "rX"
        assert rows[0]["isl_tokens"] == 10

    def test_read_chars_clamped_so_share_never_exceeds_total(
            self, pss, tmp_path):
        f = tmp_path / "sess.jsonl"
        _write_jsonl(f, [
            _tool_end("read", 1000),
            _turn_start(step=1, system_chars=10, messages=[{"chars": 5}]),
        ])
        rows = pss.load_turns(tmp_path, {"read"})
        row = rows[0]
        assert row["total_chars"] == 15
        # bound = max(total - system, 0) = max(15-10, 0) = 5
        assert row["read_chars"] == 5
        assert row["other_chars"] == 0
        assert row["read_share"] == round(5 / 15, 4)

    def test_zero_total_chars_leaves_share_and_estimate_fields_blank(
            self, pss, tmp_path):
        f = tmp_path / "sess.jsonl"
        _write_jsonl(f, [
            _turn_start(step=1, system_chars=0, messages=None),
            _llm_end(step=1, tokens={"input": 100}),
        ])
        rows = pss.load_turns(tmp_path, {"read"})
        row = rows[0]
        assert row["total_chars"] == 0
        assert row["system_share"] == ""
        assert row["read_share"] == ""
        assert row["est_system_tokens"] == ""
        assert row["est_read_tokens"] == ""
        # isl_tokens itself is computed independent of chars/total
        assert row["isl_tokens"] == 100

    def test_share_and_estimate_arithmetic(self, pss, tmp_path):
        f = tmp_path / "sess.jsonl"
        _write_jsonl(f, [
            _tool_end("read", 30),
            _turn_start(step=1, system_chars=20, messages=[{"chars": 50}]),
            _llm_end(step=1, tokens={"input": 180, "cache": {"read": 20}}),
        ])
        rows = pss.load_turns(tmp_path, {"read"})
        row = rows[0]
        # total = system(20) + messages(50) = 70; read(30) fits within the
        # bound (70-20=50) so it is NOT clamped.
        assert row["total_chars"] == 70
        assert row["read_chars"] == 30
        assert row["other_chars"] == 20  # 70 - 20 - 30
        assert row["system_share"] == round(20 / 70, 4)
        assert row["read_share"] == round(30 / 70, 4)
        assert row["isl_tokens"] == 200  # 180 + 20
        assert row["est_system_tokens"] == round(200 * 20 / 70)
        assert row["est_read_tokens"] == round(200 * 30 / 70)

    def test_session_id_falls_back_to_filename_stem(self, pss, tmp_path):
        f = tmp_path / "my-session-42.jsonl"
        _write_jsonl(f, [_turn_start(step=1, system_chars=5)])
        rows = pss.load_turns(tmp_path, {"read"})
        assert rows[0]["session_id"] == "my-session-42"


# ---------------------------------------------------------------------------
# print_sessions / print_summary
# ---------------------------------------------------------------------------

def _row(sid, step=1, system_chars=5, read_chars=0, other_chars=5,
         total_chars=10, isl_tokens=""):
    total = total_chars
    return {
        "session_id": sid, "step": step,
        "system_chars": system_chars, "read_chars": read_chars,
        "other_chars": other_chars, "total_chars": total,
        "system_share": round(system_chars / total, 4) if total else "",
        "read_share": round(read_chars / total, 4) if total else "",
        "isl_tokens": isl_tokens,
    }


class TestPrintSessions:
    def test_max_sessions_cutoff_prints_remaining_count(self, pss, capsys):
        rows = [_row(sid) for sid in ("s1", "s2", "s3")]
        pss.print_sessions(rows, limit=2)
        out = capsys.readouterr().out
        assert "session s1" in out
        assert "session s2" in out
        assert "session s3" not in out
        assert "... 1 more sessions (all of them are in the CSV)" in out

    def test_no_cutoff_message_when_within_limit(self, pss, capsys):
        rows = [_row(sid) for sid in ("s1", "s2")]
        pss.print_sessions(rows, limit=5)
        out = capsys.readouterr().out
        assert "session s1" in out
        assert "session s2" in out
        assert "more sessions" not in out


class TestPrintSummary:
    def test_no_usage_data_prints_shares_only_line(self, pss, capsys):
        rows = [
            {"total_chars": 100, "system_chars": 10, "read_chars": 5,
             "isl_tokens": ""},
            {"total_chars": 50, "system_chars": 5, "read_chars": 5,
             "isl_tokens": ""},
        ]
        pss.print_summary(rows)
        out = capsys.readouterr().out
        assert "turns: 2" in out
        assert ("no llm.end token usage in these profiles -- shares only, "
                "no token estimates") in out
        assert "estimated tokens over" not in out

    def test_with_usage_data_prints_estimated_tokens_line(self, pss, capsys):
        rows = [
            {"total_chars": 100, "system_chars": 10, "read_chars": 5,
             "isl_tokens": 200, "est_system_tokens": 20,
             "est_read_tokens": 10},
        ]
        pss.print_summary(rows)
        out = capsys.readouterr().out
        assert "estimated tokens over 1 turns with usage data" in out
        assert "no llm.end token usage" not in out

    def test_empty_rows_prints_zero_turns_and_returns(self, pss, capsys):
        pss.print_summary([])
        out = capsys.readouterr().out
        assert "turns: 0" in out


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_missing_profiles_path_returns_2(self, pss, tmp_path, capsys):
        rc = pss.main(["--profiles", str(tmp_path / "nope")])
        assert rc == 2
        err = capsys.readouterr().err
        assert "not found" in err

    def test_no_turn_start_events_returns_2(self, pss, tmp_path, capsys):
        d = tmp_path / "profiles"
        d.mkdir()
        _write_jsonl(d / "s1.jsonl", [_tool_end("read", 5)])
        rc = pss.main(["--profiles", str(d)])
        assert rc == 2
        err = capsys.readouterr().err
        assert "no turn.start events found" in err

    def test_writes_csv_with_exact_cols_header(self, pss, tmp_path):
        d = tmp_path / "profiles"
        d.mkdir()
        _write_jsonl(d / "s1.jsonl", [
            _turn_start(step=1, system_chars=10, messages=[{"chars": 5}]),
            _llm_end(step=1, request_id="r1", tokens={"input": 100}),
        ])
        out = tmp_path / "out.csv"
        rc = pss.main(["--profiles", str(d), "--out", str(out)])
        assert rc == 0
        with out.open() as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == pss.COLS
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["request_id"] == "r1"

    def test_out_parent_directory_created_if_absent(self, pss, tmp_path):
        d = tmp_path / "profiles"
        d.mkdir()
        _write_jsonl(d / "s1.jsonl", [
            _turn_start(step=1, system_chars=10, messages=[{"chars": 5}]),
        ])
        out = tmp_path / "nested" / "deep" / "out.csv"
        assert not out.parent.exists()
        rc = pss.main(["--profiles", str(d), "--out", str(out)])
        assert rc == 0
        assert out.exists()

    def test_returns_zero_without_out_flag(self, pss, tmp_path, capsys):
        d = tmp_path / "profiles"
        d.mkdir()
        _write_jsonl(d / "s1.jsonl", [
            _turn_start(step=1, system_chars=10, messages=[{"chars": 5}]),
        ])
        rc = pss.main(["--profiles", str(d)])
        assert rc == 0
        out = capsys.readouterr().out
        assert "wrote" not in out
