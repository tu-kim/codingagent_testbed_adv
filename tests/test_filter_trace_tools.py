"""Tests for scripts/filter_trace_tools.py.

Conventions mirrored from tests/test_analyze_trace_parallelism.py (importlib
module loader via spec_from_file_location, scope="module" fixture, main(argv)
called directly since filter_trace_tools.main takes an argv param -- no
sys.argv patching needed). No network, no GPU: everything here is pure
JSON/regex/text processing over synthetic trace.jsonl fixtures.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "filter_trace_tools.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "filter_trace_tools", _SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["filter_trace_tools"] = module
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# 1. detect_nondeterminism
# ===========================================================================

# One crafted positive sample per ND_PATTERNS category, checked in isolation
# so it does NOT accidentally light up any of the *other* categories too
# (verified by hand against each regex before being written down here).
_ND_POSITIVE_SAMPLES: list[tuple[str, str]] = [
    ("datetime", "log at 2024-01-01T12:00:00 done"),
    ("duration", "operation took 2.1 seconds total"),
    ("hex_address", "object at 0x7f8a3c001230"),
    ("pid", "child pid=12345 exited"),
    ("uuid", "request id 123e4567-e89b-12d3-a456-426614174000 assigned"),
    ("tmp_path", "wrote scratch to /tmp/testbed-workspaces/foo.txt"),
    ("session_dir", "workspace at session-django__django-12345-a1b2c3d4 ready"),
    ("set_repr", "result = {'a', 'b', 'c'} computed"),
    ("random_seed_word", "using random ordering this run"),
]

_CLEAN_SAMPLE = "all tests passed"


@pytest.mark.parametrize("name,text", _ND_POSITIVE_SAMPLES)
def test_detect_nondeterminism_each_category_fires_on_positive_sample(
    mod, name, text
):
    flags = mod.detect_nondeterminism(text)
    assert name in flags, f"{name!r} pattern did not fire on {text!r}: got {flags!r}"


@pytest.mark.parametrize("name,_text", _ND_POSITIVE_SAMPLES)
def test_detect_nondeterminism_silent_on_clean_sample(mod, name, _text):
    # Every category must stay silent on plain prose with no ND signal.
    assert name not in mod.detect_nondeterminism(_CLEAN_SAMPLE)


def test_detect_nondeterminism_clean_sample_returns_empty_list(mod):
    assert mod.detect_nondeterminism(_CLEAN_SAMPLE) == []


def test_detect_nondeterminism_empty_string_returns_empty_list(mod):
    assert mod.detect_nondeterminism("") == []


def test_detect_nondeterminism_order_matches_nd_patterns_declaration_order(mod):
    # All 9 categories present, deliberately placed in the OPPOSITE order
    # from their ND_PATTERNS declaration -- the returned list must still
    # come back in declaration order (the impl iterates ND_PATTERNS, not
    # the text), so this is a real regression guard, not a tautology.
    text = (
        "using random seed here {'z'} session-abc-1234 wrote to "
        "/tmp/abcde and id 123e4567-e89b-12d3-a456-426614174000 "
        "pid=1 exited at 0x1a2b3c after it took 1.0 seconds "
        "starting 2024-01-01T00:00:00"
    )
    expected_order = [name for name, _pat in mod.ND_PATTERNS]
    assert mod.detect_nondeterminism(text) == expected_order


# ===========================================================================
# 2. iter_tool_parts robustness
# ===========================================================================

def _tool_part(tool: str, call_id: str = "c1", message_id: str | None = "m1",
               status: str = "completed", input_: Any = None,
               output: Any = None, metadata: dict | None = None) -> dict:
    return {
        "type": "tool",
        "tool": tool,
        "callID": call_id,
        "messageID": message_id,
        "state": {
            "status": status,
            "input": input_ if input_ is not None else {},
            "output": output if output is not None else "",
            "metadata": metadata if metadata is not None else {},
        },
    }


def test_iter_tool_parts_empty_messages_yields_nothing(mod):
    record = {"instance_id": "i1", "messages": []}
    assert list(mod.iter_tool_parts(record)) == []


def test_iter_tool_parts_missing_messages_key_yields_nothing(mod):
    record = {"instance_id": "i1"}
    assert list(mod.iter_tool_parts(record)) == []


def test_iter_tool_parts_message_without_parts_key_yields_nothing(mod):
    record = {"messages": [{"info": {"id": "m1"}}]}
    assert list(mod.iter_tool_parts(record)) == []


def test_iter_tool_parts_non_dict_message_skipped(mod):
    record = {"messages": ["not a dict", {"info": {"id": "m1"}, "parts": [
        _tool_part("bash"),
    ]}]}
    results = list(mod.iter_tool_parts(record))
    assert len(results) == 1
    assert results[0][1]["tool"] == "bash"


def test_iter_tool_parts_non_dict_part_skipped(mod):
    record = {"messages": [{"info": {"id": "m1"}, "parts": [
        "not a dict",
        _tool_part("bash"),
    ]}]}
    results = list(mod.iter_tool_parts(record))
    assert len(results) == 1
    assert results[0][1]["tool"] == "bash"


def test_iter_tool_parts_text_parts_skipped(mod):
    record = {"messages": [{"info": {"id": "m1"}, "parts": [
        {"type": "text", "text": "hello"},
        _tool_part("task"),
    ]}]}
    results = list(mod.iter_tool_parts(record))
    assert len(results) == 1
    assert results[0][1]["tool"] == "task"


def test_iter_tool_parts_yields_info_and_part_pairs(mod):
    record = {"messages": [{"info": {"id": "m1", "extra": "x"}, "parts": [
        _tool_part("bash"),
    ]}]}
    (info, part), = list(mod.iter_tool_parts(record))
    assert info == {"id": "m1", "extra": "x"}
    assert part["tool"] == "bash"


# ===========================================================================
# 3. extract_task_calls
# ===========================================================================

def test_extract_task_calls_picks_only_task_tool(mod):
    records = [{
        "instance_id": "i1", "session_id": "ses1",
        "messages": [{"info": {"id": "m1"}, "parts": [
            _tool_part("task", call_id="tcall", input_={"prompt": "p"},
                       output="done"),
            _tool_part("bash", call_id="bcall"),
        ]}],
    }]
    calls = mod.extract_task_calls(records)
    assert len(calls) == 1
    assert calls[0]["call_id"] == "tcall"


def test_extract_task_calls_field_mapping(mod):
    records = [{
        "instance_id": "django__django-1", "session_id": "ses_abc",
        "messages": [{"info": {"id": "info-id-1"}, "parts": [
            _tool_part("task", call_id="call1", message_id="msg1",
                       status="completed", input_={"prompt": "do X"},
                       output="task done"),
        ]}],
    }]
    calls = mod.extract_task_calls(records)
    assert calls == [{
        "instance_id": "django__django-1",
        "session_id": "ses_abc",
        "message_id": "msg1",
        "call_id": "call1",
        "status": "completed",
        "input": {"prompt": "do X"},
        "output": "task done",
    }]


def test_extract_task_calls_message_id_falls_back_to_info_id(mod):
    # part.get("messageID") is None (not present on the part at all) ->
    # must fall back to info.get("id") per the module docstring/spec.
    part = _tool_part("task", call_id="call1", message_id=None)
    del part["messageID"]
    records = [{
        "instance_id": "i1", "session_id": "s1",
        "messages": [{"info": {"id": "fallback-id"}, "parts": [part]}],
    }]
    calls = mod.extract_task_calls(records)
    assert calls[0]["message_id"] == "fallback-id"


def test_extract_task_calls_no_matches_returns_empty_list(mod):
    records = [{
        "instance_id": "i1", "session_id": "s1",
        "messages": [{"info": {"id": "m1"}, "parts": [_tool_part("bash")]}],
    }]
    assert mod.extract_task_calls(records) == []


# ===========================================================================
# 4. extract_bash_calls
# ===========================================================================

def test_extract_bash_calls_field_mapping(mod):
    records = [{
        "instance_id": "i1",
        "messages": [{"info": {"id": "m1"}, "parts": [
            _tool_part(
                "bash", call_id="bcall1", status="completed",
                input_={"command": "pytest -x", "description": "run tests"},
                output="3 passed in 0.42s",
                metadata={"exit": 0},
            ),
        ]}],
    }]
    calls = mod.extract_bash_calls(records)
    assert len(calls) == 1
    c = calls[0]
    assert c["instance_id"] == "i1"
    assert c["call_id"] == "bcall1"
    assert c["status"] == "completed"
    assert c["command"] == "pytest -x"
    assert c["description"] == "run tests"
    assert c["exit"] == 0
    assert c["output"] == "3 passed in 0.42s"
    assert "duration" in c["nd_flags"]


def test_extract_bash_calls_empty_output_yields_no_nd_flags(mod):
    records = [{
        "instance_id": "i1",
        "messages": [{"info": {"id": "m1"}, "parts": [
            _tool_part("bash", input_={"command": "echo hi"}, output=""),
        ]}],
    }]
    calls = mod.extract_bash_calls(records)
    assert calls[0]["nd_flags"] == []


def test_extract_bash_calls_only_bash_tool(mod):
    records = [{
        "instance_id": "i1",
        "messages": [{"info": {"id": "m1"}, "parts": [
            _tool_part("task", call_id="tcall"),
            _tool_part("bash", call_id="bcall"),
        ]}],
    }]
    calls = mod.extract_bash_calls(records)
    assert len(calls) == 1
    assert calls[0]["call_id"] == "bcall"


# ===========================================================================
# 5. _preview
# ===========================================================================

def test_preview_no_truncation_when_within_limit(mod):
    assert mod._preview("short text", 200) == "short text"


def test_preview_truncates_long_text_with_ellipsis(mod):
    text = "x" * 500
    preview = mod._preview(text, 50)
    assert len(preview) == 50
    assert preview.endswith("…")


def test_preview_escapes_newlines_regardless_of_length(mod):
    text = "line one\nline two"
    preview = mod._preview(text, 200)
    assert "\n" not in preview
    assert "\\n" in preview


# ===========================================================================
# 6. main() end-to-end
# ===========================================================================

def _task_part(**kw) -> dict:
    return _tool_part("task", **kw)


def _bash_part(**kw) -> dict:
    return _tool_part("bash", **kw)


def _write_trace(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _synthetic_records() -> list[dict]:
    rec1 = {
        "instance_id": "django__django-1",
        "session_id": "ses_abc",
        "messages": [{"info": {"id": "m1"}, "parts": [
            _task_part(call_id="call1", message_id="m1",
                       input_={"prompt": "do the subtask"},
                       output="subtask done"),
            _bash_part(call_id="call2",
                       input_={"command": "pytest", "description": "run tests"},
                       output="3 passed in 0.42s",
                       metadata={"exit": 0}),
            {"type": "text", "text": "narration, not a tool part"},
        ]}],
    }
    # error.stage upstream of `list` -> messages == [] contributes nothing.
    rec2 = {
        "instance_id": "django__django-2",
        "session_id": None,
        "messages": [],
    }
    return [rec1, rec2]


def test_main_default_writes_both_jsonl_files(mod, tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    _write_trace(run_dir / "trace.jsonl", _synthetic_records())

    rc = mod.main(["--run", str(run_dir)])

    assert rc == 0
    out_dir = run_dir / "tool_filter"
    assert (out_dir / "task_calls.jsonl").is_file()
    assert (out_dir / "bash_calls.jsonl").is_file()

    tasks = [json.loads(l) for l in
              (out_dir / "task_calls.jsonl").read_text().splitlines() if l.strip()]
    bashes = [json.loads(l) for l in
              (out_dir / "bash_calls.jsonl").read_text().splitlines() if l.strip()]
    assert len(tasks) == 1
    assert tasks[0]["instance_id"] == "django__django-1"
    assert tasks[0]["call_id"] == "call1"
    assert len(bashes) == 1
    assert bashes[0]["call_id"] == "call2"
    assert "duration" in bashes[0]["nd_flags"]


def test_main_only_task_writes_only_task_calls(mod, tmp_path):
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    _write_trace(run_dir / "trace.jsonl", _synthetic_records())

    rc = mod.main(["--run", str(run_dir), "--only", "task"])

    assert rc == 0
    out_dir = run_dir / "tool_filter"
    assert (out_dir / "task_calls.jsonl").is_file()
    assert not (out_dir / "bash_calls.jsonl").exists()


def test_main_only_bash_writes_only_bash_calls(mod, tmp_path):
    run_dir = tmp_path / "run3"
    run_dir.mkdir()
    _write_trace(run_dir / "trace.jsonl", _synthetic_records())

    rc = mod.main(["--run", str(run_dir), "--only", "bash"])

    assert rc == 0
    out_dir = run_dir / "tool_filter"
    assert not (out_dir / "task_calls.jsonl").exists()
    assert (out_dir / "bash_calls.jsonl").is_file()


def test_main_missing_trace_path_returns_exit_2(mod, tmp_path, capsys):
    run_dir = tmp_path / "does-not-exist"

    rc = mod.main(["--run", str(run_dir)])

    assert rc == 2
    captured = capsys.readouterr()
    assert "trace not found" in captured.err


def test_main_trace_overrides_run(mod, tmp_path):
    # --run points at a directory whose trace.jsonl must be IGNORED once
    # --trace is also given.
    run_dir = tmp_path / "run4"
    run_dir.mkdir()
    _write_trace(run_dir / "trace.jsonl", [{
        "instance_id": "should-be-ignored", "messages": [],
    }])

    explicit_trace = tmp_path / "elsewhere" / "custom_trace.jsonl"
    explicit_trace.parent.mkdir(parents=True)
    _write_trace(explicit_trace, [{
        "instance_id": "should-be-used",
        "messages": [{"info": {"id": "m1"}, "parts": [
            _task_part(call_id="c1"),
        ]}],
    }])

    rc = mod.main(["--run", str(run_dir), "--trace", str(explicit_trace)])

    assert rc == 0
    # Default --out is derived from the TRACE path's parent, not --run.
    out_dir = explicit_trace.parent / "tool_filter"
    tasks = [json.loads(l) for l in
              (out_dir / "task_calls.jsonl").read_text().splitlines() if l.strip()]
    assert len(tasks) == 1
    assert tasks[0]["instance_id"] == "should-be-used"
    assert not (run_dir / "tool_filter").exists()


def test_main_neither_run_nor_trace_exits_nonzero(mod):
    # argparse's ap.error() raises SystemExit(2) -- distinct code path from
    # the "file not found" `return 2`, but both signal "bad invocation".
    with pytest.raises(SystemExit):
        mod.main([])


def test_main_custom_out_dir(mod, tmp_path):
    run_dir = tmp_path / "run5"
    run_dir.mkdir()
    _write_trace(run_dir / "trace.jsonl", _synthetic_records())
    out_dir = tmp_path / "custom_out"

    rc = mod.main(["--run", str(run_dir), "--out", str(out_dir)])

    assert rc == 0
    assert (out_dir / "task_calls.jsonl").is_file()
    assert (out_dir / "bash_calls.jsonl").is_file()


# ---------------------------------------------------------------------------
# stdout content checks
# ---------------------------------------------------------------------------

def test_main_stdout_histogram_and_per_instance_counts(mod, tmp_path, capsys):
    run_dir = tmp_path / "run6"
    run_dir.mkdir()
    _write_trace(run_dir / "trace.jsonl", _synthetic_records())

    rc = mod.main(["--run", str(run_dir)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "task tool: 1 call(s) across 1 instance(s)" in out
    assert "django__django-1: 1 call(s)" in out
    assert "bash tool: 1 call(s)" in out
    assert "with nondeterminism flags" in out
    # histogram line for the "duration" category (bash output "3 passed
    # in 0.42s" triggers it -- see test_extract_bash_calls_field_mapping).
    import re
    assert re.search(r"duration\s+\d+ output\(s\)", out)


def test_main_stdout_preview_truncation_and_newline_escape(mod, tmp_path, capsys):
    run_dir = tmp_path / "run7"
    run_dir.mkdir()
    # Embeds a real newline EARLY (so it survives into the truncated
    # prefix) plus session_dir/tmp_path markers so the call carries
    # nd_flags and therefore actually appears in the "worst offenders"
    # stdout section (only flagged bash calls print there).
    long_output = "session-abcd\ntmp file at /tmp/scratchdir " + "z" * 300
    records = [{
        "instance_id": "i1",
        "messages": [{"info": {"id": "m1"}, "parts": [
            _bash_part(call_id="c1",
                       input_={"command": "produce_long_output"},
                       output=long_output, metadata={"exit": 0}),
        ]}],
    }]
    _write_trace(run_dir / "trace.jsonl", records)

    rc = mod.main(["--run", str(run_dir), "--preview-chars", "40"])
    assert rc == 0

    out = capsys.readouterr().out
    output_lines = [l for l in out.splitlines() if l.strip().startswith("output:")]
    assert len(output_lines) == 1, f"expected exactly one 'output:' line, got {output_lines!r}"
    line = output_lines[0]
    # Ellipsis-truncated since the source text is far longer than 40 chars.
    assert "…" in line
    # The embedded real newline must have been escaped to the literal
    # two-char sequence "\n" (backslash + n), never a raw line break.
    assert "\\n" in line
