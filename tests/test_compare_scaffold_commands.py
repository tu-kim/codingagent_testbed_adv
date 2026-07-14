"""Tests for scripts/compare_scaffold_commands.py.

Conventions mirrored from tests/test_filter_trace_tools.py (importlib module
loader via spec_from_file_location, scope="module" fixture, main(argv) called
directly since main takes an argv param -- no sys.argv patching needed) and
tests/test_sweagent_apps.py (real trajectory JSON fixtures under tmp_path, no
mocking of the JSON/filesystem layer). No network, no GPU: everything here is
pure JSON/text processing over synthetic trace.jsonl + .traj fixtures.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "compare_scaffold_commands.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "compare_scaffold_commands", _SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_scaffold_commands"] = module
    spec.loader.exec_module(module)
    return module


# ===========================================================================
# shared fixture helpers
# ===========================================================================

def _tool_part(tool: str, input_: dict | None = None) -> dict:
    return {"type": "tool", "tool": tool, "state": {"input": input_ or {}}}


def _write_trace(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _write_traj(path: Path, steps: list[dict]) -> None:
    path.write_text(json.dumps({"trajectory": steps}))


def _sa_record(instance_id: str, traj_dir: Path, **extra: Any) -> dict:
    rec = {"instance_id": instance_id, "traj_dir": str(traj_dir)}
    rec.update(extra)
    return rec


def _build_oc_run(tmp_path: Path) -> Path:
    """3 opencode-run instances: apps-00001/00002 also appear in the
    sweagent run built by _build_sa_run (matched); apps-00099 is
    opencode-only."""
    run_dir = tmp_path / "oc_run"
    run_dir.mkdir()
    records = [
        {
            "instance_id": "apps-00001", "rtt_s": 12.5, "success": True,
            "messages": [{"parts": [
                _tool_part("read", {"filePath": "/ws/PROBLEM.md"}),
                _tool_part("bash", {"command": "python solution.py"}),
                _tool_part("edit", {"filePath": "/ws/solution.py"}),
            ]}],
        },
        {
            "instance_id": "apps-00002", "rtt_s": 8.0, "success": True,
            "messages": [{"parts": [_tool_part("bash", {"command": "ls"})]}],
        },
        {
            "instance_id": "apps-00099", "rtt_s": 3.0, "success": True,
            "messages": [{"parts": [_tool_part("bash", {"command": "echo hi"})]}],
        },
    ]
    _write_trace(run_dir / "trace.jsonl", records)
    return run_dir


def _build_sa_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "sa_run"
    trajs = run_dir / "trajs"
    trajs.mkdir(parents=True)

    d1 = trajs / "apps-00001"
    d1.mkdir()
    _write_traj(d1 / "apps-00001.traj", [
        {"action": "cat PROBLEM.md"},
        {"action": "python solution.py"},
    ])

    d2 = trajs / "apps-00002"
    d2.mkdir()
    _write_traj(d2 / "apps-00002.traj", [{"action": "submit"}])

    records = [
        _sa_record("apps-00001", d1, rtt_s=40.0, success=True),
        _sa_record("apps-00002", d2, rtt_s=30.0, success=True),
    ]
    _write_trace(run_dir / "trace.jsonl", records)
    return run_dir


# ===========================================================================
# 1. load_trace
# ===========================================================================

def test_load_trace_missing_file_returns_empty_and_warns(mod, tmp_path, capsys):
    run_dir = tmp_path / "nope"
    run_dir.mkdir()

    out = mod.load_trace(run_dir)

    assert out == []
    assert "trace not found" in capsys.readouterr().err


def test_load_trace_parses_valid_lines(mod, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_trace(run_dir / "trace.jsonl", [
        {"instance_id": "a"}, {"instance_id": "b"},
    ])

    out = mod.load_trace(run_dir)

    assert [r["instance_id"] for r in out] == ["a", "b"]


def test_load_trace_skips_blank_lines(mod, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "trace.jsonl").write_text(
        json.dumps({"instance_id": "a"}) + "\n\n   \n"
        + json.dumps({"instance_id": "b"}) + "\n"
    )

    out = mod.load_trace(run_dir)

    assert len(out) == 2


def test_load_trace_skips_malformed_line_and_warns_but_keeps_rest(
    mod, tmp_path, capsys,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "trace.jsonl").write_text(
        json.dumps({"instance_id": "a"}) + "\n"
        + "{not valid json\n"
        + json.dumps({"instance_id": "b"}) + "\n"
    )

    out = mod.load_trace(run_dir)

    assert [r["instance_id"] for r in out] == ["a", "b"]
    err = capsys.readouterr().err
    assert "malformed" in err
    assert "line 2" in err


# ===========================================================================
# 2. _oc_summary
# ===========================================================================

_OC_SUMMARY_CASES: list[tuple[str, dict, str]] = [
    ("bash", {"command": "pytest -x"}, "pytest -x"),
    ("read", {"filePath": "/a/b.py"}, "/a/b.py"),
    ("read", {"path": "/a/b.py"}, "/a/b.py"),  # fallback when no filePath
    ("write", {"filePath": "/x.py"}, "/x.py"),
    ("edit", {"filePath": "/y.py"}, "/y.py"),
    ("edit", {"path": "/y.py"}, "/y.py"),  # fallback when no filePath
    ("task", {"description": "do X", "prompt": "ignored"}, "do X"),
    ("task", {"prompt": "do the subtask"}, "do the subtask"),  # fallback
    ("glob", {"pattern": "*.py"}, "*.py"),  # no path -> no "(in ...)" suffix
    ("grep", {"pattern": "foo", "path": "/src"}, "foo  (in /src)"),
    ("list", {"path": "/dir"}, "/dir"),
    ("webfetch", {"url": "http://x"}, "http://x"),
]


@pytest.mark.parametrize("tool,inp,expected", _OC_SUMMARY_CASES)
def test_oc_summary_per_tool_mapping(mod, tool, inp, expected):
    assert mod._oc_summary(tool, inp) == expected


def test_oc_summary_unknown_tool_returns_compact_json(mod):
    result = mod._oc_summary("mystery", {"a": 1, "b": "two"})
    assert json.loads(result) == {"a": 1, "b": "two"}


def test_oc_summary_none_input_treated_as_empty(mod):
    assert mod._oc_summary("bash", None) == ""


# ===========================================================================
# 3. extract_opencode
# ===========================================================================

def test_extract_opencode_builds_seq_tool_command(mod):
    records = [{"instance_id": "i1", "messages": [{"parts": [
        _tool_part("bash", {"command": "ls"}),
        _tool_part("read", {"filePath": "/a.py"}),
    ]}]}]

    per = mod.extract_opencode(records, bash_only=False)

    assert [c["tool"] for c in per["i1"]] == ["bash", "read"]
    assert [c["seq"] for c in per["i1"]] == [1, 2]
    assert per["i1"][0]["command"] == "ls"
    assert per["i1"][1]["command"] == "/a.py"


def test_extract_opencode_seq_increments_across_messages(mod):
    # seq is 1-indexed PER INSTANCE and increments across ALL messages of
    # that record, not reset per message.
    records = [{"instance_id": "i1", "messages": [
        {"parts": [_tool_part("bash", {"command": "one"})]},
        {"parts": [_tool_part("bash", {"command": "two"})]},
    ]}]

    per = mod.extract_opencode(records, bash_only=False)

    assert [c["seq"] for c in per["i1"]] == [1, 2]
    assert [c["command"] for c in per["i1"]] == ["one", "two"]


def test_extract_opencode_bash_only_filters_non_bash_tools(mod):
    records = [{"instance_id": "i1", "messages": [{"parts": [
        _tool_part("bash", {"command": "ls"}),
        _tool_part("read", {"filePath": "/a.py"}),
        _tool_part("edit", {"filePath": "/b.py"}),
    ]}]}]

    per = mod.extract_opencode(records, bash_only=True)

    assert [c["tool"] for c in per["i1"]] == ["bash"]
    # seq re-numbers over the surviving (filtered) calls only.
    assert per["i1"][0]["seq"] == 1


def test_extract_opencode_missing_messages_key_contributes_nothing(mod):
    # Upstream-error TaskRecord (e.g. error.stage before `list`) has no
    # "messages" key at all.
    records = [{"instance_id": "i1"}]

    per = mod.extract_opencode(records, bash_only=False)

    assert per == {}


def test_extract_opencode_empty_messages_list_contributes_nothing(mod):
    records = [{"instance_id": "i1", "messages": []}]

    per = mod.extract_opencode(records, bash_only=False)

    assert "i1" not in per


def test_extract_opencode_text_parts_are_skipped(mod):
    records = [{"instance_id": "i1", "messages": [{"parts": [
        {"type": "text", "text": "narration, not a tool call"},
        _tool_part("bash", {"command": "ls"}),
    ]}]}]

    per = mod.extract_opencode(records, bash_only=False)

    assert len(per["i1"]) == 1
    assert per["i1"][0]["tool"] == "bash"


# ===========================================================================
# 4. extract_sweagent
# ===========================================================================

def test_extract_sweagent_reads_exact_traj_file(mod, tmp_path):
    traj_dir = tmp_path / "trajs"
    traj_dir.mkdir()
    _write_traj(traj_dir / "i1.traj", [{"action": "python solution.py"}])
    records = [_sa_record("i1", traj_dir)]

    per = mod.extract_sweagent(records, bash_only=False)

    assert per["i1"][0]["seq"] == 1
    assert per["i1"][0]["tool"] == "python"
    assert per["i1"][0]["command"] == "python solution.py"


def test_extract_sweagent_glob_fallback_when_filename_differs(mod, tmp_path):
    # The .traj filename does NOT match the instance_id -- extract_sweagent
    # must fall back to the first *.traj glob hit in that dir.
    traj_dir = tmp_path / "trajs" / "i2"
    traj_dir.mkdir(parents=True)
    _write_traj(traj_dir / "some_other_rollout_name.traj", [{"action": "cat file.py"}])
    records = [_sa_record("i2", traj_dir)]

    per = mod.extract_sweagent(records, bash_only=False)

    assert per["i2"][0]["tool"] == "cat"
    assert per["i2"][0]["command"] == "cat file.py"


def test_extract_sweagent_missing_traj_dir_key_contributes_nothing(mod):
    records = [{"instance_id": "i1"}]  # no traj_dir field at all

    per = mod.extract_sweagent(records, bash_only=False)

    assert per == {}


def test_extract_sweagent_no_traj_file_warns_and_contributes_nothing(
    mod, tmp_path, capsys,
):
    traj_dir = tmp_path / "empty"
    traj_dir.mkdir()
    records = [_sa_record("i1", traj_dir)]

    per = mod.extract_sweagent(records, bash_only=False)

    assert per == {}
    assert "no .traj for i1" in capsys.readouterr().err


def test_extract_sweagent_malformed_json_warns_and_contributes_nothing(
    mod, tmp_path, capsys,
):
    traj_dir = tmp_path / "trajs"
    traj_dir.mkdir()
    (traj_dir / "i1.traj").write_text("{not valid json")
    records = [_sa_record("i1", traj_dir)]

    per = mod.extract_sweagent(records, bash_only=False)

    assert per == {}
    assert "not JSON" in capsys.readouterr().err


def test_extract_sweagent_bad_steps_are_skipped_without_consuming_seq(mod, tmp_path):
    # Both a non-dict step and a dict step with no recognized action key
    # must be silently skipped -- and neither should consume a seq number
    # (the surviving step must still be seq=1, not seq=3).
    traj_dir = tmp_path / "trajs"
    traj_dir.mkdir()
    _write_traj(traj_dir / "i1.traj", [
        "not a dict", {"no_action_here": True}, {"action": "ls"},
    ])
    records = [_sa_record("i1", traj_dir)]

    per = mod.extract_sweagent(records, bash_only=False)

    assert len(per["i1"]) == 1
    assert per["i1"][0]["seq"] == 1
    assert per["i1"][0]["command"] == "ls"


def test_extract_sweagent_action_key_priority_prefers_action(mod, tmp_path):
    # _ACTION_KEYS = ("action", "command", "thought_action") -- "action"
    # must win when multiple candidate keys are present.
    traj_dir = tmp_path / "trajs"
    traj_dir.mkdir()
    _write_traj(traj_dir / "i1.traj", [
        {"action": "first", "command": "second", "thought_action": "third"},
    ])
    records = [_sa_record("i1", traj_dir)]

    per = mod.extract_sweagent(records, bash_only=False)

    assert per["i1"][0]["command"] == "first"


def test_extract_sweagent_bash_only_is_a_noop(mod, tmp_path):
    # Every SWE-agent action is itself a shell action, so --bash-only is
    # documented as a no-op filter here (kept for symmetry with opencode).
    traj_dir = tmp_path / "trajs"
    traj_dir.mkdir()
    _write_traj(traj_dir / "i1.traj", [
        {"action": "python x.py"}, {"action": "submit"},
    ])
    records = [_sa_record("i1", traj_dir)]

    per_all = mod.extract_sweagent(records, bash_only=False)
    per_bash = mod.extract_sweagent(records, bash_only=True)

    assert per_all == per_bash
    assert len(per_all["i1"]) == 2


# ===========================================================================
# 5. _traj_path
# ===========================================================================

def test_traj_path_prefers_exact_match_over_glob(mod, tmp_path):
    traj_dir = tmp_path / "trajs"
    traj_dir.mkdir()
    exact = traj_dir / "i1.traj"
    exact.write_text("{}")
    (traj_dir / "decoy.traj").write_text("{}")  # would also glob-match

    assert mod._traj_path(traj_dir, "i1") == exact


def test_traj_path_glob_fallback_returns_sorted_first(mod, tmp_path):
    traj_dir = tmp_path / "trajs"
    traj_dir.mkdir()
    (traj_dir / "b_rollout.traj").write_text("{}")
    (traj_dir / "a_rollout.traj").write_text("{}")

    found = mod._traj_path(traj_dir, "i1")

    assert found is not None
    assert found.name == "a_rollout.traj"


def test_traj_path_none_when_dir_missing_or_empty(mod, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()

    assert mod._traj_path(tmp_path / "does-not-exist", "i1") is None
    assert mod._traj_path(empty, "i1") is None


# ===========================================================================
# 6. _action_of
# ===========================================================================

def test_action_of_returns_action_key_unstripped(mod):
    # Returns the raw matched string as-is (not .strip()'d).
    assert mod._action_of({"action": "  ls  "}) == "  ls  "


def test_action_of_falls_back_to_command_when_action_blank(mod):
    assert mod._action_of({"action": "   ", "command": "ls -la"}) == "ls -la"


def test_action_of_skips_non_string_values(mod):
    assert mod._action_of({"action": 123, "command": "ls"}) == "ls"


def test_action_of_returns_none_when_nothing_matches(mod):
    assert mod._action_of({}) is None
    assert mod._action_of({"unrelated_key": "value"}) is None


# ===========================================================================
# 7. _cmd_type
# ===========================================================================

def test_cmd_type_first_whitespace_token(mod):
    assert mod._cmd_type("python solution.py") == "python"


def test_cmd_type_skips_leading_blank_lines(mod):
    assert mod._cmd_type("\n\n  \ncat file.py") == "cat"


def test_cmd_type_multiline_uses_first_nonblank_line(mod):
    assert mod._cmd_type("submit\nextra stuff on the next line") == "submit"


@pytest.mark.parametrize("text", ["", "   \n   ", "\n\n"])
def test_cmd_type_empty_or_whitespace_only_returns_placeholder(mod, text):
    assert mod._cmd_type(text) == "?"


# ===========================================================================
# 8. _median
# ===========================================================================

def test_median_empty_list_is_nan(mod):
    assert math.isnan(mod._median([]))


def test_median_odd_length_returns_middle_value(mod):
    assert mod._median([3.0, 1.0, 2.0]) == 2.0


def test_median_even_length_averages_middle_two(mod):
    assert mod._median([1.0, 2.0, 3.0, 4.0]) == 2.5


# ===========================================================================
# 9. _fmt_rtt
# ===========================================================================

@pytest.mark.parametrize("value,expected", [
    (12.345, "12.3s"),
    (5, "5.0s"),
    (None, "-"),
    ("n/a", "-"),
])
def test_fmt_rtt_formats_numeric_and_dashes_non_numeric(mod, value, expected):
    assert mod._fmt_rtt(value) == expected


# ===========================================================================
# 10. _preview
# ===========================================================================

def test_preview_no_truncation_when_within_limit(mod):
    assert mod._preview("short text", 200) == "short text"


def test_preview_truncates_long_text_with_ellipsis(mod):
    text = "x" * 500
    preview = mod._preview(text, 50)

    assert len(preview) == 50
    assert preview.endswith("…")


def test_preview_escapes_newlines_regardless_of_length(mod):
    preview = mod._preview("line one\nline two", 200)

    assert "\n" not in preview
    assert "\\n" in preview


# ===========================================================================
# 11. main() end-to-end
# ===========================================================================

def test_main_writes_both_csvs_with_expected_header_and_rows(mod, tmp_path):
    oc_run = _build_oc_run(tmp_path)
    sa_run = _build_sa_run(tmp_path)
    out_dir = tmp_path / "out"

    rc = mod.main([
        "--opencode-run", str(oc_run), "--sweagent-run", str(sa_run),
        "--out", str(out_dir),
    ])

    assert rc == 0
    cmds_csv = out_dir / "scaffold_commands.csv"
    summ_csv = out_dir / "scaffold_summary.csv"
    assert cmds_csv.is_file()
    assert summ_csv.is_file()

    with cmds_csv.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    assert header == ["scaffold", "instance_id", "seq", "tool", "command"]

    # The matched instance (apps-00001) must appear under BOTH scaffolds.
    scaffolds_for_i1 = {r[0] for r in rows if r[1] == "apps-00001"}
    assert scaffolds_for_i1 == {"opencode", "sweagent"}
    # Opencode-only instance appears solely under "opencode".
    scaffolds_for_i99 = {r[0] for r in rows if r[1] == "apps-00099"}
    assert scaffolds_for_i99 == {"opencode"}

    with summ_csv.open(newline="") as f:
        summ_rows = list(csv.DictReader(f))
    assert {"scaffold", "instance_id", "n_commands", "rtt_s", "success"} <= set(
        summ_rows[0].keys()
    )
    oc_summ = next(
        r for r in summ_rows
        if r["scaffold"] == "opencode" and r["instance_id"] == "apps-00001"
    )
    assert oc_summ["n_commands"] == "3"
    sa_summ = next(
        r for r in summ_rows
        if r["scaffold"] == "sweagent" and r["instance_id"] == "apps-00001"
    )
    assert sa_summ["n_commands"] == "2"


def test_main_bash_only_reduces_opencode_to_shell_commands(mod, tmp_path, capsys):
    oc_run = _build_oc_run(tmp_path)
    sa_run = _build_sa_run(tmp_path)
    out_dir = tmp_path / "out"

    rc = mod.main([
        "--opencode-run", str(oc_run), "--sweagent-run", str(sa_run),
        "--out", str(out_dir), "--bash-only",
    ])

    assert rc == 0
    with (out_dir / "scaffold_commands.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    oc_rows_i1 = [
        r for r in rows if r["scaffold"] == "opencode" and r["instance_id"] == "apps-00001"
    ]
    # read+bash+edit -> only the bash call survives --bash-only.
    assert len(oc_rows_i1) == 1
    assert oc_rows_i1[0]["tool"] == "bash"
    assert "shell commands only" in capsys.readouterr().out


def test_main_instance_filter_restricts_side_by_side_output(mod, tmp_path, capsys):
    oc_run = _build_oc_run(tmp_path)
    sa_run = _build_sa_run(tmp_path)

    rc = mod.main([
        "--opencode-run", str(oc_run), "--sweagent-run", str(sa_run),
        "--instance", "apps-00002",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "=== apps-00002 ===" in out
    assert "=== apps-00001 ===" not in out


def test_main_instance_filter_not_in_both_prints_note(mod, tmp_path, capsys):
    # apps-00099 only exists in the opencode run.
    oc_run = _build_oc_run(tmp_path)
    sa_run = _build_sa_run(tmp_path)

    rc = mod.main([
        "--opencode-run", str(oc_run), "--sweagent-run", str(sa_run),
        "--instance", "apps-00099",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "not in both traces" in out
    assert "oc=True" in out
    assert "sa=False" in out
    # Per-instance section still prints even though it's a one-sided match.
    assert "=== apps-00099 ===" in out


def test_main_both_traces_empty_returns_exit_2(mod, tmp_path, capsys):
    empty_oc = tmp_path / "empty_oc"
    empty_oc.mkdir()
    empty_sa = tmp_path / "empty_sa"
    empty_sa.mkdir()

    rc = mod.main(["--opencode-run", str(empty_oc), "--sweagent-run", str(empty_sa)])

    assert rc == 2
    assert "both traces empty/missing" in capsys.readouterr().err


def test_main_one_side_empty_still_returns_0(mod, tmp_path):
    oc_run = _build_oc_run(tmp_path)
    empty_sa = tmp_path / "empty_sa"
    empty_sa.mkdir()

    rc = mod.main(["--opencode-run", str(oc_run), "--sweagent-run", str(empty_sa)])

    assert rc == 0


def test_main_missing_messages_record_contributes_nothing(mod, tmp_path, capsys):
    # An opencode TaskRecord with no "messages" key at all (upstream error,
    # e.g. error.stage="session") must not appear anywhere in the report.
    oc_run = tmp_path / "oc_run"
    oc_run.mkdir()
    _write_trace(oc_run / "trace.jsonl", [
        {"instance_id": "apps-err", "success": False},
    ])
    sa_run = _build_sa_run(tmp_path)

    rc = mod.main(["--opencode-run", str(oc_run), "--sweagent-run", str(sa_run)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "apps-err" not in out


def test_main_sweagent_traj_glob_fallback_integration(mod, tmp_path):
    oc_run = _build_oc_run(tmp_path)

    sa_run = tmp_path / "sa_run2"
    trajs = sa_run / "trajs"
    d1 = trajs / "apps-00001"
    d1.mkdir(parents=True)
    _write_traj(d1 / "rollout_from_a_different_name.traj", [
        {"action": "python solution.py"},
    ])
    _write_trace(sa_run / "trace.jsonl", [
        _sa_record("apps-00001", d1, rtt_s=1.0, success=True),
    ])
    out_dir = tmp_path / "out"

    rc = mod.main([
        "--opencode-run", str(oc_run), "--sweagent-run", str(sa_run),
        "--out", str(out_dir),
    ])

    assert rc == 0
    with (out_dir / "scaffold_commands.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    sa_rows = [r for r in rows if r["scaffold"] == "sweagent"]
    assert len(sa_rows) == 1
    assert sa_rows[0]["command"] == "python solution.py"


def test_main_malformed_trace_line_skipped_but_rest_parsed(mod, tmp_path, capsys):
    oc_run = tmp_path / "oc_run"
    oc_run.mkdir()
    rec1 = {
        "instance_id": "apps-00001", "rtt_s": 1.0, "success": True,
        "messages": [{"parts": [_tool_part("bash", {"command": "ls"})]}],
    }
    rec2 = {
        "instance_id": "apps-00002", "rtt_s": 1.0, "success": True,
        "messages": [{"parts": [_tool_part("bash", {"command": "pwd"})]}],
    }
    (oc_run / "trace.jsonl").write_text(
        json.dumps(rec1) + "\n{not valid json\n" + json.dumps(rec2) + "\n"
    )
    sa_run = _build_sa_run(tmp_path)

    rc = mod.main(["--opencode-run", str(oc_run), "--sweagent-run", str(sa_run)])

    assert rc == 0
    captured = capsys.readouterr()
    assert "malformed" in captured.err
    # rec2 (after the malformed line) must still have been parsed and
    # counted among the matched instances.
    assert "matched instances" in captured.out


def test_main_max_instances_prints_truncation_note(mod, tmp_path, capsys):
    oc_run = tmp_path / "oc_run"
    oc_run.mkdir()
    sa_run = tmp_path / "sa_run"
    trajs = sa_run / "trajs"
    trajs.mkdir(parents=True)

    oc_records = []
    sa_records = []
    for i in range(3):
        iid = f"apps-0000{i}"
        oc_records.append({
            "instance_id": iid, "rtt_s": 1.0, "success": True,
            "messages": [{"parts": [_tool_part("bash", {"command": "ls"})]}],
        })
        d = trajs / iid
        d.mkdir()
        _write_traj(d / f"{iid}.traj", [{"action": "ls"}])
        sa_records.append(_sa_record(iid, d, rtt_s=1.0, success=True))
    _write_trace(oc_run / "trace.jsonl", oc_records)
    _write_trace(sa_run / "trace.jsonl", sa_records)

    rc = mod.main([
        "--opencode-run", str(oc_run), "--sweagent-run", str(sa_run),
        "--max-instances", "1",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "2 more matched instances" in out


def test_main_preview_chars_truncates_long_command_in_stdout(mod, tmp_path, capsys):
    oc_run = tmp_path / "oc_run"
    oc_run.mkdir()
    long_cmd = "x" * 300
    _write_trace(oc_run / "trace.jsonl", [{
        "instance_id": "apps-00001", "rtt_s": 1.0, "success": True,
        "messages": [{"parts": [_tool_part("bash", {"command": long_cmd})]}],
    }])
    sa_run = _build_sa_run(tmp_path)

    rc = mod.main([
        "--opencode-run", str(oc_run), "--sweagent-run", str(sa_run),
        "--instance", "apps-00001", "--preview-chars", "20",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "…" in out
    assert long_cmd not in out  # unbroken 300-char run must not appear


def test_main_without_out_flag_writes_no_csvs(mod, tmp_path, capsys):
    oc_run = _build_oc_run(tmp_path)
    sa_run = _build_sa_run(tmp_path)

    rc = mod.main(["--opencode-run", str(oc_run), "--sweagent-run", str(sa_run)])

    assert rc == 0
    assert not (tmp_path / "scaffold_commands.csv").exists()
    assert not (tmp_path / "scaffold_summary.csv").exists()
    assert "wrote " not in capsys.readouterr().out


def test_main_aggregate_table_histograms_and_only_in_sections_printed(
    mod, tmp_path, capsys,
):
    oc_run = _build_oc_run(tmp_path)
    sa_run = _build_sa_run(tmp_path)

    rc = mod.main(["--opencode-run", str(oc_run), "--sweagent-run", str(sa_run)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "scaffold command contrast" in out
    assert "opencode tool histogram" in out
    assert "sweagent command-type histogram" in out
    assert "matched instances" in out
    # apps-00099 is opencode-only (see _build_oc_run/_build_sa_run docstring).
    assert "only in opencode (1): apps-00099" in out
