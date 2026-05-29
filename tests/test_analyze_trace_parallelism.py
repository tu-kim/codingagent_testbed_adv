"""Tests for scripts/analyze_trace_parallelism.py.

Pure file-based parsing -- no network, no GPU, no external services.
Covers: parallel batch detection inside step-start/step-finish bounds,
task tool spawn recognition, per-instance step accounting, CSV output.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_trace_parallelism.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_trace_parallelism", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_trace_parallelism"] = module
    spec.loader.exec_module(module)
    return module


def _write_trace(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def _msg(role: str, parts: list[dict]) -> dict:
    return {"info": {"role": role}, "parts": parts}


def _step(*tools: str | tuple[str, str | dict]) -> list[dict]:
    """Build a step from a tool list. Each element is either a tool name
    (str) or a (name, input) tuple for the task tool's input payload."""
    out: list[dict] = [{"type": "step-start"}]
    for t in tools:
        if isinstance(t, tuple):
            name, inp = t
            out.append({"type": "tool", "tool": name,
                        "callID": f"call_{name}", "state": {"input": inp}})
        else:
            out.append({"type": "tool", "tool": t, "callID": f"call_{t}"})
    out.append({"type": "step-finish"})
    return out


# ---------- parallel batch detection ----------


def test_single_tool_in_step_is_not_parallel(mod, tmp_path):
    """A step with exactly one tool call doesn't count as a parallel
    batch -- only steps with >= 2 tools do. This pins the threshold
    so a refactor doesn't silently shift it to >0."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "django__django-1",
        "messages": [_msg("assistant", _step("read"))],
    }])
    batches, _, steps, par = mod.analyze_trace(p)
    assert batches == []
    assert steps["django__django-1"] == 1
    assert par == {}


def test_two_tools_in_step_becomes_one_parallel_batch(mod, tmp_path):
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "test-1",
        "messages": [_msg("assistant", _step("read", "grep"))],
    }])
    batches, _, steps, par = mod.analyze_trace(p)
    assert len(batches) == 1
    assert batches[0].instance_id == "test-1"
    assert batches[0].tools == ["read", "grep"]   # original order preserved
    assert steps["test-1"] == 1
    assert par["test-1"] == 1


def test_step_idx_increments_within_message(mod, tmp_path):
    """A single assistant message may contain multiple steps (one per
    LLM call inside the same turn boundary). step_idx is 0 for the
    first parallel batch, 1 for the second, etc. -- so CSV consumers
    can recover the original ordering."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t",
        "messages": [_msg("assistant",
                          _step("read", "grep") + _step("edit") + _step("read", "read"))],
    }])
    batches, _, _, _ = mod.analyze_trace(p)
    # Two parallel batches -- step indices 0 and 2 (step 1 was single-tool).
    assert [(b.step_idx, b.tools) for b in batches] == [
        (0, ["read", "grep"]),
        (2, ["read", "read"]),
    ]


def test_tools_between_message_boundaries_are_not_joined(mod, tmp_path):
    """Two assistant messages, each with a single tool. The walker
    must NOT pool their tools into one batch -- step-start resets
    the accumulator inside each message."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t",
        "messages": [
            _msg("assistant", _step("read")),
            _msg("assistant", _step("grep")),
        ],
    }])
    batches, _, _, _ = mod.analyze_trace(p)
    assert batches == []


def test_user_messages_ignored(mod, tmp_path):
    """Tools never appear under user-role messages; the walker must
    skip them anyway since the role gate is the only thing keeping
    title/summary agent messages out."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t",
        "messages": [
            _msg("user", _step("read", "grep")),    # impossible IRL but
            _msg("assistant", _step("read", "edit")),
        ],
    }])
    batches, _, _, _ = mod.analyze_trace(p)
    assert len(batches) == 1
    assert batches[0].tools == ["read", "edit"]


def test_tool_outside_step_is_dropped(mod, tmp_path):
    """A tool part with no preceding step-start (malformed / truncated
    message) gets dropped silently rather than crashing."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t",
        "messages": [_msg("assistant", [
            {"type": "tool", "tool": "orphan_before_step"},
            {"type": "step-start"},
            {"type": "tool", "tool": "read"},
            {"type": "tool", "tool": "edit"},
            {"type": "step-finish"},
            {"type": "tool", "tool": "orphan_after_step"},
        ])],
    }])
    batches, _, _, _ = mod.analyze_trace(p)
    assert len(batches) == 1
    assert batches[0].tools == ["read", "edit"]


def test_assistant_message_with_empty_parts_is_no_op(mod, tmp_path):
    """An assistant message with an explicit empty `parts: []`
    (possible when the agent loop emits a step that errored before
    any tool/text part was produced) must not crash and must not
    create a phantom step."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t",
        "messages": [{"info": {"role": "assistant"}, "parts": []}],
    }])
    batches, spawns, steps, par = mod.analyze_trace(p)
    assert batches == []
    assert spawns == []
    assert steps == {}
    assert par == {}


def test_assistant_tool_parts_without_any_step_markers_are_dropped(mod, tmp_path):
    """If an assistant message contains tool parts but NO step-start
    or step-finish (malformed upstream), every tool is gated out by
    `in_step` and no step is counted. The `in_step` gate is therefore
    the only defense against pre-step-recording opencode versions
    leaking tools into the analysis."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t",
        "messages": [{"info": {"role": "assistant"}, "parts": [
            {"type": "tool", "tool": "read"},
            {"type": "tool", "tool": "edit"},
        ]}],
    }])
    batches, _, steps, _ = mod.analyze_trace(p)
    assert batches == []
    assert steps == {}


def test_step_without_finish_is_not_counted(mod, tmp_path):
    """A step-start with no matching step-finish (crashed mid-turn)
    must not count toward steps_per_task -- otherwise the parallel
    rate denominator drifts."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t",
        "messages": [_msg("assistant", [
            {"type": "step-start"},
            {"type": "tool", "tool": "read"},
            {"type": "tool", "tool": "edit"},
            # no step-finish
        ])],
    }])
    batches, _, steps, _ = mod.analyze_trace(p)
    assert batches == []
    assert steps == {}


# ---------- task tool spawn detection ----------


def test_task_tool_invocation_recorded(mod, tmp_path):
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "iid-1",
        "messages": [_msg("assistant", _step(
            ("task", {"description": "Find all model save() overrides"}),
        ))],
    }])
    _, spawns, _, _ = mod.analyze_trace(p)
    assert len(spawns) == 1
    assert spawns[0].instance_id == "iid-1"
    assert spawns[0].call_id == "call_task"
    assert "save() overrides" in spawns[0].description


def test_task_input_description_falls_back_through_keys(mod, tmp_path):
    """opencode has used `description` / `prompt` / `instructions` as
    the human-readable input field across versions. The script tries
    each in order, falling back to a json repr."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t",
        "messages": [_msg("assistant",
                          _step(("task", {"prompt": "Find tests"})) +
                          _step(("task", {"instructions": "Run linter"})) +
                          _step(("task", {"foo": "bar"})))],
    }])
    _, spawns, _, _ = mod.analyze_trace(p)
    descs = [s.description for s in spawns]
    assert "Find tests" in descs[0]
    assert "Run linter" in descs[1]
    # Unknown shape falls back to JSON repr.
    assert "foo" in descs[2] and "bar" in descs[2]


def test_multiple_task_tools_in_one_step_detected(mod, tmp_path):
    """Two `task` tools in a single LLM response = two sub-agents
    spawned concurrently. Both must register as TaskSpawns AND the
    batch's tools list must contain both, so the summary's
    `tools.count("task") > 1` filter catches it."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t",
        "messages": [_msg("assistant", _step(
            ("task", {"description": "agent A: find tests"}),
            ("task", {"description": "agent B: find models"}),
        ))],
    }])
    batches, spawns, _, _ = mod.analyze_trace(p)
    assert len(batches) == 1
    assert batches[0].tools == ["task", "task"]
    assert batches[0].tools.count("task") == 2   # the multi-subagent signal
    assert len(spawns) == 2
    assert {s.description for s in spawns} == {
        "agent A: find tests", "agent B: find models",
    }
    # callIDs share the synthetic name in the fixture but both recorded.
    assert all(s.instance_id == "t" for s in spawns)


def test_task_tool_inside_parallel_batch(mod, tmp_path):
    """A task tool called alongside other tools in the same step
    is both: a member of a parallel batch AND a sub-agent spawn."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t",
        "messages": [_msg("assistant", _step(
            "read",
            ("task", {"description": "investigate ORM"}),
            "grep",
        ))],
    }])
    batches, spawns, _, _ = mod.analyze_trace(p)
    assert len(batches) == 1
    assert batches[0].tools == ["read", "task", "grep"]
    assert len(spawns) == 1
    assert spawns[0].description == "investigate ORM"


# ---------- per-task accounting ----------


def test_steps_and_parallel_counts_partition_by_instance_id(mod, tmp_path):
    """Two TaskRecords -- counts MUST stay separated per instance_id
    so cross-task comparisons (reproducibility diffs) work."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [
        {"instance_id": "task-A", "messages": [_msg("assistant",
                                                     _step("read") +
                                                     _step("read", "grep"))]},
        {"instance_id": "task-B", "messages": [_msg("assistant",
                                                     _step("read", "edit") +
                                                     _step("read", "edit", "grep"))]},
    ])
    _, _, steps, par = mod.analyze_trace(p)
    assert steps == {"task-A": 2, "task-B": 2}
    assert par == {"task-A": 1, "task-B": 2}


def test_malformed_json_lines_skipped(mod, tmp_path):
    """trace.jsonl is written line-by-line with flush; a SIGKILL can
    leave a half-written final line. The analyzer must skip rather
    than crash."""
    p = tmp_path / "trace.jsonl"
    p.write_text(
        json.dumps({"instance_id": "ok-1", "messages": [_msg("assistant", _step("read", "grep"))]}) + "\n"
        + "this is not json\n"
        + json.dumps({"instance_id": "ok-2", "messages": [_msg("assistant", _step("edit"))]}) + "\n"
    )
    batches, _, steps, _ = mod.analyze_trace(p)
    assert len(batches) == 1
    assert batches[0].instance_id == "ok-1"
    assert steps == {"ok-1": 1, "ok-2": 1}


# ---------- CSV output ----------


def test_parallel_batches_csv_round_trip(mod, tmp_path):
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t-7",
        "messages": [_msg("assistant", _step("read", "grep", "edit"))],
    }])
    out = tmp_path / "out"
    rc = mod.main(["--trace", str(p), "--output", str(out)])
    assert rc == 0
    rows = list(csv.DictReader((out / "parallel_batches.csv").open()))
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "t-7"
    assert rows[0]["n_tools"] == "3"
    assert rows[0]["tools"] == "read,grep,edit"


def test_task_spawns_csv_round_trip(mod, tmp_path):
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [{
        "instance_id": "t-9",
        "messages": [_msg("assistant", _step(
            ("task", {"description": "look into recursive FK"}),
        ))],
    }])
    out = tmp_path / "out"
    rc = mod.main(["--trace", str(p), "--output", str(out)])
    assert rc == 0
    rows = list(csv.DictReader((out / "task_spawns.csv").open()))
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "t-9"
    assert rows[0]["callID"] == "call_task"
    assert "recursive FK" in rows[0]["description"]


def test_main_returns_nonzero_when_trace_missing(mod, tmp_path, capsys):
    rc = mod.main(["--trace", str(tmp_path / "does-not-exist"),
                   "--output", str(tmp_path / "out")])
    assert rc == 2
    assert "trace not found" in capsys.readouterr().err
