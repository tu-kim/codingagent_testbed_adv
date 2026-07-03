"""Tests for the APPS workload: src/testbed/apps.py plus its integration
points in runner.py (Workload dataclass, WORKLOADS registry, manifest
naming) and cli.py (_resolve_split).

Conventions mirrored from test_swebench.py (load_samples determinism via
monkeypatching _load_dataset) and test_runner.py (mock OpenCode client,
_stub_pre_clone-style fixtures, tmp_path workspaces). No network, no GPU —
_load_dataset (which imports `datasets`) is always monkeypatched.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import click
from click.testing import CliRunner

from testbed import apps, runner
from testbed.runner import WORKLOADS, _run_one, get_workload, workspace_manifest_path


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

def _fake_problem(problem_id: int, *, difficulty: str = "introductory",
                  starter_code: str = "", fn_name: str | None = None,
                  question: str = "Solve it.") -> dict[str, Any]:
    io: dict[str, Any] = {"inputs": [["1"]], "outputs": [["1"]]}
    if fn_name:
        io["fn_name"] = fn_name
    return {
        "problem_id": problem_id,
        "question": question,
        "starter_code": starter_code,
        "input_output": json.dumps(io),
        "solutions": "[]",
        "difficulty": difficulty,
        "url": f"https://example.com/{problem_id}",
    }


def _fake_train_samples():
    # Deliberately unsorted / non-contiguous ids.
    return [
        _fake_problem(30, difficulty="interview"),
        _fake_problem(5, difficulty="introductory"),
        _fake_problem(17, difficulty="competition"),
    ]


def _fake_test_samples():
    return [
        _fake_problem(9005, difficulty="competition"),
        _fake_problem(9001, difficulty="introductory"),
        _fake_problem(9002, difficulty="introductory"),
        _fake_problem(9003, difficulty="interview"),
        _fake_problem(9004, difficulty="interview"),
    ]


def _install_dataset(monkeypatch):
    """Route apps._load_dataset(hf_id, split) -> the fake train/test pools."""
    def _loader(hf_id, split):
        assert hf_id == "codeparrot/apps"
        if split == "train":
            return _fake_train_samples()
        if split == "test":
            return _fake_test_samples()
        raise AssertionError(f"unexpected hf split {split!r}")

    monkeypatch.setattr(apps, "_load_dataset", _loader)


# ---------------------------------------------------------------------------
# load_samples: determinism, filtering, id injection, error paths
# ---------------------------------------------------------------------------

def test_load_samples_deterministic_same_seed(monkeypatch):
    _install_dataset(monkeypatch)
    out1 = apps.load_samples("train", seed=7, n=2)
    out2 = apps.load_samples("train", seed=7, n=2)
    assert [s["instance_id"] for s in out1] == [s["instance_id"] for s in out2]
    assert out1 == out2


def test_load_samples_deterministic_regardless_of_input_ordering(monkeypatch):
    """Same (split, seed, n) must select the same samples even if the
    underlying dataset iterator yields rows in a different order (sort by
    problem_id happens before Random.sample, per the module contract)."""
    def _loader_reversed(hf_id, split):
        assert split == "train"
        return list(reversed(_fake_train_samples()))

    def _loader_forward(hf_id, split):
        assert split == "train"
        return _fake_train_samples()

    monkeypatch.setattr(apps, "_load_dataset", _loader_forward)
    forward = apps.load_samples("train", seed=3, n=2)

    monkeypatch.setattr(apps, "_load_dataset", _loader_reversed)
    reversed_order = apps.load_samples("train", seed=3, n=2)

    assert [s["instance_id"] for s in forward] == [s["instance_id"] for s in reversed_order]


def test_load_samples_difficulty_pseudo_split_filters_and_uses_test_hf_split(monkeypatch):
    """'introductory' must (a) draw from the HF 'test' split only, and
    (b) contain only difficulty == 'introductory' rows."""
    calls: list[str] = []

    def _loader(hf_id, split):
        calls.append(split)
        return _fake_test_samples()

    monkeypatch.setattr(apps, "_load_dataset", _loader)
    out = apps.load_samples("introductory", seed=0, n=2)

    assert calls == ["test"]  # never touched "train"
    assert len(out) == 2
    for s in out:
        assert s["difficulty"] == "introductory"


def test_load_samples_other_difficulty_pseudo_splits(monkeypatch):
    _install_dataset(monkeypatch)
    interview = apps.load_samples("interview", seed=0, n=2)
    competition = apps.load_samples("competition", seed=0, n=1)
    assert all(s["difficulty"] == "interview" for s in interview)
    assert all(s["difficulty"] == "competition" for s in competition)


def test_load_samples_unknown_split_raises(monkeypatch):
    _install_dataset(monkeypatch)
    with pytest.raises(ValueError):
        apps.load_samples("nope", seed=0, n=1)


def test_load_samples_n_too_large_raises(monkeypatch):
    _install_dataset(monkeypatch)
    # "train" fake pool has 3 samples.
    with pytest.raises(ValueError):
        apps.load_samples("train", seed=0, n=4)


def test_load_samples_n_too_large_for_difficulty_filter_raises(monkeypatch):
    """The n-too-large check must apply to the FILTERED pool, not the
    full unfiltered split."""
    _install_dataset(monkeypatch)
    # "introductory" pseudo-split has exactly 2 matching rows in the fake pool.
    with pytest.raises(ValueError):
        apps.load_samples("introductory", seed=0, n=3)


def test_load_samples_injects_zero_padded_instance_id(monkeypatch):
    _install_dataset(monkeypatch)
    out = apps.load_samples("train", seed=0, n=3)
    for s in out:
        assert s["instance_id"] == f"apps-{int(s['problem_id']):05d}"
        assert len(s["instance_id"]) == len("apps-") + 5


def test_load_samples_zero_n_returns_empty(monkeypatch):
    _install_dataset(monkeypatch)
    assert apps.load_samples("train", seed=0, n=0) == []


# ---------------------------------------------------------------------------
# parse_input_output
# ---------------------------------------------------------------------------

def test_parse_input_output_call_based_with_fn_name():
    sample = {"input_output": json.dumps({
        "inputs": [[1, 2]], "outputs": [[3]], "fn_name": "add",
    })}
    out = apps.parse_input_output(sample)
    assert out == {"inputs": [[1, 2]], "outputs": [[3]], "fn_name": "add"}


def test_parse_input_output_stdio_without_fn_name():
    sample = {"input_output": json.dumps({"inputs": [["1\n"]], "outputs": [["1\n"]]})}
    out = apps.parse_input_output(sample)
    assert out["fn_name"] is None
    assert out["inputs"] == [["1\n"]]
    assert out["outputs"] == [["1\n"]]


def test_parse_input_output_malformed_json_is_tolerated():
    sample = {"input_output": "{not valid json,,,"}
    out = apps.parse_input_output(sample)
    assert out == {"inputs": [], "outputs": [], "fn_name": None}


def test_parse_input_output_empty_string_is_tolerated():
    sample = {"input_output": ""}
    out = apps.parse_input_output(sample)
    assert out == {"inputs": [], "outputs": [], "fn_name": None}


def test_parse_input_output_missing_key_is_tolerated():
    sample = {}  # no "input_output" key at all
    out = apps.parse_input_output(sample)
    assert out == {"inputs": [], "outputs": [], "fn_name": None}


def test_parse_input_output_non_dict_json_is_tolerated():
    """A malformed-but-valid-JSON payload (e.g. a bare list) must not crash."""
    sample = {"input_output": json.dumps([1, 2, 3])}
    out = apps.parse_input_output(sample)
    assert out == {"inputs": [], "outputs": [], "fn_name": None}


def test_parse_input_output_empty_fn_name_string_treated_as_absent():
    """fn_name: "" is falsy -> treated as no function name (stdio mode)."""
    sample = {"input_output": json.dumps({"inputs": [], "outputs": [], "fn_name": ""})}
    out = apps.parse_input_output(sample)
    assert out["fn_name"] is None


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------

def test_render_prompt_stdio_mode_has_stdin_stdout_block_not_fn_name():
    sample = _fake_problem(1, starter_code="", fn_name=None, question="Print hello.")
    out = apps.render_prompt(sample)
    assert "STDIN/STDOUT" in out
    assert "FUNCTION-CALL" not in out
    assert "Print hello." in out


def test_render_prompt_call_mode_names_fn_name_not_stdio_block():
    sample = _fake_problem(2, starter_code="def add(a, b):\n    pass\n",
                           fn_name="add", question="Add two numbers.")
    out = apps.render_prompt(sample)
    assert "FUNCTION-CALL" in out
    assert "`add`" in out
    assert "STDIN/STDOUT" not in out
    assert "Add two numbers." in out


def test_render_prompt_includes_question_text_verbatim_stripped():
    sample = _fake_problem(3, question="  \n  Compute the sum.  \n  ")
    out = apps.render_prompt(sample)
    assert "Compute the sum." in out


def test_render_prompt_mentions_problem_and_solution_filenames():
    sample = _fake_problem(4)
    out = apps.render_prompt(sample)
    assert "PROBLEM.md" in out
    assert "solution.py" in out


# ---------------------------------------------------------------------------
# prepare_workspace
# ---------------------------------------------------------------------------

async def test_prepare_workspace_fresh_dir_stdio_gets_scaffold(tmp_path: Path):
    sample = _fake_problem(1, starter_code="", fn_name=None, question="Q1")
    dest = tmp_path / "ws1"
    await apps.prepare_workspace(sample, dest)

    assert (dest / apps.PROBLEM_FILE).exists()
    assert (dest / apps.SOLUTION_FILE).exists()
    assert (dest / apps.PROBLEM_FILE).read_text().strip() == "Q1"
    solution = (dest / apps.SOLUTION_FILE).read_text()
    assert "Write your Python 3 solution" in solution


async def test_prepare_workspace_fresh_dir_call_based_gets_starter_code_verbatim(tmp_path: Path):
    starter = "def add(a, b):\n    pass\n"
    sample = _fake_problem(2, starter_code=starter, fn_name="add", question="Q2")
    dest = tmp_path / "ws2"
    await apps.prepare_workspace(sample, dest)

    solution = (dest / apps.SOLUTION_FILE).read_text()
    assert solution == starter
    assert "Write your Python 3 solution" not in solution


async def test_prepare_workspace_no_reset_existing_with_marker_is_noop(tmp_path: Path):
    """reset=False + PROBLEM.md already present: agent-created extra files
    and solution.py edits must survive untouched."""
    sample = _fake_problem(3, question="Original question")
    dest = tmp_path / "ws3"
    await apps.prepare_workspace(sample, dest)

    # Simulate agent activity: edits solution.py, creates a new file.
    (dest / apps.SOLUTION_FILE).write_text("# agent's edited solution\n")
    (dest / "notes.txt").write_text("scratch notes")

    # A "changed" sample object (e.g. same instance re-fetched) must not
    # overwrite anything when reset=False and the marker exists.
    changed_sample = _fake_problem(3, question="A DIFFERENT question")
    await apps.prepare_workspace(changed_sample, dest, reset=False)

    assert (dest / apps.SOLUTION_FILE).read_text() == "# agent's edited solution\n"
    assert (dest / "notes.txt").read_text() == "scratch notes"
    # PROBLEM.md also untouched (still the original question).
    assert (dest / apps.PROBLEM_FILE).read_text().strip() == "Original question"


async def test_prepare_workspace_reset_true_wipes_extras_and_restores_initial(tmp_path: Path):
    sample = _fake_problem(4, starter_code="", fn_name=None, question="Q4 original")
    dest = tmp_path / "ws4"
    await apps.prepare_workspace(sample, dest)

    (dest / apps.SOLUTION_FILE).write_text("# agent garbage\n")
    (dest / "extra_agent_file.py").write_text("junk")

    await apps.prepare_workspace(sample, dest, reset=True)

    assert not (dest / "extra_agent_file.py").exists()
    solution = (dest / apps.SOLUTION_FILE).read_text()
    assert "Write your Python 3 solution" in solution
    assert "agent garbage" not in solution
    assert (dest / apps.PROBLEM_FILE).read_text().strip() == "Q4 original"


async def test_prepare_workspace_existing_dir_without_marker_gets_rewritten(tmp_path: Path):
    """An existing dir missing PROBLEM.md (interrupted materialization, or
    just some other pre-existing directory) must get both files written,
    even with reset=False."""
    sample = _fake_problem(5, question="Q5")
    dest = tmp_path / "ws5"
    dest.mkdir(parents=True)
    (dest / "unrelated.txt").write_text("pre-existing junk")

    await apps.prepare_workspace(sample, dest, reset=False)

    assert (dest / apps.PROBLEM_FILE).read_text().strip() == "Q5"
    assert (dest / apps.SOLUTION_FILE).exists()
    # Unrelated pre-existing file is not required to be removed (only
    # reset=True guarantees a wipe); the load-bearing assertion is that
    # the two task files now exist and are correct.


async def test_prepare_workspace_is_idempotent_across_two_no_reset_calls(tmp_path: Path):
    """Determinism check: calling prepare_workspace twice with reset=False
    on a fresh dest produces identical PROBLEM.md/solution.py content."""
    sample = _fake_problem(6, starter_code="def f():\n    pass\n", fn_name="f")
    dest1 = tmp_path / "ws6a"
    dest2 = tmp_path / "ws6b"
    await apps.prepare_workspace(sample, dest1, reset=False)
    await apps.prepare_workspace(sample, dest2, reset=False)

    assert (dest1 / apps.PROBLEM_FILE).read_text() == (dest2 / apps.PROBLEM_FILE).read_text()
    assert (dest1 / apps.SOLUTION_FILE).read_text() == (dest2 / apps.SOLUTION_FILE).read_text()


# ---------------------------------------------------------------------------
# workspace_manifest_path: legacy swebench vs. apps prefixing
# ---------------------------------------------------------------------------

def test_workspace_manifest_path_legacy_four_arg_call_unchanged(tmp_path: Path):
    """Calling with the original 4 positional/keyword args (no workload=)
    must still produce the legacy swebench filename -- backward compat
    with already-written manifests."""
    p = workspace_manifest_path(tmp_path, "lite", 42, 10)
    assert p == tmp_path / ".workspaces-lite-s42-n10.json"


def test_workspace_manifest_path_explicit_swebench_matches_legacy(tmp_path: Path):
    p = workspace_manifest_path(tmp_path, "lite", 42, 10, workload="swebench")
    assert p == tmp_path / ".workspaces-lite-s42-n10.json"


def test_workspace_manifest_path_apps_gets_prefixed_name(tmp_path: Path):
    p = workspace_manifest_path(tmp_path, "test", 42, 10, workload="apps")
    assert p == tmp_path / ".workspaces-apps-test-s42-n10.json"


def test_workspace_manifest_path_apps_and_swebench_distinct_for_same_tuple(tmp_path: Path):
    """Even for a (split, seed, n) tuple that happens to overlap in name
    (not realistic since split-name sets don't intersect, but this is the
    belt-and-suspenders guarantee), apps and swebench manifests must not
    collide on the same path."""
    swebench_path = workspace_manifest_path(tmp_path, "test", 1, 5, workload="swebench")
    apps_path = workspace_manifest_path(tmp_path, "test", 1, 5, workload="apps")
    assert swebench_path != apps_path


# ---------------------------------------------------------------------------
# get_workload / WORKLOADS registry
# ---------------------------------------------------------------------------

def test_get_workload_unknown_raises_value_error():
    with pytest.raises(ValueError):
        get_workload("not-a-real-workload")


def test_registry_has_swebench_and_apps_with_correct_default_split():
    assert set(WORKLOADS) >= {"swebench", "apps"}
    assert WORKLOADS["swebench"].default_split == "lite"
    assert WORKLOADS["apps"].default_split == "test"


def test_registry_entries_wire_up_apps_module_functions():
    wl = get_workload("apps")
    assert wl.load_samples is apps.load_samples
    assert wl.render_prompt is apps.render_prompt
    assert wl.prepare is apps.prepare_workspace
    assert wl.splits == apps.SPLITS


def test_get_workload_returns_same_object_as_registry_lookup():
    assert get_workload("apps") is WORKLOADS["apps"]
    assert get_workload("swebench") is WORKLOADS["swebench"]


# ---------------------------------------------------------------------------
# runner integration: _run_one with the apps Workload, mocked OpenCode client
# ---------------------------------------------------------------------------

class _FakeClient:
    """Minimal stand-in for OpenCodeClient (mirrors test_runner.py's fake)."""

    def __init__(self) -> None:
        self.create_calls: list[str] = []
        self.send_calls: list[tuple[str, str, str]] = []
        self.list_calls: list[tuple[str, str]] = []

    async def create_session(self, directory: str) -> str:
        self.create_calls.append(directory)
        return "ses_apps_test"

    async def send_message(self, session_id: str, prompt: str, directory: str) -> dict:
        self.send_calls.append((session_id, prompt, directory))
        return {"info": {"id": "msg_x"}, "parts": []}

    async def list_messages(self, session_id: str, directory: str) -> list[dict]:
        self.list_calls.append((session_id, directory))
        return [{"info": {}, "parts": []}]


_APPS_SAMPLE = _fake_problem(42, starter_code="", fn_name=None, question="Print the input back.")
_APPS_SAMPLE["instance_id"] = apps.instance_id_for(_APPS_SAMPLE)


async def test_run_one_apps_workload_materializes_files_and_sends_prompt(tmp_path: Path):
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    wl = WORKLOADS["apps"]

    rec = await _run_one(client, _APPS_SAMPLE, 0.0, tmp_path, sem, workload=wl)

    assert rec.success is True
    assert rec.error is None

    dest = tmp_path / rec.directory
    assert (dest / apps.PROBLEM_FILE).exists(), "stage 1 must materialize PROBLEM.md"
    assert (dest / apps.SOLUTION_FILE).exists()
    assert (dest / apps.PROBLEM_FILE).read_text().strip() == "Print the input back."

    # The prompt actually sent to OpenCode must be the APPS-rendered prompt,
    # carrying the stdio mode block (fn_name is None for this sample).
    sent_prompt = client.send_calls[0][1]
    assert "STDIN/STDOUT" in sent_prompt
    assert "FUNCTION-CALL" not in sent_prompt
    assert "Print the input back." in sent_prompt


async def test_run_one_apps_workload_call_based_prompt_names_function(tmp_path: Path):
    sample = _fake_problem(43, starter_code="def solve(x):\n    pass\n",
                           fn_name="solve", question="Return x doubled.")
    sample["instance_id"] = apps.instance_id_for(sample)
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    wl = WORKLOADS["apps"]

    rec = await _run_one(client, sample, 0.0, tmp_path, sem, workload=wl)

    assert rec.success is True
    sent_prompt = client.send_calls[0][1]
    assert "FUNCTION-CALL" in sent_prompt
    assert "`solve`" in sent_prompt

    dest = tmp_path / rec.directory
    assert (dest / apps.SOLUTION_FILE).read_text() == "def solve(x):\n    pass\n"


async def test_run_one_apps_workload_clone_stage_failure_on_prepare_error(tmp_path: Path, monkeypatch):
    """A prepare() failure for the apps workload must land as
    error.stage='clone', same trace-schema name as swebench (see the
    Workload docstring in runner.py)."""
    async def _boom(sample, dest, *, reset=False):
        raise OSError("disk full")

    wl = WORKLOADS["apps"]
    broken_wl = runner.Workload(
        name=wl.name, default_split=wl.default_split, splits=wl.splits,
        load_samples=wl.load_samples, render_prompt=wl.render_prompt,
        prepare=_boom,
    )
    client = _FakeClient()
    sem = asyncio.Semaphore(1)

    rec = await _run_one(client, _APPS_SAMPLE, 0.0, tmp_path, sem, workload=broken_wl)

    assert rec.success is False
    assert rec.error and rec.error["stage"] == "clone"
    assert client.create_calls == []


async def test_run_one_apps_workload_reset_forwards_reset_kwarg(tmp_path: Path):
    """reset_workspace=True must flow through to apps.prepare_workspace,
    producing the stable session-<instance_id> directory name (no uuid)."""
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    wl = WORKLOADS["apps"]

    rec1 = await _run_one(client, _APPS_SAMPLE, 0.0, tmp_path, sem,
                          reset_workspace=True, workload=wl)
    rec2 = await _run_one(client, _APPS_SAMPLE, 0.0, tmp_path, sem,
                          reset_workspace=True, workload=wl)

    assert rec1.directory == f"session-{_APPS_SAMPLE['instance_id']}"
    assert rec1.directory == rec2.directory


# ---------------------------------------------------------------------------
# cli._resolve_split
# ---------------------------------------------------------------------------

def test_resolve_split_none_uses_swebench_default():
    assert runner.get_workload("swebench").default_split == "lite"
    from testbed.cli import _resolve_split
    assert _resolve_split("swebench", None) == "lite"


def test_resolve_split_none_uses_apps_default():
    from testbed.cli import _resolve_split
    assert _resolve_split("apps", None) == "test"


def test_resolve_split_wrong_workload_split_raises_bad_parameter():
    from testbed.cli import _resolve_split
    with pytest.raises(click.BadParameter):
        _resolve_split("apps", "lite")  # 'lite' belongs to swebench
    with pytest.raises(click.BadParameter):
        _resolve_split("swebench", "introductory")  # apps-only pseudo-split


def test_resolve_split_valid_split_passes_through():
    from testbed.cli import _resolve_split
    assert _resolve_split("swebench", "verified") == "verified"
    assert _resolve_split("apps", "introductory") == "introductory"
    assert _resolve_split("apps", "train") == "train"
