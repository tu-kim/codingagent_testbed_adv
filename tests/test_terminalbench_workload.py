"""Tests for the Terminal-Bench workload: src/testbed/terminalbench.py plus
its integration points in runner.py (WORKLOADS registry, manifest naming)
and cli.py (_resolve_split).

Conventions mirrored from test_apps_workload.py. No network, no GPU:
selection tests monkeypatch terminalbench._load_tasks with an in-memory task
pool; parsing tests monkeypatch terminalbench._snapshot_dataset with a
tmp_path fake dataset tree (so the task.toml/instruction.md parsing is
exercised without huggingface_hub).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import click

from testbed import runner, terminalbench
from testbed.runner import WORKLOADS, _run_one, get_workload, workspace_manifest_path


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

def _fake_task(task_id: str, *, difficulty: str | None = "medium",
               category: str = "games", tags: tuple[str, ...] = (),
               instruction: str | None = None) -> dict[str, Any]:
    return {
        "task_id": task_id,
        "instruction": (instruction if instruction is not None
                        else f"Complete the {task_id} task."),
        "difficulty": difficulty,
        "category": category,
        "tags": list(tags),
    }


def _fake_pool() -> list[dict[str, Any]]:
    # Deliberately unsorted; mixed difficulties.
    return [
        _fake_task("regex-log", difficulty="easy"),
        _fake_task("build-linux", difficulty="hard"),
        _fake_task("chess-best-move", difficulty="medium"),
        _fake_task("dna-assembly", difficulty="medium"),
        _fake_task("acl-fix", difficulty="easy"),
    ]


def _install_tasks(monkeypatch, pool: list[dict[str, Any]] | None = None) -> None:
    """Route terminalbench._load_tasks() -> the fake pool (fresh dict copies
    per call, matching the real loader which re-parses files each call)."""
    tasks = pool if pool is not None else _fake_pool()
    monkeypatch.setattr(terminalbench, "_load_tasks",
                        lambda: [dict(t) for t in tasks])


def _write_task_dir(root: Path, task_id: str, *, difficulty: str = "medium",
                    instruction: str = "Do the thing.",
                    toml_text: str | None = None,
                    write_instruction: bool = True) -> Path:
    d = root / task_id
    d.mkdir(parents=True)
    if toml_text is None:
        toml_text = (
            'version = "1.0"\n\n'
            "[metadata]\n"
            'author_email = "a@example.com"\n'
            f'difficulty = "{difficulty}"\n'
            'category = "software-engineering"\n'
            'tags = ["git", "recovery"]\n\n'
            "[agent]\n"
            "timeout_sec = 900.0\n"
        )
    (d / "task.toml").write_text(toml_text, encoding="utf-8")
    if write_instruction:
        (d / "instruction.md").write_text(instruction, encoding="utf-8")
    return d


# ---------------------------------------------------------------------------
# _load_tasks: Harbor task-dir parsing (against a fake snapshot tree)
# ---------------------------------------------------------------------------

def test_load_tasks_parses_metadata_and_instruction(monkeypatch, tmp_path: Path):
    _write_task_dir(tmp_path, "git-leak-recovery", difficulty="hard",
                    instruction="Recover the leaked secret.\n")
    # Root-level plain files (README.md, .gitattributes in the real repo)
    # must not be mistaken for tasks.
    (tmp_path / "README.md").write_text("dataset readme")
    monkeypatch.setattr(terminalbench, "_snapshot_dataset", lambda: tmp_path)

    tasks = terminalbench._load_tasks()

    assert len(tasks) == 1
    t = tasks[0]
    assert t["task_id"] == "git-leak-recovery"
    assert t["instruction"] == "Recover the leaked secret.\n"
    assert t["difficulty"] == "hard"
    assert t["category"] == "software-engineering"
    assert t["tags"] == ["git", "recovery"]


def test_load_tasks_skips_dir_missing_instruction(monkeypatch, tmp_path: Path):
    _write_task_dir(tmp_path, "with-instruction")
    _write_task_dir(tmp_path, "without-instruction", write_instruction=False)
    monkeypatch.setattr(terminalbench, "_snapshot_dataset", lambda: tmp_path)

    tasks = terminalbench._load_tasks()

    assert [t["task_id"] for t in tasks] == ["with-instruction"]


def test_load_tasks_tolerates_malformed_or_metadata_less_toml(monkeypatch, tmp_path: Path):
    _write_task_dir(tmp_path, "broken-toml", toml_text="not [ valid toml ===")
    _write_task_dir(tmp_path, "no-metadata-section", toml_text='version = "1.0"\n')
    # Valid TOML whose `metadata` key is a scalar, not a table -- exercises
    # the isinstance(meta, dict) guard.
    _write_task_dir(tmp_path, "scalar-metadata", toml_text='metadata = "oops"\n')
    monkeypatch.setattr(terminalbench, "_snapshot_dataset", lambda: tmp_path)

    tasks = terminalbench._load_tasks()

    assert [t["task_id"] for t in tasks] == [
        "broken-toml", "no-metadata-section", "scalar-metadata"]
    for t in tasks:
        assert t["difficulty"] is None
        assert t["category"] is None
        assert t["tags"] == []


def test_load_tasks_output_is_sorted_by_task_id(monkeypatch, tmp_path: Path):
    for tid in ("zeta-task", "alpha-task", "mid-task"):
        _write_task_dir(tmp_path, tid)
    monkeypatch.setattr(terminalbench, "_snapshot_dataset", lambda: tmp_path)

    tasks = terminalbench._load_tasks()

    assert [t["task_id"] for t in tasks] == ["alpha-task", "mid-task", "zeta-task"]


# ---------------------------------------------------------------------------
# load_samples: determinism, filtering, id injection, error paths
# ---------------------------------------------------------------------------

def test_load_samples_deterministic_same_seed(monkeypatch):
    _install_tasks(monkeypatch)
    out1 = terminalbench.load_samples("test", seed=7, n=3)
    out2 = terminalbench.load_samples("test", seed=7, n=3)
    assert [s["instance_id"] for s in out1] == [s["instance_id"] for s in out2]
    assert out1 == out2


def test_load_samples_deterministic_regardless_of_input_ordering(monkeypatch):
    """Same (split, seed, n) must select the same samples even if the
    underlying loader yields tasks in a different order (sort by task_id
    happens before Random.sample, per the module contract)."""
    _install_tasks(monkeypatch, _fake_pool())
    forward = terminalbench.load_samples("test", seed=3, n=2)

    _install_tasks(monkeypatch, list(reversed(_fake_pool())))
    reversed_order = terminalbench.load_samples("test", seed=3, n=2)

    assert [s["instance_id"] for s in forward] == [s["instance_id"] for s in reversed_order]


def test_load_samples_difficulty_pseudo_split_filters(monkeypatch):
    _install_tasks(monkeypatch)
    easy = terminalbench.load_samples("easy", seed=0, n=2)
    medium = terminalbench.load_samples("medium", seed=0, n=2)
    hard = terminalbench.load_samples("hard", seed=0, n=1)
    assert all(s["difficulty"] == "easy" for s in easy)
    assert all(s["difficulty"] == "medium" for s in medium)
    assert all(s["difficulty"] == "hard" for s in hard)


def test_load_samples_test_split_includes_unknown_difficulty(monkeypatch):
    """Tasks with a missing/odd difficulty stay selectable via "test" (the
    unfiltered pool) even though no pseudo-split matches them."""
    pool = [_fake_task("weird", difficulty=None), _fake_task("normal", difficulty="easy")]
    _install_tasks(monkeypatch, pool)
    out = terminalbench.load_samples("test", seed=0, n=2)
    assert {s["task_id"] for s in out} == {"weird", "normal"}


def test_load_samples_unknown_split_raises(monkeypatch):
    _install_tasks(monkeypatch)
    with pytest.raises(ValueError):
        terminalbench.load_samples("lite", seed=0, n=1)


def test_load_samples_negative_n_raises(monkeypatch):
    _install_tasks(monkeypatch)
    with pytest.raises(ValueError):
        terminalbench.load_samples("test", seed=0, n=-1)


def test_load_samples_n_too_large_raises(monkeypatch):
    _install_tasks(monkeypatch)  # pool has 5 tasks
    with pytest.raises(ValueError):
        terminalbench.load_samples("test", seed=0, n=6)


def test_load_samples_n_too_large_for_difficulty_filter_raises(monkeypatch):
    """The n-too-large check must apply to the FILTERED pool, not the
    full task set."""
    _install_tasks(monkeypatch)  # exactly 1 "hard" task in the fake pool
    with pytest.raises(ValueError):
        terminalbench.load_samples("hard", seed=0, n=2)


def test_load_samples_injects_prefixed_instance_id(monkeypatch):
    _install_tasks(monkeypatch)
    out = terminalbench.load_samples("test", seed=0, n=5)
    for s in out:
        assert s["instance_id"] == f"terminalbench-{s['task_id']}"


def test_load_samples_zero_n_returns_empty(monkeypatch):
    _install_tasks(monkeypatch)
    assert terminalbench.load_samples("test", seed=0, n=0) == []


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------

def test_render_prompt_includes_instruction_stripped():
    sample = _fake_task("t1", instruction="  \n  Fix the broken repo.  \n  ")
    out = terminalbench.render_prompt(sample)
    assert "Fix the broken repo." in out


def test_render_prompt_mentions_task_file():
    out = terminalbench.render_prompt(_fake_task("t2"))
    assert "TASK.md" in out


def test_render_prompt_reanchors_absolute_paths_into_workspace():
    """The container the task was authored for does not exist here; the
    prompt must tell the agent to keep absolute paths INSIDE the workspace
    (out-of-tree writes are measurement + host-hygiene noise)."""
    out = terminalbench.render_prompt(_fake_task("t3"))
    assert "INSIDE the workspace" in out
    assert "not write outside the workspace" in out


# ---------------------------------------------------------------------------
# prepare_workspace
# ---------------------------------------------------------------------------

async def test_prepare_workspace_fresh_dir_writes_task_file(tmp_path: Path):
    sample = _fake_task("t1", instruction="Q1")
    dest = tmp_path / "ws1"
    await terminalbench.prepare_workspace(sample, dest)

    assert (dest / terminalbench.TASK_FILE).read_text() == "Q1\n"
    # ONLY the task file: environment/solution/tests must never appear.
    assert [p.name for p in dest.iterdir()] == [terminalbench.TASK_FILE]


async def test_prepare_workspace_no_reset_existing_with_marker_is_noop(tmp_path: Path):
    """reset=False + TASK.md already present: agent-created artifacts must
    survive untouched, and a changed sample must not overwrite TASK.md."""
    sample = _fake_task("t2", instruction="Original task")
    dest = tmp_path / "ws2"
    await terminalbench.prepare_workspace(sample, dest)

    (dest / "app").mkdir()
    (dest / "app" / "data.txt").write_text("agent output")
    (dest / terminalbench.TASK_FILE).write_text("agent-mangled task\n")

    changed = _fake_task("t2", instruction="A DIFFERENT task")
    await terminalbench.prepare_workspace(changed, dest, reset=False)

    assert (dest / "app" / "data.txt").read_text() == "agent output"
    assert (dest / terminalbench.TASK_FILE).read_text() == "agent-mangled task\n"


async def test_prepare_workspace_reset_true_wipes_extras_and_restores(tmp_path: Path):
    sample = _fake_task("t3", instruction="Q3 original")
    dest = tmp_path / "ws3"
    await terminalbench.prepare_workspace(sample, dest)

    (dest / "junk.bin").write_text("junk")
    (dest / terminalbench.TASK_FILE).write_text("mangled")

    await terminalbench.prepare_workspace(sample, dest, reset=True)

    assert not (dest / "junk.bin").exists()
    assert (dest / terminalbench.TASK_FILE).read_text() == "Q3 original\n"


async def test_prepare_workspace_existing_dir_without_marker_gets_rewritten(tmp_path: Path):
    """An existing dir missing TASK.md (interrupted materialization) must
    get the task file written, even with reset=False."""
    sample = _fake_task("t4", instruction="Q4")
    dest = tmp_path / "ws4"
    dest.mkdir(parents=True)
    (dest / "unrelated.txt").write_text("pre-existing")

    await terminalbench.prepare_workspace(sample, dest, reset=False)

    assert (dest / terminalbench.TASK_FILE).read_text() == "Q4\n"


async def test_prepare_workspace_dest_is_a_file_gets_replaced(tmp_path: Path):
    sample = _fake_task("t5", instruction="Q5")
    dest = tmp_path / "ws5"
    dest.write_text("i am a file, not a dir")

    await terminalbench.prepare_workspace(sample, dest, reset=False)

    assert dest.is_dir()
    assert (dest / terminalbench.TASK_FILE).read_text() == "Q5\n"


async def test_prepare_workspace_is_idempotent_across_two_no_reset_calls(tmp_path: Path):
    sample = _fake_task("t6", instruction="Q6")
    dest1 = tmp_path / "ws6a"
    dest2 = tmp_path / "ws6b"
    await terminalbench.prepare_workspace(sample, dest1, reset=False)
    await terminalbench.prepare_workspace(sample, dest2, reset=False)

    assert ((dest1 / terminalbench.TASK_FILE).read_text()
            == (dest2 / terminalbench.TASK_FILE).read_text())


# ---------------------------------------------------------------------------
# workspace_manifest_path: terminalbench prefixing
# ---------------------------------------------------------------------------

def test_workspace_manifest_path_terminalbench_gets_prefixed_name(tmp_path: Path):
    p = workspace_manifest_path(tmp_path, "test", 42, 10, workload="terminalbench")
    assert p == tmp_path / ".workspaces-terminalbench-test-s42-n10.json"


def test_workspace_manifest_path_distinct_from_swebench_and_apps(tmp_path: Path):
    tb = workspace_manifest_path(tmp_path, "test", 1, 5, workload="terminalbench")
    sweb = workspace_manifest_path(tmp_path, "test", 1, 5, workload="swebench")
    apps_p = workspace_manifest_path(tmp_path, "test", 1, 5, workload="apps")
    assert len({tb, sweb, apps_p}) == 3


# ---------------------------------------------------------------------------
# get_workload / WORKLOADS registry
# ---------------------------------------------------------------------------

def test_registry_has_terminalbench_with_correct_default_split():
    assert "terminalbench" in WORKLOADS
    assert WORKLOADS["terminalbench"].default_split == "test"
    assert WORKLOADS["terminalbench"].splits == terminalbench.SPLITS
    assert WORKLOADS["terminalbench"].splits == ("test", "easy", "medium", "hard")


async def test_registry_entries_dispatch_to_terminalbench_module_late_bound(monkeypatch):
    """The registry deliberately wraps the module functions in LATE-BINDING
    lambdas (see the WORKLOADS comment in runner.py), so monkeypatching the
    terminalbench module attributes must take effect through the registry.
    Assert dispatch, not identity."""
    wl = get_workload("terminalbench")
    calls: list[Any] = []

    monkeypatch.setattr(terminalbench, "load_samples",
                        lambda split, seed, n: [("loaded", split, seed, n)])
    monkeypatch.setattr(terminalbench, "render_prompt",
                        lambda sample: f"prompt:{sample['task_id']}")

    async def _prep(sample, dest, *, reset=False):
        calls.append((sample["task_id"], dest, reset))
    monkeypatch.setattr(terminalbench, "prepare_workspace", _prep)

    assert wl.load_samples("test", 1, 0) == [("loaded", "test", 1, 0)]
    assert wl.render_prompt({"task_id": "x"}) == "prompt:x"
    await wl.prepare({"task_id": "x"}, Path("unused"), reset=True)
    assert calls == [("x", Path("unused"), True)]


def test_get_workload_returns_same_object_as_registry_lookup():
    assert get_workload("terminalbench") is WORKLOADS["terminalbench"]


# ---------------------------------------------------------------------------
# runner integration: _run_one with the terminalbench Workload, mocked client
# ---------------------------------------------------------------------------

class _FakeClient:
    """Minimal stand-in for OpenCodeClient (mirrors test_runner.py's fake)."""

    def __init__(self) -> None:
        self.create_calls: list[str] = []
        self.send_calls: list[tuple[str, str, str]] = []
        self.list_calls: list[tuple[str, str]] = []

    async def create_session(self, directory: str) -> str:
        self.create_calls.append(directory)
        return "ses_terminalbench_test"

    async def send_message(self, session_id: str, prompt: str, directory: str) -> dict:
        self.send_calls.append((session_id, prompt, directory))
        return {"info": {"id": "msg_x"}, "parts": []}

    async def list_messages(self, session_id: str, directory: str) -> list[dict]:
        self.list_calls.append((session_id, directory))
        return [{"info": {}, "parts": []}]


def _tb_sample(task_id: str = "chess-best-move",
               instruction: str = "Find the best move.") -> dict[str, Any]:
    sample = _fake_task(task_id, instruction=instruction)
    sample["instance_id"] = terminalbench.instance_id_for(sample)
    return sample


async def test_run_one_terminalbench_materializes_task_and_sends_prompt(tmp_path: Path):
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    wl = WORKLOADS["terminalbench"]
    sample = _tb_sample()

    rec = await _run_one(client, sample, 0.0, tmp_path, sem, workload=wl)

    assert rec.success is True
    assert rec.error is None

    dest = tmp_path / rec.directory
    assert (dest / terminalbench.TASK_FILE).exists(), "stage 1 must materialize TASK.md"
    assert (dest / terminalbench.TASK_FILE).read_text().strip() == "Find the best move."

    # The prompt actually sent to OpenCode must be the terminalbench-rendered
    # prompt: instruction + workspace framing.
    sent_prompt = client.send_calls[0][1]
    assert "Find the best move." in sent_prompt
    assert "TASK.md" in sent_prompt
    assert "INSIDE the workspace" in sent_prompt


async def test_run_one_terminalbench_clone_stage_failure_on_prepare_error(tmp_path: Path):
    """A prepare() failure must land as error.stage='clone', the documented
    trace-schema name for stage-1 workspace preparation (see the Workload
    docstring in runner.py)."""
    async def _boom(sample, dest, *, reset=False):
        raise OSError("disk full")

    wl = WORKLOADS["terminalbench"]
    broken_wl = runner.Workload(
        name=wl.name, default_split=wl.default_split, splits=wl.splits,
        load_samples=wl.load_samples, render_prompt=wl.render_prompt,
        prepare=_boom,
    )
    client = _FakeClient()
    sem = asyncio.Semaphore(1)

    rec = await _run_one(client, _tb_sample(), 0.0, tmp_path, sem, workload=broken_wl)

    assert rec.success is False
    assert rec.error and rec.error["stage"] == "clone"
    assert client.create_calls == []


async def test_run_one_terminalbench_reset_gives_stable_directory(tmp_path: Path):
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    wl = WORKLOADS["terminalbench"]
    sample = _tb_sample()

    rec1 = await _run_one(client, sample, 0.0, tmp_path, sem,
                          reset_workspace=True, workload=wl)
    rec2 = await _run_one(client, sample, 0.0, tmp_path, sem,
                          reset_workspace=True, workload=wl)

    assert rec1.directory == f"session-{sample['instance_id']}"
    assert rec1.directory == rec2.directory


# ---------------------------------------------------------------------------
# cli._resolve_split
# ---------------------------------------------------------------------------

def test_resolve_split_none_uses_terminalbench_default():
    from testbed.cli import _resolve_split
    assert _resolve_split("terminalbench", None) == "test"


def test_resolve_split_wrong_workload_split_raises_bad_parameter():
    from testbed.cli import _resolve_split
    with pytest.raises(click.BadParameter):
        _resolve_split("terminalbench", "lite")  # 'lite' belongs to swebench
    with pytest.raises(click.BadParameter):
        _resolve_split("terminalbench", "introductory")  # apps-only
    with pytest.raises(click.BadParameter):
        _resolve_split("swebench", "easy")  # terminalbench-only


def test_resolve_split_valid_split_passes_through():
    from testbed.cli import _resolve_split
    assert _resolve_split("terminalbench", "test") == "test"
    assert _resolve_split("terminalbench", "easy") == "easy"
    assert _resolve_split("terminalbench", "medium") == "medium"
    assert _resolve_split("terminalbench", "hard") == "hard"
