"""Tests for scripts/run_sweagent_apps.py and scripts/analyze_sweagent_traj.py.

Conventions mirrored from tests/test_extract_predictions.py (env-scrubbed
git helper, skipif when git is absent, real local repos in tmp_path for
apply_patch) and tests/test_analyze_eval_results.py (importlib.util module
loader, scope="module" fixture, sys.argv patching for main()). No network,
no GPU -- the `sweagent` CLI itself is never invoked; subprocess.run is
monkeypatched at the module level for anything that would shell out to it.
apps._load_dataset (HF `datasets` import) is always monkeypatched, same as
tests/test_apps_workload.py.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from testbed import apps

_RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_sweagent_apps.py"
_ANALYZE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_sweagent_traj.py"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not on PATH")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner_mod():
    return _load_module("run_sweagent_apps", _RUNNER_PATH)


@pytest.fixture(scope="module")
def analyze_mod():
    return _load_module("analyze_sweagent_traj", _ANALYZE_PATH)


# ---------------------------------------------------------------------------
# git helper (env-scrubbed, no host git config dependency) -- same
# convention as tests/test_extract_predictions.py / test_pre_clone_git.py
# ---------------------------------------------------------------------------

def _git_env() -> dict[str, str]:
    return {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
        "PATH": os.environ.get("PATH", ""),
    }


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args],
                          capture_output=True, text=True, env=_git_env())


def _git_ok(cwd: Path, *args: str) -> str:
    r = _git(cwd, *args)
    if r.returncode != 0:
        raise RuntimeError(f"git {args!r} failed:\n{r.stderr}")
    return r.stdout.strip()


def _make_repo(path: Path, filename: str = "src.py", content: str = "x = 1\n") -> str:
    """Create a repo with a single commit. Returns the base commit SHA."""
    path.mkdir(parents=True, exist_ok=True)
    _git_ok(path, "init", "-q")
    (path / filename).write_text(content)
    _git_ok(path, "add", filename)
    _git_ok(path, "commit", "-q", "-m", "base")
    return _git_ok(path, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# APPS sample fixture (mirrors test_apps_workload.py's _fake_problem)
# ---------------------------------------------------------------------------

def _fake_problem(problem_id: int, *, difficulty: str = "introductory",
                  starter_code: str = "", fn_name: str | None = None,
                  question: str = "Solve it.") -> dict[str, Any]:
    io: dict[str, Any] = {"inputs": [["1"]], "outputs": [["1"]]}
    if fn_name:
        io["fn_name"] = fn_name
    sample = {
        "problem_id": problem_id,
        "question": question,
        "starter_code": starter_code,
        "input_output": json.dumps(io),
        "solutions": "[]",
        "difficulty": difficulty,
        "url": f"https://example.com/{problem_id}",
    }
    sample["instance_id"] = apps.instance_id_for(sample)
    return sample


# ===========================================================================
# 1. build_sweagent_cmd
# ===========================================================================

def test_build_sweagent_cmd_exact_flags(runner_mod):
    cmd = runner_mod.build_sweagent_cmd(
        model_name="local", api_base="http://127.0.0.1:8000/v1", api_key="dummy",
        ws=Path("/tmp/ws1"), prompt="do the thing", instance_id="apps-00001",
        traj_root=Path("/tmp/trajs"), deployment="local", max_steps=50,
        extra_args=[],
    )
    assert cmd[0] == "sweagent"
    assert cmd[1] == "run"

    def flag_value(flag: str) -> str:
        i = cmd.index(flag)
        return cmd[i + 1]

    assert flag_value("--agent.model.name") == "openai/local"
    assert flag_value("--agent.model.api_base") == "http://127.0.0.1:8000/v1"
    assert flag_value("--agent.model.api_key") == "dummy"
    assert flag_value("--agent.model.per_instance_cost_limit") == "0"
    assert flag_value("--agent.model.total_cost_limit") == "0"
    assert flag_value("--agent.model.per_instance_call_limit") == "50"
    assert flag_value("--env.repo.path") == str(Path("/tmp/ws1"))
    assert flag_value("--env.deployment.type") == "local"
    assert flag_value("--problem_statement.text") == "do the thing"
    assert flag_value("--problem_statement.id") == "apps-00001"
    assert flag_value("--output_dir") == str(Path("/tmp/trajs"))


def test_build_sweagent_cmd_model_name_gets_openai_prefix(runner_mod):
    cmd = runner_mod.build_sweagent_cmd(
        model_name="qwen3-coder-30b-a3b", api_base="http://x/v1", api_key="k",
        ws=Path("/ws"), prompt="p", instance_id="i", traj_root=Path("/t"),
        deployment="docker", max_steps=10, extra_args=[],
    )
    idx = cmd.index("--agent.model.name")
    assert cmd[idx + 1] == "openai/qwen3-coder-30b-a3b"


def test_build_sweagent_cmd_call_limit_is_str_of_max_steps(runner_mod):
    cmd = runner_mod.build_sweagent_cmd(
        model_name="m", api_base="http://x/v1", api_key="k", ws=Path("/ws"),
        prompt="p", instance_id="i", traj_root=Path("/t"), deployment="local",
        max_steps=123, extra_args=[],
    )
    idx = cmd.index("--agent.model.per_instance_call_limit")
    assert cmd[idx + 1] == "123"
    assert isinstance(cmd[idx + 1], str)


def test_build_sweagent_cmd_extra_args_appended_last(runner_mod):
    extra = ["--some.flag", "value", "--another"]
    cmd = runner_mod.build_sweagent_cmd(
        model_name="m", api_base="http://x/v1", api_key="k", ws=Path("/ws"),
        prompt="p", instance_id="i", traj_root=Path("/t"), deployment="local",
        max_steps=10, extra_args=extra,
    )
    assert cmd[-len(extra):] == extra


def test_build_sweagent_cmd_deployment_type_passthrough(runner_mod):
    for deployment in ("local", "docker"):
        cmd = runner_mod.build_sweagent_cmd(
            model_name="m", api_base="http://x/v1", api_key="k", ws=Path("/ws"),
            prompt="p", instance_id="i", traj_root=Path("/t"),
            deployment=deployment, max_steps=10, extra_args=[],
        )
        idx = cmd.index("--env.deployment.type")
        assert cmd[idx + 1] == deployment


# ===========================================================================
# 2. find_patch
# ===========================================================================

def test_find_patch_exact_path_preferred(runner_mod, tmp_path):
    traj_root = tmp_path / "trajs"
    iid_dir = traj_root / "apps-00001"
    iid_dir.mkdir(parents=True)
    exact = iid_dir / "apps-00001.patch"
    exact.write_text("exact patch")
    # A decoy that would also match the glob fallback, to prove exact wins.
    (iid_dir / "apps-00001.other.patch").write_text("decoy")

    found = runner_mod.find_patch(traj_root, "apps-00001")
    assert found == exact
    assert found.read_text() == "exact patch"


def test_find_patch_glob_fallback_when_no_exact(runner_mod, tmp_path):
    traj_root = tmp_path / "trajs"
    nested = traj_root / "apps-00002" / "some_subdir"
    nested.mkdir(parents=True)
    fallback = nested / "apps-00002.model.patch"
    fallback.write_text("fallback patch")

    found = runner_mod.find_patch(traj_root, "apps-00002")
    assert found == fallback


def test_find_patch_none_when_nothing_matches(runner_mod, tmp_path):
    traj_root = tmp_path / "trajs"
    traj_root.mkdir()
    (traj_root / "unrelated").mkdir()
    assert runner_mod.find_patch(traj_root, "apps-00003") is None


def test_find_patch_none_for_empty_traj_root(runner_mod, tmp_path):
    traj_root = tmp_path / "empty_trajs"
    traj_root.mkdir()
    assert runner_mod.find_patch(traj_root, "apps-00004") is None


# ===========================================================================
# 3. apply_patch
# ===========================================================================

def test_apply_patch_success_applies_onto_pristine_checkout(runner_mod, tmp_path):
    # Build a repo, make an edit, capture the diff as a patch, then reset
    # back to pristine and re-apply -- mirrors run_sweagent_apps.py's real
    # flow: sweagent's patch is applied onto the workspace it materialized.
    ws = tmp_path / "ws"
    base = _make_repo(ws, content="original = True\n")
    (ws / "src.py").write_text("original = False\nadded = 1\n")
    patch_text = _git(ws, "diff").stdout
    assert patch_text  # sanity: the edit produced a non-empty diff

    _git_ok(ws, "checkout", "--", "src.py")  # back to pristine
    assert (ws / "src.py").read_text() == "original = True\n"

    patch_path = tmp_path / "fix.patch"
    patch_path.write_text(patch_text)

    ok = runner_mod.apply_patch(ws, patch_path)

    assert ok is True
    assert (ws / "src.py").read_text() == "original = False\nadded = 1\n"


def test_apply_patch_failure_on_garbage_patch(runner_mod, tmp_path, capsys):
    ws = tmp_path / "ws"
    _make_repo(ws)

    patch_path = tmp_path / "garbage.patch"
    patch_path.write_text("this is not a valid unified diff\n@@garbage@@\n")

    ok = runner_mod.apply_patch(ws, patch_path)

    assert ok is False
    captured = capsys.readouterr()
    assert "git apply failed" in captured.err


# ===========================================================================
# 4. prepare_git_workspace
# ===========================================================================

def test_prepare_git_workspace_creates_files_and_one_commit(runner_mod, tmp_path):
    sample = _fake_problem(1, question="Print hello.")
    ws = tmp_path / "ws"

    runner_mod.prepare_git_workspace(sample, ws)

    assert (ws / apps.PROBLEM_FILE).exists()
    assert (ws / apps.SOLUTION_FILE).exists()
    assert (ws / ".git").is_dir()
    log = _git_ok(ws, "log", "--oneline")
    assert len(log.splitlines()) == 1
    status = _git_ok(ws, "status", "--porcelain")
    assert status == ""  # clean tree right after the base commit


def test_prepare_git_workspace_rerun_resets_to_pristine(runner_mod, tmp_path):
    """Re-running prepare_git_workspace (reset=True semantics, always on)
    must wipe agent-made modifications and leave a pristine one-commit
    repo again -- mirrors apps.prepare_workspace's reset=True contract."""
    sample = _fake_problem(2, question="Q2")
    ws = tmp_path / "ws"
    runner_mod.prepare_git_workspace(sample, ws)

    # Simulate agent activity in a prior (failed) attempt.
    (ws / apps.SOLUTION_FILE).write_text("# agent garbage\n")
    (ws / "extra_agent_file.py").write_text("junk")

    runner_mod.prepare_git_workspace(sample, ws)

    assert not (ws / "extra_agent_file.py").exists()
    solution = (ws / apps.SOLUTION_FILE).read_text()
    assert "agent garbage" not in solution
    status = _git_ok(ws, "status", "--porcelain")
    assert status == ""


# ===========================================================================
# 5. main --dry-run
# ===========================================================================

class _StubDynamo:
    host = "127.0.0.1"
    port = 9000


class _StubModel:
    served_name = "local-stub"


class _StubCfg:
    def __init__(self, workspace_root: str):
        self.dynamo = _StubDynamo()
        self.model = _StubModel()
        self.workspace_root = workspace_root


def _install_dataset(monkeypatch, samples: list[dict[str, Any]]):
    def _loader(hf_id, split):
        return samples

    monkeypatch.setattr(apps, "_load_dataset", _loader)


def _run_main(runner_mod, argv: list[str]) -> None:
    saved_argv = sys.argv
    try:
        sys.argv = ["run_sweagent_apps.py", *argv]
        rc = runner_mod.main(argv)
    finally:
        sys.argv = saved_argv
    assert rc == 0


def test_main_dry_run_prints_one_command_per_sample_no_dirs_created(
    runner_mod, tmp_path, monkeypatch, capsys,
):
    samples = [_fake_problem(i, question=f"Q{i}") for i in (10, 11, 12)]
    _install_dataset(monkeypatch, samples)
    monkeypatch.setattr(
        runner_mod.config_mod, "load",
        lambda path=None: _StubCfg(str(tmp_path / "ws-root")),
    )

    out_dir = tmp_path / "out"
    _run_main(runner_mod, [
        "--split", "test", "--num-samples", "3", "--seed", "0",
        "--out", str(out_dir), "--dry-run",
    ])

    captured = capsys.readouterr()
    lines = [l for l in captured.out.splitlines() if l.strip()]
    assert len(lines) == 3
    for line in lines:
        assert line.startswith("sweagent run")
        assert "--agent.model.name openai/local-stub" in line
        assert "http://127.0.0.1:9000/v1" in line

    assert not out_dir.exists()
    assert not (tmp_path / "ws-root").exists()


def test_main_dry_run_respects_model_and_api_base_overrides(
    runner_mod, tmp_path, monkeypatch, capsys,
):
    samples = [_fake_problem(20, question="Q20")]
    _install_dataset(monkeypatch, samples)
    monkeypatch.setattr(
        runner_mod.config_mod, "load",
        lambda path=None: _StubCfg(str(tmp_path / "ws-root")),
    )

    _run_main(runner_mod, [
        "--split", "test", "--num-samples", "1", "--seed", "0",
        "--out", str(tmp_path / "out"), "--dry-run",
        "--model", "custom-model", "--api-base", "http://elsewhere:1234/v1",
    ])

    out = capsys.readouterr().out
    assert "openai/custom-model" in out
    assert "http://elsewhere:1234/v1" in out


# ===========================================================================
# 6. main real mode with sweagent subprocess mocked
# ===========================================================================

def _install_fake_subprocess_run(monkeypatch, runner_mod, sweagent_behavior):
    """Patch runner_mod.subprocess.run so that git invocations (argv[0] ==
    "git") pass through to the REAL subprocess.run (prepare_git_workspace /
    apply_patch need real git), while the `sweagent run` invocation is
    intercepted and handled by `sweagent_behavior(cmd) -> CompletedProcess`
    (or raises, e.g. TimeoutExpired)."""
    real_run = subprocess.run

    def _fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "git":
            return real_run(cmd, *args, **kwargs)
        if cmd and cmd[0] == "sweagent":
            return sweagent_behavior(cmd)
        raise AssertionError(f"unexpected subprocess.run call: {cmd!r}")

    monkeypatch.setattr(runner_mod.subprocess, "run", _fake_run)


def _base_main_args(tmp_path: Path, num_samples: int) -> list[str]:
    return [
        "--split", "test", "--num-samples", str(num_samples), "--seed", "0",
        "--out", str(tmp_path / "out"),
    ]


def test_main_real_mode_success_writes_patch_applied_true(
    runner_mod, tmp_path, monkeypatch,
):
    sample = _fake_problem(30, question="Q30")
    _install_dataset(monkeypatch, [sample])
    ws_root = tmp_path / "ws-root"
    monkeypatch.setattr(runner_mod.config_mod, "load",
                        lambda path=None: _StubCfg(str(ws_root)))

    iid = sample["instance_id"]
    out_dir = tmp_path / "out"
    traj_root = out_dir / "trajs"

    def sweagent_ok(cmd):
        # Simulate sweagent writing its patch output on "success".
        iid_dir = traj_root / iid
        iid_dir.mkdir(parents=True, exist_ok=True)
        ws = ws_root / "sweagent" / f"session-{iid}"
        # The workspace at this point is a pristine 1-commit repo containing
        # PROBLEM.md + solution.py (written by prepare_git_workspace).
        (ws / apps.SOLUTION_FILE).write_text("print(input())\n")
        patch_text = subprocess.run(
            ["git", "-C", str(ws), "diff"], capture_output=True, text=True,
            env=_git_env(),
        ).stdout
        subprocess.run(["git", "-C", str(ws), "checkout", "--", apps.SOLUTION_FILE],
                       capture_output=True, text=True, env=_git_env())
        (iid_dir / f"{iid}.patch").write_text(patch_text)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    _install_fake_subprocess_run(monkeypatch, runner_mod, sweagent_ok)

    _run_main(runner_mod, _base_main_args(tmp_path, 1))

    out_dir = tmp_path / "out"
    config = json.loads((out_dir / "config.json").read_text())
    assert config["workload"] == "apps"
    assert config["agent"] == "swe-agent"
    assert config["split"] == "test"
    assert config["seed"] == 0
    assert config["num_samples"] == 1
    assert config["config"]["workspace_root"] == str(ws_root / "sweagent")

    records = [json.loads(l) for l in (out_dir / "trace.jsonl").read_text().splitlines() if l.strip()]
    assert len(records) == 1
    rec = records[0]
    assert rec["instance_id"] == iid
    assert rec["success"] is True
    assert rec["error"] is None
    assert rec["patch_applied"] is True
    assert rec["rtt_s"] is not None
    assert "task_start_unix_s" in rec
    assert "task_end_unix_s" in rec

    ws = ws_root / "sweagent" / f"session-{iid}"
    assert ws.exists()


def test_main_real_mode_nonzero_exit_is_message_stage(
    runner_mod, tmp_path, monkeypatch,
):
    sample = _fake_problem(31, question="Q31")
    _install_dataset(monkeypatch, [sample])
    ws_root = tmp_path / "ws-root"
    monkeypatch.setattr(runner_mod.config_mod, "load",
                        lambda path=None: _StubCfg(str(ws_root)))

    def sweagent_fail(cmd):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom: agent crashed")

    _install_fake_subprocess_run(monkeypatch, runner_mod, sweagent_fail)

    _run_main(runner_mod, _base_main_args(tmp_path, 1))

    out_dir = tmp_path / "out"
    records = [json.loads(l) for l in (out_dir / "trace.jsonl").read_text().splitlines() if l.strip()]
    rec = records[0]
    assert rec["success"] is False
    assert rec["error"]["stage"] == "message"
    assert "boom" in rec["error"]["msg"]
    assert rec["patch_applied"] is False
    assert rec["rtt_s"] is not None  # rtt measured even on failure


def test_main_real_mode_timeout_is_timeout_stage(
    runner_mod, tmp_path, monkeypatch,
):
    sample = _fake_problem(32, question="Q32")
    _install_dataset(monkeypatch, [sample])
    ws_root = tmp_path / "ws-root"
    monkeypatch.setattr(runner_mod.config_mod, "load",
                        lambda path=None: _StubCfg(str(ws_root)))

    def sweagent_timeout(cmd):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1800.0)

    _install_fake_subprocess_run(monkeypatch, runner_mod, sweagent_timeout)

    _run_main(runner_mod, [*_base_main_args(tmp_path, 1), "--task-timeout-s", "1800"])

    out_dir = tmp_path / "out"
    records = [json.loads(l) for l in (out_dir / "trace.jsonl").read_text().splitlines() if l.strip()]
    rec = records[0]
    assert rec["success"] is False
    assert rec["error"]["stage"] == "timeout"
    assert rec["rtt_s"] is not None
    assert rec["patch_applied"] is False


def test_main_real_mode_clone_stage_on_prepare_failure(
    runner_mod, tmp_path, monkeypatch,
):
    """A prepare_git_workspace failure must land as error.stage='clone' and
    never reach the sweagent subprocess at all."""
    sample = _fake_problem(33, question="Q33")
    _install_dataset(monkeypatch, [sample])
    ws_root = tmp_path / "ws-root"
    monkeypatch.setattr(runner_mod.config_mod, "load",
                        lambda path=None: _StubCfg(str(ws_root)))

    def _boom(sample, ws):
        raise OSError("disk full")

    monkeypatch.setattr(runner_mod, "prepare_git_workspace", _boom)

    called = {"n": 0}
    real_run = subprocess.run

    def _fake_run(cmd, *a, **kw):
        called["n"] += 1
        return real_run(cmd, *a, **kw)

    monkeypatch.setattr(runner_mod.subprocess, "run", _fake_run)

    _run_main(runner_mod, _base_main_args(tmp_path, 1))

    out_dir = tmp_path / "out"
    records = [json.loads(l) for l in (out_dir / "trace.jsonl").read_text().splitlines() if l.strip()]
    rec = records[0]
    assert rec["success"] is False
    assert rec["error"]["stage"] == "clone"
    assert rec["rtt_s"] is None
    assert called["n"] == 0  # no subprocess (git or sweagent) ever invoked


# ===========================================================================
# 7. analyze_sweagent_traj
# ===========================================================================

def test_traj_tool_seconds_sums_execution_time_steps(analyze_mod, tmp_path):
    traj_dir = tmp_path / "apps-00001"
    traj_dir.mkdir()
    data = {"trajectory": [
        {"execution_time": 1.5, "other": "x"},
        {"execution_time": 2.25},
        {"no_time_field": True},
    ]}
    (traj_dir / "apps-00001.traj").write_text(json.dumps(data))

    total = analyze_mod.traj_tool_seconds(traj_dir, "apps-00001")

    assert total == pytest.approx(3.75)


def test_traj_tool_seconds_nan_when_no_time_fields_present(analyze_mod, tmp_path):
    traj_dir = tmp_path / "apps-00002"
    traj_dir.mkdir()
    data = {"trajectory": [{"foo": "bar"}, {"baz": 1}]}
    (traj_dir / "apps-00002.traj").write_text(json.dumps(data))

    total = analyze_mod.traj_tool_seconds(traj_dir, "apps-00002")

    assert math.isnan(total)


def test_traj_tool_seconds_nan_when_file_missing(analyze_mod, tmp_path):
    traj_dir = tmp_path / "apps-00003"
    traj_dir.mkdir()
    # No .traj file written at all.

    total = analyze_mod.traj_tool_seconds(traj_dir, "apps-00003")

    assert math.isnan(total)


def test_traj_tool_seconds_probes_alternate_key_names(analyze_mod, tmp_path):
    """Different sweagent versions use different step-time field names;
    the second candidate in _STEP_TIME_KEYS must also be picked up."""
    traj_dir = tmp_path / "apps-00004"
    traj_dir.mkdir()
    data = {"trajectory": [{"env_time": 4.0}]}
    (traj_dir / "apps-00004.traj").write_text(json.dumps(data))

    total = analyze_mod.traj_tool_seconds(traj_dir, "apps-00004")

    assert total == pytest.approx(4.0)


def test_traj_tool_seconds_determinism(analyze_mod, tmp_path):
    traj_dir = tmp_path / "apps-00005"
    traj_dir.mkdir()
    data = {"trajectory": [{"execution_time": 1.0}, {"execution_time": 2.0}]}
    (traj_dir / "apps-00005.traj").write_text(json.dumps(data))

    first = analyze_mod.traj_tool_seconds(traj_dir, "apps-00005")
    second = analyze_mod.traj_tool_seconds(traj_dir, "apps-00005")

    assert first == second == pytest.approx(3.0)


def test_load_frontend_completions_parses_ansi_colored_lines(analyze_mod, tmp_path):
    log = tmp_path / "frontend.log"
    # \x1b[32m...\x1b[0m simulates ANSI green coloring around the message.
    log.write_text(
        "\x1b[32m2024-01-01T00:00:10.000Z\x1b[0m INFO request completed "
        "elapsed_ms=1500\n"
        "2024-01-01T00:00:20Z INFO request completed elapsed_ms=250.5\n"
    )

    completions = analyze_mod.load_frontend_completions(log)

    assert len(completions) == 2
    ts0, el0 = completions[0]
    assert el0 == pytest.approx(1.5)
    ts1, el1 = completions[1]
    assert el1 == pytest.approx(0.2505)
    assert ts1 > ts0


def test_load_frontend_completions_ignores_non_matching_lines(analyze_mod, tmp_path):
    log = tmp_path / "frontend.log"
    log.write_text(
        "2024-01-01T00:00:10Z INFO some unrelated line\n"
        "not even a timestamp at the start\n"
        "2024-01-01T00:00:11Z INFO request completed elapsed_ms=100\n"
    )

    completions = analyze_mod.load_frontend_completions(log)

    assert len(completions) == 1
    assert completions[0][1] == pytest.approx(0.1)


def test_llm_seconds_in_window_boundaries_inclusive(analyze_mod):
    completions = [(10.0, 1.0), (20.0, 2.0), (30.0, 3.0)]

    # Inclusive on both ends per the docstring ("start <= ts <= end").
    assert analyze_mod.llm_seconds_in_window(completions, 10.0, 20.0) == pytest.approx(3.0)
    assert analyze_mod.llm_seconds_in_window(completions, 10.0, 10.0) == pytest.approx(1.0)
    assert analyze_mod.llm_seconds_in_window(completions, 0.0, 9.999) == pytest.approx(0.0)
    assert analyze_mod.llm_seconds_in_window(completions, 0.0, 100.0) == pytest.approx(6.0)


def test_llm_seconds_in_window_determinism(analyze_mod):
    completions = [(5.0, 0.5), (15.0, 1.5)]
    first = analyze_mod.llm_seconds_in_window(completions, 0.0, 20.0)
    second = analyze_mod.llm_seconds_in_window(completions, 0.0, 20.0)
    assert first == second == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# analyze_sweagent_traj main() end-to-end
# ---------------------------------------------------------------------------

def _write_synthetic_run(tmp_path: Path, *, with_frontend: bool) -> tuple[Path, Path | None]:
    run_dir = tmp_path / "run"
    trajs = run_dir / "trajs"
    trajs.mkdir(parents=True)

    # Task A: fully decomposable (has a .traj with execution_time AND falls
    # inside the frontend completion window).
    iid_a = "apps-00001"
    dir_a = trajs / iid_a
    dir_a.mkdir()
    (dir_a / f"{iid_a}.traj").write_text(json.dumps(
        {"trajectory": [{"execution_time": 2.0}, {"execution_time": 1.0}]}
    ))

    # Task B: no .traj at all -> tool_s is NaN -> others_s stays NaN too.
    iid_b = "apps-00002"
    dir_b = trajs / iid_b
    dir_b.mkdir()

    records = [
        {
            "instance_id": iid_a, "success": True, "rtt_s": 10.0,
            "traj_dir": str(dir_a),
            "task_start_unix_s": 1000.0, "task_end_unix_s": 1010.0,
        },
        {
            "instance_id": iid_b, "success": True, "rtt_s": 5.0,
            "traj_dir": str(dir_b),
            "task_start_unix_s": 2000.0, "task_end_unix_s": 2005.0,
        },
    ]
    (run_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    frontend_path = None
    if with_frontend:
        frontend_path = tmp_path / "frontend.log"
        # One completion inside task A's window (1000-1010): elapsed 4.0s.
        frontend_path.write_text(
            "2024-01-01T00:00:00Z INFO request completed elapsed_ms=4000\n"
        )
        # Rewrite with a timestamp that actually falls in [1000, 1010] unix
        # time is awkward with ISO dates; instead directly exercise via a
        # controlled epoch. 1970-01-01 + 1000s = 1970-01-01T00:16:40Z.
        frontend_path.write_text(
            "1970-01-01T00:16:40Z INFO request completed elapsed_ms=4000\n"
        )

    return run_dir, frontend_path


def test_analyze_main_end_to_end_writes_csv_and_pooled_share(
    analyze_mod, tmp_path, capsys,
):
    run_dir, frontend_path = _write_synthetic_run(tmp_path, with_frontend=True)

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_sweagent_traj", "--run", str(run_dir),
                    "--frontend", str(frontend_path)]
        rc = analyze_mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    csv_path = run_dir / "sweagent_decomposition.csv"
    assert csv_path.exists()

    import csv as csv_mod
    with csv_path.open() as f:
        rows = list(csv_mod.DictReader(f))
    assert len(rows) == 2
    row_a = next(r for r in rows if r["instance_id"] == "apps-00001")
    assert float(row_a["total_s"]) == pytest.approx(10.0)
    assert float(row_a["tool_s"]) == pytest.approx(3.0)
    assert float(row_a["llm_s"]) == pytest.approx(4.0)
    assert float(row_a["others_s"]) == pytest.approx(3.0)  # 10 - 3 - 4

    row_b = next(r for r in rows if r["instance_id"] == "apps-00002")
    assert row_b["tool_s"] == "nan"

    out = capsys.readouterr().out
    assert "wrote " in out
    assert "fully-decomposed=1" in out
    assert "pooled share over 1 tasks" in out


def test_analyze_main_tool_only_fallback_without_frontend(
    analyze_mod, tmp_path, capsys,
):
    """Without --frontend, llm_s stays NaN for every row, so the pooled
    'fully-decomposed' branch is empty and the tool-only fallback prints
    instead."""
    run_dir, _ = _write_synthetic_run(tmp_path, with_frontend=False)

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_sweagent_traj", "--run", str(run_dir)]
        rc = analyze_mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    out = capsys.readouterr().out
    assert "fully-decomposed=0" in out
    assert "tool-only pooled share" in out
    # Only task A has a known tool_s (3.0s out of 10.0s total).
    assert "30.00%" in out
