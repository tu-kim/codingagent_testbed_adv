"""Tests for scripts/extract_predictions.py.

Uses REAL local git repos in tmp_path (env-scrubbed _git helper, skipif git
not on PATH) to exercise extract_patch. No network, no GPU.

load_run / base_commits_from_dataset (dataset import) and main() are tested
via --base-commits-json to avoid touching testbed.swebench.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_predictions.py"

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not on PATH"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("extract_predictions", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["extract_predictions"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Minimal git helper (same env-scrub convention as test_pre_clone_git.py)
# ---------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
        "PATH": os.environ.get("PATH", ""),
    }
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True, text=True, env=env,
    )


def _git_ok(cwd: Path, *args: str) -> str:
    r = _git(cwd, *args)
    if r.returncode != 0:
        raise RuntimeError(f"git {args!r} failed:\n{r.stderr}")
    return r.stdout.strip()


def _make_repo(path: Path) -> str:
    """Create a repo with a single commit. Returns the base_commit SHA."""
    path.mkdir(parents=True, exist_ok=True)
    _git_ok(path, "init", "-q")
    (path / "src.py").write_text("x = 1\n")
    _git_ok(path, "add", "src.py")
    _git_ok(path, "commit", "-q", "-m", "base")
    return _git_ok(path, "rev-parse", "HEAD")


# ---------------------------------------------------------------------------
# extract_patch — happy path and error branches
# ---------------------------------------------------------------------------

def test_extract_patch_captures_edit(mod, tmp_path):
    """Editing a tracked file produces a unified diff containing that change."""
    ws = tmp_path / "ws"
    base = _make_repo(ws)
    (ws / "src.py").write_text("x = 2\n")

    patch, err = mod.extract_patch(ws, base, mod.DEFAULT_EXCLUDES)

    assert err is None
    assert "src.py" in patch
    assert "-x = 1" in patch
    assert "+x = 2" in patch


def test_extract_patch_captures_new_file(mod, tmp_path):
    """An untracked new file is staged by git add -A and appears in the diff."""
    ws = tmp_path / "ws"
    base = _make_repo(ws)
    (ws / "fix.py").write_text("def answer(): return 42\n")

    patch, err = mod.extract_patch(ws, base, mod.DEFAULT_EXCLUDES)

    assert err is None
    assert "fix.py" in patch
    assert "+def answer" in patch


def test_extract_patch_excludes_pycache(mod, tmp_path):
    """__pycache__/*.pyc files are excluded from the diff via pathspec."""
    ws = tmp_path / "ws"
    base = _make_repo(ws)
    pycache = ws / "__pycache__"
    pycache.mkdir()
    (pycache / "src.cpython-311.pyc").write_bytes(b"\x00junk")
    (ws / "src.py").write_text("x = 99\n")

    patch, err = mod.extract_patch(ws, base, mod.DEFAULT_EXCLUDES)

    assert err is None
    assert "__pycache__" not in patch
    assert ".pyc" not in patch
    assert "src.py" in patch


def test_extract_patch_empty_for_untouched_workspace(mod, tmp_path):
    """A workspace with no changes relative to base_commit gives an empty patch."""
    ws = tmp_path / "ws"
    base = _make_repo(ws)

    patch, err = mod.extract_patch(ws, base, mod.DEFAULT_EXCLUDES)

    assert err is None
    assert patch == ""


def test_extract_patch_error_for_non_git_dir(mod, tmp_path):
    """A directory without a .git folder returns an error string, not a crash."""
    non_git = tmp_path / "not_a_repo"
    non_git.mkdir()
    (non_git / "file.py").write_text("pass\n")

    patch, err = mod.extract_patch(non_git, "deadbeef1234", mod.DEFAULT_EXCLUDES)

    assert patch == ""
    assert err is not None
    assert "not a git repo" in err


def test_extract_patch_head_as_base_diffs_against_head(mod, tmp_path):
    """Passing base_commit=None triggers the --head-as-base path:
    rev-parse HEAD is used, so no changes → empty patch."""
    ws = tmp_path / "ws"
    _make_repo(ws)
    # No modifications after the initial commit.
    patch, err = mod.extract_patch(ws, None, mod.DEFAULT_EXCLUDES)
    assert err is None
    assert patch == ""


def test_extract_patch_head_as_base_with_edit(mod, tmp_path):
    """base_commit=None + a modified file → diff against HEAD (which IS base_commit
    since the agent never committed), so the edit is visible."""
    ws = tmp_path / "ws"
    _make_repo(ws)
    (ws / "src.py").write_text("x = 77\n")

    patch, err = mod.extract_patch(ws, None, mod.DEFAULT_EXCLUDES)

    assert err is None
    assert "src.py" in patch
    assert "+x = 77" in patch


def test_extract_patch_custom_exclude(mod, tmp_path):
    """A caller-supplied extra exclude pattern filters that file out."""
    ws = tmp_path / "ws"
    base = _make_repo(ws)
    (ws / "notes.txt").write_text("scratch\n")
    (ws / "fix.py").write_text("fixed = True\n")

    patch, err = mod.extract_patch(ws, base, mod.DEFAULT_EXCLUDES + ["*.txt"])

    assert err is None
    assert "fix.py" in patch
    assert "notes.txt" not in patch


# ---------------------------------------------------------------------------
# load_run
# ---------------------------------------------------------------------------

def test_load_run_parses_config_and_records(mod, tmp_path):
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    cfg = {"split": "lite", "seed": 42, "num_samples": 3,
           "config": {"workspace_root": "/tmp/ws"}}
    (run_dir / "config.json").write_text(json.dumps(cfg))
    records = [
        {"instance_id": "a__a-1", "directory": "session-a__a-1", "success": True},
        {"instance_id": "b__b-2", "directory": "session-b__b-2", "success": False},
    ]
    (run_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    got_cfg, got_records = mod.load_run(run_dir)

    assert got_cfg["split"] == "lite"
    assert len(got_records) == 2
    assert got_records[0]["instance_id"] == "a__a-1"
    assert got_records[1]["success"] is False


def test_load_run_skips_blank_lines_in_trace(mod, tmp_path):
    run_dir = tmp_path / "run2"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({"split": "lite", "seed": 1}))
    (run_dir / "trace.jsonl").write_text(
        json.dumps({"instance_id": "x"}) + "\n\n\n"
        + json.dumps({"instance_id": "y"}) + "\n"
    )

    _, records = mod.load_run(run_dir)
    assert len(records) == 2


# ---------------------------------------------------------------------------
# main() — argparse integration via sys.argv patching
# ---------------------------------------------------------------------------

def _make_run_dir(tmp_path: Path, ws_root: Path,
                  records: list[dict],
                  base_commits: dict[str, str] | None = None,
                  config_extra: dict | None = None) -> tuple[Path, Path]:
    """Build a minimal run dir and workspace tree for main() tests.

    Returns (run_dir, base_commits_json_path). base_commits_json_path is None
    when base_commits is None (test should use --head-as-base instead).
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = {"split": "lite", "seed": 42, "num_samples": len(records),
           "config": {"workspace_root": str(ws_root)}}
    if config_extra:
        cfg.update(config_extra)
    (run_dir / "config.json").write_text(json.dumps(cfg))
    (run_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )

    bc_path = None
    if base_commits is not None:
        bc_path = tmp_path / "base_commits.json"
        bc_path.write_text(json.dumps(base_commits))

    return run_dir, bc_path


def test_main_writes_predictions_jsonl(mod, tmp_path):
    """Happy path: workspace exists, has a patch, predictions.jsonl is written."""
    ws_root = tmp_path / "ws"
    iid = "django__django-1"
    ws = ws_root / f"session-{iid}"
    base = _make_repo(ws)
    (ws / "src.py").write_text("x = 999\n")

    records = [{"instance_id": iid, "directory": f"session-{iid}", "success": True}]
    run_dir, bc_path = _make_run_dir(tmp_path, ws_root, records,
                                     base_commits={iid: base})
    out = tmp_path / "preds.jsonl"

    saved_argv = sys.argv
    try:
        sys.argv = ["extract_predictions", "--run", str(run_dir),
                    "--out", str(out),
                    "--base-commits-json", str(bc_path),
                    "--model-name", "mymodel"]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0]["instance_id"] == iid
    assert lines[0]["model_name_or_path"] == "mymodel"
    assert "src.py" in lines[0]["model_patch"]


def test_main_empty_patch_for_missing_workspace(mod, tmp_path):
    """A record whose workspace directory does not exist gets an empty patch."""
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    iid = "missing__repo-1"
    records = [{"instance_id": iid, "directory": "session-missing", "success": True}]
    run_dir, bc_path = _make_run_dir(tmp_path, ws_root, records,
                                     base_commits={iid: "deadbeef1234"})
    out = tmp_path / "preds.jsonl"

    saved_argv = sys.argv
    try:
        sys.argv = ["extract_predictions", "--run", str(run_dir),
                    "--out", str(out),
                    "--base-commits-json", str(bc_path)]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert lines[0]["model_patch"] == ""


def test_main_head_as_base_with_edit(mod, tmp_path):
    """--head-as-base: diff is against HEAD; an edit in the workspace appears
    in model_patch without needing a --base-commits-json file."""
    ws_root = tmp_path / "ws"
    iid = "proj__proj-42"
    ws = ws_root / f"session-{iid}"
    _make_repo(ws)
    (ws / "src.py").write_text("patched = True\n")

    records = [{"instance_id": iid, "directory": f"session-{iid}", "success": True}]
    run_dir, _ = _make_run_dir(tmp_path, ws_root, records, base_commits=None)
    out = tmp_path / "preds.jsonl"

    saved_argv = sys.argv
    try:
        sys.argv = ["extract_predictions", "--run", str(run_dir),
                    "--out", str(out),
                    "--head-as-base"]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert "+patched = True" in lines[0]["model_patch"]


def test_main_default_out_path_in_run_dir(mod, tmp_path):
    """Without --out, predictions.jsonl lands in the run dir itself."""
    ws_root = tmp_path / "ws"
    ws_root.mkdir()
    records = [{"instance_id": "a__b-1", "directory": "session-a__b-1",
                "success": False}]
    run_dir, bc_path = _make_run_dir(tmp_path, ws_root, records,
                                     base_commits={"a__b-1": "deadbeef"})

    saved_argv = sys.argv
    try:
        sys.argv = ["extract_predictions", "--run", str(run_dir),
                    "--base-commits-json", str(bc_path)]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    assert (run_dir / "predictions.jsonl").exists()


def test_main_workspace_root_override(mod, tmp_path):
    """--workspace-root overrides the value in config.json."""
    # config will have wrong ws root; correct one passed via flag
    wrong_ws = tmp_path / "wrong_ws"
    wrong_ws.mkdir()
    real_ws = tmp_path / "real_ws"
    iid = "x__y-1"
    ws = real_ws / f"session-{iid}"
    base = _make_repo(ws)
    (ws / "src.py").write_text("override = 1\n")

    records = [{"instance_id": iid, "directory": f"session-{iid}", "success": True}]
    run_dir, bc_path = _make_run_dir(tmp_path, wrong_ws, records,
                                     base_commits={iid: base})
    out = tmp_path / "preds.jsonl"

    saved_argv = sys.argv
    try:
        sys.argv = ["extract_predictions", "--run", str(run_dir),
                    "--out", str(out),
                    "--workspace-root", str(real_ws),
                    "--base-commits-json", str(bc_path)]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert "+override = 1" in lines[0]["model_patch"]


def test_main_extra_exclude_flag(mod, tmp_path):
    """--exclude adds on top of DEFAULT_EXCLUDES; the excluded file is absent
    from the patch."""
    ws_root = tmp_path / "ws"
    iid = "excl__test-1"
    ws = ws_root / f"session-{iid}"
    base = _make_repo(ws)
    (ws / "scratch.log").write_text("agent debug output\n")
    (ws / "src.py").write_text("fixed = True\n")

    records = [{"instance_id": iid, "directory": f"session-{iid}", "success": True}]
    run_dir, bc_path = _make_run_dir(tmp_path, ws_root, records,
                                     base_commits={iid: base})
    out = tmp_path / "preds.jsonl"

    saved_argv = sys.argv
    try:
        sys.argv = ["extract_predictions", "--run", str(run_dir),
                    "--out", str(out),
                    "--base-commits-json", str(bc_path),
                    "--exclude", "*.log"]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert "scratch.log" not in lines[0]["model_patch"]
    assert "src.py" in lines[0]["model_patch"]


def test_main_multiple_records(mod, tmp_path):
    """Multiple trace records each become one predictions line."""
    ws_root = tmp_path / "ws"
    iids = ["aa__aa-1", "bb__bb-2", "cc__cc-3"]
    base_commits = {}
    for iid in iids:
        ws = ws_root / f"session-{iid}"
        base = _make_repo(ws)
        (ws / "src.py").write_text(f"# {iid}\n")
        base_commits[iid] = base

    records = [
        {"instance_id": iid, "directory": f"session-{iid}", "success": True}
        for iid in iids
    ]
    run_dir, bc_path = _make_run_dir(tmp_path, ws_root, records,
                                     base_commits=base_commits)
    out = tmp_path / "preds.jsonl"

    saved_argv = sys.argv
    try:
        sys.argv = ["extract_predictions", "--run", str(run_dir),
                    "--out", str(out),
                    "--base-commits-json", str(bc_path)]
        mod.main()
    finally:
        sys.argv = saved_argv

    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(lines) == 3
    assert {l["instance_id"] for l in lines} == set(iids)
    # All have non-empty patches (each ws has a file edit).
    assert all(l["model_patch"] for l in lines)


def test_base_commits_from_dataset_monkeypatched(mod, tmp_path, monkeypatch):
    """base_commits_from_dataset calls testbed.swebench.load_samples;
    verify it converts the return value to {instance_id: base_commit}."""
    fake_samples = [
        {"instance_id": "proj__proj-1", "base_commit": "aabbcc"},
        {"instance_id": "proj__proj-2", "base_commit": "ddeeff"},
    ]

    class _FakeSwebench:
        @staticmethod
        def load_samples(split, seed, n):
            return fake_samples

    # Ensure the import inside the function resolves to our fake.
    import types
    fake_testbed = types.ModuleType("testbed")
    fake_testbed.swebench = _FakeSwebench()
    monkeypatch.setitem(sys.modules, "testbed", fake_testbed)
    monkeypatch.setitem(sys.modules, "testbed.swebench", _FakeSwebench())

    config = {"split": "lite", "seed": 42, "num_samples": 2}
    result = mod.base_commits_from_dataset(config)

    assert result == {"proj__proj-1": "aabbcc", "proj__proj-2": "ddeeff"}
