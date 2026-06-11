"""Tests for _pre_clone in runner.py.

Uses REAL local git repos (created in tmp_path, served via file:// by
monkeypatching runner._repo_url) -- no network. Skipped if git isn't on
PATH. This is the cleanest way to exercise clone/checkout/reset logic
without fragile subprocess mocking.
"""

from __future__ import annotations

import asyncio
import shutil
import subprocess
from pathlib import Path

import pytest

from testbed import runner


pytestmark = pytest.mark.skipif(
    shutil.which("git") is None, reason="git not on PATH"
)


def _git(cwd: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
        "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null",
        "PATH": __import__("os").environ.get("PATH", ""),
    }
    out = subprocess.run(["git", "-C", str(cwd), *args], check=True,
                         capture_output=True, text=True, env=env)
    return out.stdout.strip()


def _make_source_repo(path: Path) -> tuple[str, str]:
    """Create a git repo with two commits. Returns (base_commit, head)."""
    path.mkdir(parents=True)
    # No `-b main` -- that needs git >= 2.28; we reference commits by SHA,
    # so the default branch name is irrelevant.
    _git(path, "init", "-q")
    (path / "a.txt").write_text("one\n")
    _git(path, "add", "a.txt")
    _git(path, "commit", "-q", "-m", "first")
    base = _git(path, "rev-parse", "HEAD")
    (path / "b.txt").write_text("two\n")
    _git(path, "add", "b.txt")
    _git(path, "commit", "-q", "-m", "second")
    head = _git(path, "rev-parse", "HEAD")
    return base, head


@pytest.fixture
def source_repo(tmp_path, monkeypatch):
    """A local 'remote' repo + runner._repo_url pointed at it via file://."""
    src = tmp_path / "src" / "owner__proj"
    base, head = _make_source_repo(src)
    monkeypatch.setattr(runner, "_repo_url", lambda repo: f"file://{src}")
    return {"repo": "owner/proj", "base": base, "head": head, "src": src}


# ---------- clone + checkout ----------


def test_pre_clone_clones_and_checks_out_base_commit(source_repo, tmp_path):
    """Direct clone via monkeypatched file:// URL lands at base_commit."""
    dest = tmp_path / "ws" / "session-i"
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest))
    assert (dest / ".git").is_dir()
    assert (dest / "a.txt").exists()         # base commit content
    assert not (dest / "b.txt").exists()     # second commit NOT checked out
    assert _git(dest, "rev-parse", "HEAD") == source_repo["base"]


def test_pre_clone_can_checkout_head_commit(source_repo, tmp_path):
    """Cloning to head commit works the same way."""
    dest = tmp_path / "ws" / "session-head"
    asyncio.run(runner._pre_clone("owner/proj", source_repo["head"], dest))
    assert (dest / "a.txt").exists()
    assert (dest / "b.txt").exists()
    assert _git(dest, "rev-parse", "HEAD") == source_repo["head"]


# ---------- existing dest no-op (reset=False) ----------


def test_pre_clone_existing_dest_is_noop_when_reset_false(source_repo, tmp_path):
    """With reset=False (default), an existing dest is returned immediately
    without touching it -- even if it's at a different commit."""
    dest = tmp_path / "ws" / "session-i"
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest))
    # Write a sentinel file to prove the dest is not re-cloned.
    (dest / "sentinel.txt").write_text("untouched\n")
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest))
    assert (dest / "sentinel.txt").read_text() == "untouched\n"


# ---------- reset=True rewinds tracked edits + removes untracked ----------


def test_pre_clone_reset_true_rewinds_tracked_edit(source_repo, tmp_path):
    """reset=True on an existing checkout rewinds a tracked file edit."""
    dest = tmp_path / "ws" / "session-i"
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest))
    # Agent edits a tracked file.
    (dest / "a.txt").write_text("agent modified\n")
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  reset=True))
    assert (dest / "a.txt").read_text() == "one\n"   # rewound


def test_pre_clone_reset_true_removes_untracked_files(source_repo, tmp_path):
    """reset=True removes untracked files left by the agent."""
    dest = tmp_path / "ws" / "session-i"
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest))
    (dest / "untracked.py").write_text("junk\n")
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  reset=True))
    assert not (dest / "untracked.py").exists()


def test_pre_clone_reset_true_preserves_content_at_commit(source_repo, tmp_path):
    """After reset, HEAD is still exactly base_commit."""
    dest = tmp_path / "ws" / "session-i"
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest))
    (dest / "a.txt").write_text("dirtied\n")
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  reset=True))
    assert _git(dest, "rev-parse", "HEAD") == source_repo["base"]


# ---------- interrupted dest (not a git repo) is recreated ----------


def test_pre_clone_non_git_dest_is_recreated(source_repo, tmp_path):
    """An existing dest that is NOT a valid git repo (e.g. an interrupted
    prior clone) is removed and re-cloned from scratch."""
    dest = tmp_path / "ws" / "session-i"
    # Simulate a partially interrupted clone: directory exists but no .git
    dest.mkdir(parents=True)
    (dest / "partial_file.txt").write_text("partial clone artifact\n")

    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  reset=True))
    assert (dest / ".git").is_dir()
    assert (dest / "a.txt").exists()
    assert not (dest / "partial_file.txt").exists()


# ---------- half-written .git dir (broken checkout in BOTH modes) ----------


def test_pre_clone_half_git_reset_false_reclones(source_repo, tmp_path):
    """A dest that has a .git subdirectory but is NOT a valid repo (e.g. an
    interrupted git clone that wrote .git/ but nothing else) must be re-cloned
    with reset=False.

    This reproduces the user crash: previously reset=False returned early on
    any existing dest, silently reusing a broken .git directory and causing
    every subsequent git command to die with "fatal: not a git repository"."""
    dest = tmp_path / "ws" / "session-broken"
    # Simulate an interrupted clone: directory + empty .git subdir, no objects.
    dest.mkdir(parents=True)
    (dest / ".git").mkdir()

    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  reset=False))

    # The broken dir was nuked and replaced with a real checkout.
    assert (dest / ".git").is_dir()
    assert (dest / "a.txt").exists()
    assert not (dest / "b.txt").exists()
    assert _git(dest, "rev-parse", "HEAD") == source_repo["base"]


def test_pre_clone_half_git_reset_true_reclones(source_repo, tmp_path):
    """Same broken-.git scenario with reset=True also re-clones (not just
    skips or crashes on the git reset --hard on a broken repo)."""
    dest = tmp_path / "ws" / "session-broken-reset"
    dest.mkdir(parents=True)
    (dest / ".git").mkdir()

    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  reset=True))

    assert (dest / ".git").is_dir()
    assert (dest / "a.txt").exists()
    assert _git(dest, "rev-parse", "HEAD") == source_repo["base"]


# ---------- valid repo + reset=True where reset fails → re-clone ----------


def test_pre_clone_reset_fails_falls_through_to_reclone(source_repo, tmp_path,
                                                         monkeypatch):
    """If git reset --hard raises RuntimeError on a valid-looking repo (e.g.
    missing pack objects), _pre_clone must fall through to rmtree + fresh
    clone rather than propagating the error or returning the dirty checkout.

    We monkeypatch runner._run_git so that reset --hard raises but all other
    git subcommands (clone, checkout, rev-parse) are forwarded to the REAL
    implementation."""
    dest = tmp_path / "ws" / "session-reset-fails"

    # First do a real clone so dest is a valid repo.
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest))
    assert _git(dest, "rev-parse", "HEAD") == source_repo["base"]

    # Dirty the workspace so we can confirm it gets cleaned up.
    (dest / "agent_artifact.py").write_text("print('leftover')\n")
    (dest / "a.txt").write_text("agent modified\n")

    # Intercept _run_git: raise on reset --hard; pass everything else through.
    real_run_git = runner._run_git

    async def _selective_run_git(args):
        if "reset" in args and "--hard" in args:
            raise RuntimeError("simulated: missing pack objects")
        return await real_run_git(args)

    monkeypatch.setattr(runner, "_run_git", _selective_run_git)

    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  reset=True))

    # After the fallthrough re-clone, dest must be a clean checkout.
    assert (dest / ".git").is_dir()
    assert (dest / "a.txt").read_text() == "one\n"           # restored
    assert not (dest / "agent_artifact.py").exists()          # untracked removed
    assert _git(dest, "rev-parse", "HEAD") == source_repo["base"]


# ---------- _is_valid_repo direct tests ----------


def test_is_valid_repo_true_for_real_clone(source_repo, tmp_path):
    """_is_valid_repo returns True for a properly cloned repository."""
    dest = tmp_path / "ws" / "session-valid"
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest))
    assert asyncio.run(runner._is_valid_repo(dest)) is True


def test_is_valid_repo_false_for_empty_git_subdir(tmp_path):
    """_is_valid_repo returns False for a directory that has a .git subdir
    but is not a real git repo (half-written clone)."""
    dest = tmp_path / "broken"
    dest.mkdir()
    (dest / ".git").mkdir()
    assert asyncio.run(runner._is_valid_repo(dest)) is False


def test_is_valid_repo_false_for_nonexistent_path(tmp_path):
    """_is_valid_repo returns False for a path that does not exist at all."""
    dest = tmp_path / "does_not_exist"
    assert asyncio.run(runner._is_valid_repo(dest)) is False


# ---------- unreachable URL raises ----------


def test_pre_clone_unreachable_repo_raises(tmp_path, monkeypatch):
    """_pre_clone against a nonexistent file:// URL raises after retries."""
    monkeypatch.setattr(runner, "_repo_url",
                        lambda repo: f"file://{tmp_path}/does-not-exist")

    # _run_git_retry sleeps between attempts; no-op it so the test doesn't
    # wait through the exponential backoff.
    async def _no_sleep(*a, **k):
        return None

    monkeypatch.setattr(runner.asyncio, "sleep", _no_sleep)

    dest = tmp_path / "ws" / "session-fail"
    with pytest.raises((RuntimeError, Exception)):
        asyncio.run(runner._pre_clone("x/y", "deadbeef", dest))
