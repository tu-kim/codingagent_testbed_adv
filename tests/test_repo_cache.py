"""Tests for the repo-cache pre-clone path in runner.py.

Uses REAL local git repos (created in tmp_path, served via file:// by
monkeypatching runner._repo_url) -- no network. Skipped if git isn't on
PATH. This is the cleanest way to exercise clone/fetch/checkout logic
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


# ---------- warm_repo_cache ----------


def test_warm_repo_cache_clones_unique_repos_once(source_repo, tmp_path):
    cache_dir = tmp_path / "cache"
    samples = [
        {"repo": "owner/proj", "base_commit": source_repo["base"], "instance_id": "i1"},
        {"repo": "owner/proj", "base_commit": source_repo["head"], "instance_id": "i2"},
    ]
    cached = asyncio.run(runner.warm_repo_cache(samples, cache_dir))
    assert set(cached) == {"owner/proj"}
    cache = runner._repo_cache_path(cache_dir, "owner/proj")
    assert (cache / ".git").is_dir()
    # Both base commits are present in the single cached clone.
    for c in (source_repo["base"], source_repo["head"]):
        _git(cache, "cat-file", "-e", f"{c}^{{commit}}")


def test_warm_repo_cache_idempotent(source_repo, tmp_path):
    cache_dir = tmp_path / "cache"
    samples = [{"repo": "owner/proj", "base_commit": source_repo["base"], "instance_id": "i"}]
    asyncio.run(runner.warm_repo_cache(samples, cache_dir))
    # Second warm must not fail or re-clone over the existing cache.
    cached = asyncio.run(runner.warm_repo_cache(samples, cache_dir))
    assert "owner/proj" in cached


def test_warm_repo_cache_unreachable_repo_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "_repo_url",
                        lambda repo: f"file://{tmp_path}/does-not-exist")
    samples = [{"repo": "x/y", "base_commit": "deadbeef", "instance_id": "i"}]
    # _run_git_retry sleeps (exponential backoff) between attempts; no-op it
    # so the test doesn't actually wait through the retries.
    async def _no_sleep(*a, **k):
        return None
    monkeypatch.setattr(runner.asyncio, "sleep", _no_sleep)
    cached = asyncio.run(runner.warm_repo_cache(samples, tmp_path / "cache"))
    assert cached == {}     # failed repo absent → caller falls back


# ---------- _pre_clone using the cache ----------


def test_pre_clone_from_cache_no_network(source_repo, tmp_path, monkeypatch):
    """With a warmed cache, _pre_clone copies locally and checks out the
    base_commit. We assert it never calls the network URL by making
    _repo_url raise if used."""
    cache_dir = tmp_path / "cache"
    samples = [{"repo": "owner/proj", "base_commit": source_repo["base"], "instance_id": "i"}]
    asyncio.run(runner.warm_repo_cache(samples, cache_dir))

    # After warming, any further _repo_url use would mean a network clone.
    monkeypatch.setattr(runner, "_repo_url",
                        lambda repo: (_ for _ in ()).throw(AssertionError("network clone!")))
    dest = tmp_path / "ws" / "session-i"
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  cache_dir=cache_dir))
    assert (dest / ".git").is_dir()
    assert (dest / "a.txt").exists()        # base commit content
    assert not (dest / "b.txt").exists()    # second commit NOT checked out
    assert _git(dest, "rev-parse", "HEAD") == source_repo["base"]


def test_pre_clone_falls_back_to_network_when_no_cache(source_repo, tmp_path):
    """No cache dir → direct clone from _repo_url (here the local file://
    source). Confirms the fallback path still works."""
    dest = tmp_path / "ws" / "session-i"
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  cache_dir=None))
    assert (dest / "a.txt").exists()
    assert _git(dest, "rev-parse", "HEAD") == source_repo["base"]


def test_pre_clone_reset_mode_rewinds_existing(source_repo, tmp_path):
    """reset=True on an existing checkout rewinds to base_commit and wipes
    untracked files -- without re-cloning."""
    cache_dir = tmp_path / "cache"
    samples = [{"repo": "owner/proj", "base_commit": source_repo["base"], "instance_id": "i"}]
    asyncio.run(runner.warm_repo_cache(samples, cache_dir))
    dest = tmp_path / "ws" / "session-i"
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  cache_dir=cache_dir))
    # Dirty the workspace like an agent would.
    (dest / "a.txt").write_text("agent edited\n")
    (dest / "untracked.py").write_text("junk\n")
    # Reset path.
    asyncio.run(runner._pre_clone("owner/proj", source_repo["base"], dest,
                                  reset=True, cache_dir=cache_dir))
    assert (dest / "a.txt").read_text() == "one\n"   # rewound
    assert not (dest / "untracked.py").exists()      # cleaned
