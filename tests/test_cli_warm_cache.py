"""Tests for the `warm-cache` CLI subcommand in cli.py.

All stubs are mock-based — no network, no git, no real config file.
config_mod.load is monkeypatched to return a SimpleNamespace so there's no
need for a real testbed.yaml on disk.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner

from testbed.cli import main


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_REPO_A = "django/django"
_REPO_B = "pytest-dev/pytest"

# Two samples that share _REPO_A; one sample for _REPO_B.
_SAMPLES = [
    {"instance_id": "django__django-1", "repo": _REPO_A, "base_commit": "aaa"},
    {"instance_id": "django__django-2", "repo": _REPO_A, "base_commit": "bbb"},
    {"instance_id": "pytest-dev__pytest-1", "repo": _REPO_B, "base_commit": "ccc"},
]

# Unique repos derived from the sample list (sorted, matching cli.py logic).
_UNIQUE_REPOS = sorted({s["repo"] for s in _SAMPLES})  # [_REPO_A, _REPO_B]


def _make_cfg(workspace_root: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(workspace_root=workspace_root)


def _make_warm_stub(return_dict: dict) -> tuple[AsyncMock, list]:
    """Return (async stub, call_recorder).

    The stub records (samples, cache_dir, concurrency) on each call
    and returns return_dict.
    """
    calls: list[tuple] = []

    async def _stub(samples, cache_dir, *, concurrency=4):
        calls.append((samples, cache_dir, concurrency))
        return return_dict

    return _stub, calls


def _invoke(args: list[str], monkeypatch, tmp_path: Path,
            warm_return: dict | None = None) -> tuple:
    """
    Invoke `main warm-cache <args>` with all external calls stubbed.

    Returns (result, warm_calls) where warm_calls is the list recorded by
    the warm_repo_cache stub.
    """
    if warm_return is None:
        warm_return = {r: tmp_path / r for r in _UNIQUE_REPOS}

    warm_stub, warm_calls = _make_warm_stub(warm_return)

    monkeypatch.setattr("testbed.cli.swebench.load_samples", lambda *a, **kw: _SAMPLES)
    monkeypatch.setattr("testbed.cli.runner_mod.warm_repo_cache", warm_stub)
    monkeypatch.setattr(
        "testbed.cli.config_mod.load",
        lambda *a, **kw: _make_cfg(str(tmp_path / "ws")),
    )

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(main, ["warm-cache"] + args, catch_exceptions=False)
    return result, warm_calls


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


def test_all_repos_cached_exits_0(monkeypatch, tmp_path):
    """Happy path: warm_repo_cache returns every unique repo → exit 0."""
    result, _ = _invoke([], monkeypatch, tmp_path)

    assert result.exit_code == 0, result.output
    assert _REPO_A in result.output
    assert _REPO_B in result.output
    assert "ok" in result.output
    # Both repos cached; only two unique repos.
    assert "cached 2/2" in result.output
    # No "FAILED" anywhere.
    assert "FAILED" not in result.output


def test_one_repo_missing_exits_1_and_shows_failed(monkeypatch, tmp_path):
    """One repo absent from stub return dict → exit 1, FAILED shown, stderr mentions fallback."""
    # Return only _REPO_A; _REPO_B is missing.
    partial = {_REPO_A: tmp_path / _REPO_A}
    result, _ = _invoke([], monkeypatch, tmp_path, warm_return=partial)

    assert result.exit_code == 1
    # Output must show FAILED for the missing repo.
    assert "FAILED" in result.output
    assert _REPO_B in result.output
    # Stderr (mix_stderr=False) should mention the fallback / investigation note.
    assert result.stderr != "" or "fallback" in result.output or _REPO_B in (result.stderr or "")
    # Summary line reflects partial success.
    assert "cached 1/2" in result.output


def test_repo_cache_dir_override_forwarded(monkeypatch, tmp_path):
    """--repo-cache-dir override reaches warm_repo_cache as cache_dir."""
    override = tmp_path / "my-custom-cache"
    result, warm_calls = _invoke(
        ["--repo-cache-dir", str(override)],
        monkeypatch,
        tmp_path,
    )

    assert result.exit_code == 0, result.output
    assert len(warm_calls) == 1
    _, actual_cache_dir, _ = warm_calls[0]
    assert Path(actual_cache_dir) == override


def test_default_cache_dir_is_workspace_root_dot_repo_cache(monkeypatch, tmp_path):
    """Without --repo-cache-dir, cache_dir == <workspace_root>/.repo-cache."""
    workspace_root = tmp_path / "ws"
    expected_cache_dir = workspace_root / ".repo-cache"

    monkeypatch.setattr("testbed.cli.swebench.load_samples", lambda *a, **kw: _SAMPLES)
    monkeypatch.setattr(
        "testbed.cli.config_mod.load",
        lambda *a, **kw: _make_cfg(str(workspace_root)),
    )

    warm_return = {r: workspace_root / ".repo-cache" / r for r in _UNIQUE_REPOS}
    warm_stub, warm_calls = _make_warm_stub(warm_return)
    monkeypatch.setattr("testbed.cli.runner_mod.warm_repo_cache", warm_stub)

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(main, ["warm-cache"], catch_exceptions=False)

    assert result.exit_code == 0, result.output
    assert len(warm_calls) == 1
    _, actual_cache_dir, _ = warm_calls[0]
    assert Path(actual_cache_dir) == expected_cache_dir


def test_concurrency_forwarded_to_warm_repo_cache(monkeypatch, tmp_path):
    """--concurrency <N> reaches warm_repo_cache's concurrency kwarg."""
    result, warm_calls = _invoke(
        ["--concurrency", "8"],
        monkeypatch,
        tmp_path,
    )

    assert result.exit_code == 0, result.output
    assert len(warm_calls) == 1
    _, _, actual_concurrency = warm_calls[0]
    assert actual_concurrency == 8


def test_samples_forwarded_to_warm_repo_cache(monkeypatch, tmp_path):
    """warm_repo_cache receives the full samples list returned by load_samples."""
    result, warm_calls = _invoke([], monkeypatch, tmp_path)

    assert result.exit_code == 0, result.output
    assert len(warm_calls) == 1
    actual_samples, _, _ = warm_calls[0]
    assert actual_samples == _SAMPLES


def test_all_repos_missing_exits_1(monkeypatch, tmp_path):
    """Edge case: warm_repo_cache returns empty dict → exit 1, all repos FAILED."""
    result, _ = _invoke([], monkeypatch, tmp_path, warm_return={})

    assert result.exit_code == 1
    assert "FAILED" in result.output
    assert "cached 0/2" in result.output
