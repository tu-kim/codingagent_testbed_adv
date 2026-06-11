"""Tests for the `pre-clone` CLI subcommand in cli.py.

All stubs are mock-based — no network, no git, no real config file.
config_mod.load is monkeypatched to return a SimpleNamespace so there's no
need for a real testbed.yaml on disk.  runner_mod.pre_clone_run is stubbed
with an async coroutine that records kwargs and returns a configurable
failures dict.
"""

from __future__ import annotations

import asyncio
import types
from pathlib import Path

import pytest
from click.testing import CliRunner

from testbed.cli import main


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_cfg(workspace_root: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(workspace_root=workspace_root)


def _make_pre_clone_stub(failures: dict | None = None):
    """Return (async stub, call_recorder).

    The stub records the keyword arguments passed by cli pre_clone_cmd and
    returns `failures` (default empty dict = all workspaces ready).
    """
    if failures is None:
        failures = {}
    recorded: list[dict] = []

    async def _stub(cfg, *, split, num_samples, seed, reset_workspace,
                    concurrency):
        recorded.append({
            "cfg": cfg,
            "split": split,
            "num_samples": num_samples,
            "seed": seed,
            "reset_workspace": reset_workspace,
            "concurrency": concurrency,
        })
        return failures

    return _stub, recorded


def _invoke(args: list[str], monkeypatch, tmp_path: Path,
            failures: dict | None = None) -> tuple:
    """Invoke `main pre-clone <args>` with all external calls stubbed.

    Returns (click Result, recorded kwarg dicts from the stub).
    """
    stub, recorded = _make_pre_clone_stub(failures)

    monkeypatch.setattr("testbed.cli.config_mod.load",
                        lambda *a, **kw: _make_cfg(str(tmp_path / "ws")))
    monkeypatch.setattr("testbed.cli.runner_mod.pre_clone_run", stub)

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(main, ["pre-clone"] + args, catch_exceptions=False)
    return result, recorded


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_success_exits_0_and_prints_ready_count(monkeypatch, tmp_path):
    """Happy path: pre_clone_run returns {} (no failures) → exit 0 +
    'workspaces ready: N/N' on stdout."""
    result, _ = _invoke(["--num-samples", "5"], monkeypatch, tmp_path,
                        failures={})

    assert result.exit_code == 0, result.output
    assert "workspaces ready: 5/5" in result.output


def test_success_default_num_samples_is_10(monkeypatch, tmp_path):
    """Default --num-samples is 10 per cli.py."""
    result, recorded = _invoke([], monkeypatch, tmp_path, failures={})

    assert result.exit_code == 0
    assert recorded[0]["num_samples"] == 10
    assert "workspaces ready: 10/10" in result.output


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------

def test_one_failure_exits_1_and_shows_iid(monkeypatch, tmp_path):
    """When pre_clone_run returns a non-empty failures dict: exit 1,
    the failing iid appears somewhere in the output (stdout or stderr)."""
    failures = {"django__django-99": "RuntimeError: git clone exploded"}
    result, _ = _invoke(["--num-samples", "3"], monkeypatch, tmp_path,
                        failures=failures)

    assert result.exit_code == 1
    combined = result.output + (result.stderr or "")
    assert "django__django-99" in combined


def test_all_failures_exits_1(monkeypatch, tmp_path):
    """All N workspaces fail → exit 1, ready count shows 0/N."""
    n = 4
    failures = {f"iid_{i}": "boom" for i in range(n)}
    result, _ = _invoke(["--num-samples", str(n)], monkeypatch, tmp_path,
                        failures=failures)

    assert result.exit_code == 1
    assert f"workspaces ready: 0/{n}" in result.output


def test_partial_failure_shows_ready_count(monkeypatch, tmp_path):
    """2 of 5 fail → 'workspaces ready: 3/5'."""
    failures = {"iid_a": "err", "iid_b": "err"}
    result, _ = _invoke(["--num-samples", "5"], monkeypatch, tmp_path,
                        failures=failures)

    assert result.exit_code == 1
    assert "workspaces ready: 3/5" in result.output


# ---------------------------------------------------------------------------
# Kwarg forwarding
# ---------------------------------------------------------------------------

def test_reset_workspace_flag_forwarded(monkeypatch, tmp_path):
    """--reset-workspace flows to pre_clone_run as reset_workspace=True."""
    _, recorded = _invoke(["--reset-workspace"], monkeypatch, tmp_path)

    assert recorded[0]["reset_workspace"] is True


def test_no_reset_workspace_flag_default_false(monkeypatch, tmp_path):
    """Without --reset-workspace the kwarg defaults to False."""
    _, recorded = _invoke([], monkeypatch, tmp_path)

    assert recorded[0]["reset_workspace"] is False


def test_concurrency_forwarded(monkeypatch, tmp_path):
    """--concurrency 16 reaches pre_clone_run as concurrency=16."""
    _, recorded = _invoke(["--concurrency", "16"], monkeypatch, tmp_path)

    assert recorded[0]["concurrency"] == 16


def test_default_concurrency_is_8(monkeypatch, tmp_path):
    """Default concurrency is 8 per cli.py."""
    _, recorded = _invoke([], monkeypatch, tmp_path)

    assert recorded[0]["concurrency"] == 8


def test_repo_cache_flag_rejected_as_unknown_option(monkeypatch, tmp_path):
    """--repo-cache was removed from the pre-clone command; invoking it must
    be rejected by Click as an unrecognised option (exit code 2)."""
    stub, _ = _make_pre_clone_stub()
    monkeypatch.setattr("testbed.cli.config_mod.load",
                        lambda *a, **kw: _make_cfg(str(tmp_path / "ws")))
    monkeypatch.setattr("testbed.cli.runner_mod.pre_clone_run", stub)

    cli_runner = CliRunner(mix_stderr=False)
    result = cli_runner.invoke(main, ["pre-clone", "--repo-cache"],
                               catch_exceptions=False)
    assert result.exit_code == 2
    assert "no such option" in (result.output + (result.stderr or "")).lower()


def test_split_forwarded(monkeypatch, tmp_path):
    """--split verified reaches pre_clone_run as split='verified'."""
    _, recorded = _invoke(["--split", "verified"], monkeypatch, tmp_path)

    assert recorded[0]["split"] == "verified"


def test_seed_forwarded(monkeypatch, tmp_path):
    """--seed 99 reaches pre_clone_run as seed=99."""
    _, recorded = _invoke(["--seed", "99"], monkeypatch, tmp_path)

    assert recorded[0]["seed"] == 99
