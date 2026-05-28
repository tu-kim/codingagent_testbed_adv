from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from testbed import runner
from testbed.runner import TaskRecord, _env_truthy, _run_one, _summary


class _FakeClient:
    """Stand-in for OpenCodeClient — drives the four failure stages by configuration."""

    def __init__(
        self,
        *,
        session_raises: BaseException | None = None,
        message_raises: BaseException | None = None,
        list_raises: BaseException | None = None,
        list_payload: list[dict[str, Any]] | None = None,
    ) -> None:
        self.session_raises = session_raises
        self.message_raises = message_raises
        self.list_raises = list_raises
        self.list_payload = list_payload if list_payload is not None else [{"info": {}, "parts": []}]
        self.create_calls: list[str] = []
        self.send_calls: list[tuple[str, str, str]] = []
        self.list_calls: list[tuple[str, str]] = []

    async def create_session(self, directory: str) -> str:
        self.create_calls.append(directory)
        if self.session_raises is not None:
            raise self.session_raises
        return "ses_test"

    async def send_message(self, session_id: str, prompt: str, directory: str) -> dict:
        self.send_calls.append((session_id, prompt, directory))
        if self.message_raises is not None:
            raise self.message_raises
        return {"info": {"id": "msg_x"}, "parts": []}

    async def list_messages(self, session_id: str, directory: str) -> list[dict]:
        self.list_calls.append((session_id, directory))
        if self.list_raises is not None:
            raise self.list_raises
        return self.list_payload


_SAMPLE = {
    "instance_id": "django__django-1",
    "repo": "django/django",
    "base_commit": "abc123",
    "problem_statement": "x",
    "hints_text": "",
}


@pytest.fixture(autouse=True)
def _stub_pre_clone(monkeypatch):
    async def _ok(repo: str, base_commit: str, dest: Path, *, reset: bool = False) -> None:
        return None

    monkeypatch.setattr(runner, "_pre_clone", _ok)


async def test_happy_path_records_success_and_messages(tmp_path: Path):
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    rec = await _run_one(client, _SAMPLE, 1.5, tmp_path, sem)
    assert rec.success is True
    assert rec.error is None
    assert rec.session_id == "ses_test"
    assert rec.directory.startswith("session-django__django-1-")
    assert rec.rtt_s is not None and rec.rtt_s >= 0.0
    assert rec.messages == [{"info": {}, "parts": []}]
    # Runner must send an ABSOLUTE path to OpenCode — its InstanceMiddleware
    # resolves `?directory=` against its own CWD, so a bare subfolder name
    # would silently mis-anchor onto opencode/<name>/ instead of
    # <workspace_root>/<name>/.
    sent = client.send_calls[0][2]
    assert Path(sent).is_absolute(), f"directory passed to OpenCode must be absolute, got {sent!r}"
    assert sent == str(tmp_path / rec.directory)
    # create_session got the same absolute path.
    assert client.create_calls[0] == sent
    # list_messages too.
    assert client.list_calls[0][1] == sent


async def test_reset_workspace_strips_uuid_suffix_from_directory(tmp_path: Path):
    """With reset_workspace=True, the workspace dir name is
    `session-<instance_id>` (no uuid). Same sample run twice produces
    the SAME absolute path -- the load-bearing property for
    reproducible system prompts (opencode embeds cwd into the prompt
    so a different path -> different first token -> divergent agent
    loop)."""
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    rec1 = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem, reset_workspace=True)
    rec2 = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem, reset_workspace=True)
    # No uuid8 suffix appended:
    assert rec1.directory == "session-django__django-1"
    # And it's stable across two invocations (the whole point).
    assert rec1.directory == rec2.directory


async def test_reset_workspace_default_keeps_uuid_suffix(tmp_path: Path):
    """Backward compat: reset_workspace defaults to False, dir name
    keeps the uuid8 suffix so concurrent runs of the same instance_id
    don't collide on disk."""
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    rec1 = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem)
    rec2 = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem)
    assert rec1.directory != rec2.directory
    # Each has the legacy `session-<id>-<8hex>` shape.
    assert rec1.directory.startswith("session-django__django-1-")
    assert rec1.directory.split("-")[-1] != rec2.directory.split("-")[-1]


async def test_reset_workspace_forwards_reset_kwarg_to_pre_clone(monkeypatch, tmp_path: Path):
    """The flag has to actually flow to _pre_clone(reset=True) -- otherwise
    the dir name is stable but the workspace state from the prior agent
    run isn't cleaned, defeating the reproducibility purpose."""
    seen: list[bool] = []
    async def _spy(repo, base, dest, *, reset=False):
        seen.append(reset)
    monkeypatch.setattr(runner, "_pre_clone", _spy)
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    await _run_one(client, _SAMPLE, 0.0, tmp_path, sem, reset_workspace=True)
    await _run_one(client, _SAMPLE, 0.0, tmp_path, sem, reset_workspace=False)
    assert seen == [True, False]


async def test_clone_failure_marks_clone_stage(monkeypatch, tmp_path: Path):
    async def _boom(repo, base, dest, *, reset=False):
        raise RuntimeError("git clone failed: network")

    monkeypatch.setattr(runner, "_pre_clone", _boom)
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    rec = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem)
    assert rec.success is False
    assert rec.error and rec.error["stage"] == "clone"
    assert rec.rtt_s is None
    assert rec.session_id is None
    assert client.create_calls == []  # never reached the OpenCode call


async def test_session_failure_has_no_rtt(tmp_path: Path):
    client = _FakeClient(session_raises=RuntimeError("502 from opencode"))
    sem = asyncio.Semaphore(1)
    rec = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem)
    assert rec.success is False
    assert rec.error and rec.error["stage"] == "session"
    assert rec.rtt_s is None
    assert rec.session_id is None
    assert client.send_calls == []


async def test_message_failure_records_walltime_rtt(tmp_path: Path):
    client = _FakeClient(message_raises=RuntimeError("agent crashed"))
    sem = asyncio.Semaphore(1)
    rec = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem)
    assert rec.success is False
    assert rec.error and rec.error["stage"] == "message"
    assert rec.rtt_s is not None and rec.rtt_s >= 0.0
    assert rec.session_id == "ses_test"


async def test_list_failure_keeps_success_true_and_rtt_from_post(tmp_path: Path):
    client = _FakeClient(list_raises=RuntimeError("list timed out"))
    sem = asyncio.Semaphore(1)
    rec = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem)
    assert rec.success is True
    assert rec.error and rec.error["stage"] == "list"
    assert rec.messages == []
    assert rec.rtt_s is not None and rec.rtt_s >= 0.0


class _HangingClient(_FakeClient):
    """send_message blocks indefinitely — simulates an agent loop hang."""

    def __init__(self) -> None:
        super().__init__()
        self.send_started = asyncio.Event()

    async def send_message(self, session_id: str, prompt: str, directory: str) -> dict:
        self.send_started.set()
        # Sleep way longer than any test would tolerate, so the only way out
        # is the runner's asyncio.wait_for cap.
        await asyncio.sleep(3600)
        return {"info": {}, "parts": []}


async def test_message_timeout_marks_timeout_stage(tmp_path: Path):
    """When send_message exceeds task_timeout_s, the runner must abort it
    with error.stage='timeout' instead of hanging the whole run."""
    client = _HangingClient()
    sem = asyncio.Semaphore(1)
    rec = await _run_one(
        client, _SAMPLE, 0.0, tmp_path, sem, task_timeout_s=0.05
    )
    assert client.send_started.is_set(), "send_message must have started"
    assert rec.success is False
    assert rec.error and rec.error["stage"] == "timeout"
    assert rec.error["type"] == "TimeoutError"
    # rtt_s reflects wall-clock to the abort, not None.
    assert rec.rtt_s is not None and rec.rtt_s >= 0.05
    # session_id was acquired before the hang, so it's recoverable for diagnosis.
    assert rec.session_id == "ses_test"
    # No list_messages call happens after a timeout.
    assert client.list_calls == []


async def test_no_timeout_when_task_timeout_s_is_none(tmp_path: Path):
    """task_timeout_s=None must preserve the prior unbounded behavior."""
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    rec = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem, task_timeout_s=None)
    assert rec.success is True
    assert rec.error is None


def test_summary_p50_p95_and_zero_count():
    assert _summary([]) == {"count": 0, "success_rate": None, "rtt_s": {"p50": None, "p95": None}}

    rs = [
        TaskRecord("a", "ses_a", "d-a", 0.0, 1.0, True, None),
        TaskRecord("b", "ses_b", "d-b", 0.0, 2.0, True, None),
        TaskRecord("c", "ses_c", "d-c", 0.0, 3.0, True, None),
        TaskRecord("d", "ses_d", "d-d", 0.0, None, False, {"stage": "clone", "type": "X", "msg": "y"}),
    ]
    s = _summary(rs)
    assert s["count"] == 4
    assert s["success_rate"] == 0.75
    assert s["rtt_s"]["p50"] == pytest.approx(2.0, abs=0.01)
    assert s["rtt_s"]["p95"] >= 2.5  # interpolation, but well above the median


def test_taskrecord_jsonl_round_trip():
    rec = TaskRecord("a", "ses_x", "d-a", 0.5, 1.25, True, None, [{"info": {"id": "m1"}, "parts": []}])
    line = rec.to_jsonl()
    assert line.endswith("\n")
    parsed = json.loads(line)
    assert parsed == asdict(rec)


# ---------------------------------------------------------------------------
# _env_truthy truth table
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    # Falsy: unset, empty, "0", "false" (any case)
    (None,    False),
    ("",      False),
    ("0",     False),
    ("false", False),
    ("FALSE", False),
    ("False", False),
    # Truthy: anything else
    ("1",     True),
    ("true",  True),
    ("TRUE",  True),
    ("yes",   True),
    ("on",    True),
    ("2",     True),
    ("enabled", True),
])
def test_env_truthy_truth_table(value, expected):
    assert _env_truthy(value) is expected


# ---------------------------------------------------------------------------
# Minimal TestbedCfg builder — used by run() tests below.
# ---------------------------------------------------------------------------

def _minimal_cfg(workspace_root: str):
    """Return a TestbedCfg with only the fields run() actually touches."""
    from testbed.config import (
        DynamoCfg,
        ModelCfg,
        OpenCodeCfg,
        TestbedCfg,
        VLLMCfg,
        VLLMRoleCfg,
        WorkerCfg,
    )
    role_cfg = VLLMRoleCfg(
        max_model_len=4096,
        max_num_batched_tokens=1024,
        max_num_seqs=8,
        gpu_memory_utilization=0.9,
    )
    worker = WorkerCfg(name="w0", gpus="0", tp=1, pp=1)
    vllm = VLLMCfg(
        prefill_workers=[worker],
        decode_workers=[worker],
        prefill=role_cfg,
        decode=role_cfg,
    )
    return TestbedCfg(
        workspace_root=workspace_root,
        model=ModelCfg(name="test-model", served_name="local"),
        vllm=vllm,
        dynamo=DynamoCfg(),
        opencode=OpenCodeCfg(),
    )


# ---------------------------------------------------------------------------
# Fake async context manager that yields a _FakeClient
# ---------------------------------------------------------------------------

class _FakeClientCtx:
    """Wraps _FakeClient as an async context manager so it can stand in for
    ``async with OpenCodeClient(...) as client``."""

    def __init__(self, fake_client):
        self._client = fake_client

    async def __aenter__(self):
        return self._client

    async def __aexit__(self, *_):
        pass


# ---------------------------------------------------------------------------
# run() → config.json opencode_profile tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_config_json_opencode_profile_enabled_when_env_set(
    monkeypatch, tmp_path: Path
):
    """OPENCODE_PROFILE=1 → config.json opencode_profile.enabled is True."""
    monkeypatch.setenv("OPENCODE_PROFILE", "1")
    monkeypatch.setenv("OPENCODE_PROFILE_DIR", "/tmp/prof")
    monkeypatch.setenv("OPENCODE_PROFILE_MESSAGES", "50")
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)

    workspace = tmp_path / "ws"
    cfg = _minimal_cfg(str(workspace))
    fake_client = _FakeClient()

    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake_client)):
        with patch.object(runner.swebench, "load_samples", return_value=[_SAMPLE]):
            with patch.object(runner.poisson, "arrival_offsets", return_value=[0.0]):
                with patch.object(runner.poisson, "arrivals", _fake_arrivals([0])):
                    await runner.run(
                        cfg,
                        split="lite",
                        num_samples=1,
                        qps=1.0,
                        seed=42,
                        max_in_flight=1,
                        out_dir=tmp_path / "out",
                        router_label="test",
                        task_timeout_s=None,
                    )

    config_json = json.loads((tmp_path / "out" / "config.json").read_text())
    profile = config_json["opencode_profile"]
    assert profile["enabled"] is True
    assert profile["raw"] == "1"
    assert profile["dir"] == "/tmp/prof"
    assert profile["messages"] == "50"


@pytest.mark.asyncio
async def test_run_config_json_opencode_profile_disabled_when_env_unset(
    monkeypatch, tmp_path: Path
):
    """Unset OPENCODE_PROFILE → config.json opencode_profile.enabled is False."""
    monkeypatch.delenv("OPENCODE_PROFILE", raising=False)
    monkeypatch.delenv("OPENCODE_PROFILE_DIR", raising=False)
    monkeypatch.delenv("OPENCODE_PROFILE_MESSAGES", raising=False)
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)

    workspace = tmp_path / "ws"
    cfg = _minimal_cfg(str(workspace))
    fake_client = _FakeClient()

    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake_client)):
        with patch.object(runner.swebench, "load_samples", return_value=[_SAMPLE]):
            with patch.object(runner.poisson, "arrival_offsets", return_value=[0.0]):
                with patch.object(runner.poisson, "arrivals", _fake_arrivals([0])):
                    await runner.run(
                        cfg,
                        split="lite",
                        num_samples=1,
                        qps=1.0,
                        seed=42,
                        max_in_flight=1,
                        out_dir=tmp_path / "out",
                        router_label="test",
                        task_timeout_s=None,
                    )

    config_json = json.loads((tmp_path / "out" / "config.json").read_text())
    profile = config_json["opencode_profile"]
    assert profile["enabled"] is False
    assert profile["raw"] is None
    assert profile["dir"] is None
    assert profile["messages"] is None


@pytest.mark.asyncio
async def test_run_config_json_opencode_profile_dir_and_messages_round_trip(
    monkeypatch, tmp_path: Path
):
    """dir and messages values are captured verbatim when set."""
    monkeypatch.setenv("OPENCODE_PROFILE", "true")
    monkeypatch.setenv("OPENCODE_PROFILE_DIR", "/var/prof/run7")
    monkeypatch.setenv("OPENCODE_PROFILE_MESSAGES", "100")
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)

    workspace = tmp_path / "ws"
    cfg = _minimal_cfg(str(workspace))
    fake_client = _FakeClient()

    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake_client)):
        with patch.object(runner.swebench, "load_samples", return_value=[_SAMPLE]):
            with patch.object(runner.poisson, "arrival_offsets", return_value=[0.0]):
                with patch.object(runner.poisson, "arrivals", _fake_arrivals([0])):
                    await runner.run(
                        cfg,
                        split="lite",
                        num_samples=1,
                        qps=1.0,
                        seed=42,
                        max_in_flight=1,
                        out_dir=tmp_path / "out",
                        router_label="test",
                        task_timeout_s=None,
                    )

    config_json = json.loads((tmp_path / "out" / "config.json").read_text())
    profile = config_json["opencode_profile"]
    assert profile["enabled"] is True
    assert profile["dir"] == "/var/prof/run7"
    assert profile["messages"] == "100"


# ---------------------------------------------------------------------------
# Helper: async generator that yields index values, mimicking poisson.arrivals
# ---------------------------------------------------------------------------

def _fake_arrivals(indices):
    """Return a coroutine-based replacement for poisson.arrivals that yields
    the given index sequence without sleeping."""
    async def _gen(*_args, **_kwargs):
        for i in indices:
            yield i
    return _gen
