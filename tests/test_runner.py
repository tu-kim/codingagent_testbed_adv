from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from testbed import runner
from testbed.runner import (
    TaskRecord,
    _directory_for,
    _env_truthy,
    _run_one,
    _summary,
    prepare_workspaces,
)


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
    async def _ok(repo: str, base_commit: str, dest: Path, *,
                  reset: bool = False, cache_dir=None) -> None:
        return None

    monkeypatch.setattr(runner, "_pre_clone", _ok)

    # run() warms the repo cache before the task loop; stub it so the
    # run-level tests never shell out to real git / the network.
    async def _warm(samples, cache_dir, **kw):
        return {}

    monkeypatch.setattr(runner, "warm_repo_cache", _warm)


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
    async def _spy(repo, base, dest, *, reset=False, cache_dir=None):
        seen.append(reset)
    monkeypatch.setattr(runner, "_pre_clone", _spy)
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    await _run_one(client, _SAMPLE, 0.0, tmp_path, sem, reset_workspace=True)
    await _run_one(client, _SAMPLE, 0.0, tmp_path, sem, reset_workspace=False)
    assert seen == [True, False]


async def test_run_one_forwards_repo_cache_dir_to_pre_clone(monkeypatch, tmp_path: Path):
    """The cache dir must reach _pre_clone so the per-task clone copies
    from the warmed cache instead of hitting the network."""
    seen: list = []
    async def _spy(repo, base, dest, *, reset=False, cache_dir=None):
        seen.append(cache_dir)
    monkeypatch.setattr(runner, "_pre_clone", _spy)
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    cache = tmp_path / "cache"
    await _run_one(client, _SAMPLE, 0.0, tmp_path, sem, repo_cache_dir=cache)
    await _run_one(client, _SAMPLE, 0.0, tmp_path, sem)   # default None
    assert seen == [cache, None]


async def test_clone_failure_marks_clone_stage(monkeypatch, tmp_path: Path):
    async def _boom(repo, base, dest, *, reset=False, cache_dir=None):
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
    with error.stage='timeout' instead of hanging the whole run -- and
    best-effort fetch the turns completed before the abort."""
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
    # Partial turns ARE fetched after the abort (so the trace shows
    # how far the agent got), and the count is recorded on the error.
    assert client.list_calls == [("ses_test", str(tmp_path / rec.directory))]
    assert rec.messages == [{"info": {}, "parts": []}]
    assert rec.error["partial_messages"] == 1


class _HangingListFailsClient(_HangingClient):
    """Hangs on send AND fails the post-timeout list (e.g. session gone)."""
    async def list_messages(self, session_id: str, directory: str) -> list[dict]:
        raise RuntimeError("session unavailable after abort")


async def test_timeout_partial_list_failure_falls_back_to_empty(tmp_path: Path):
    """If the best-effort partial list fails, the timeout record still
    lands cleanly with messages=[] (the list failure must not mask the
    timeout or re-hang the task)."""
    client = _HangingListFailsClient()
    sem = asyncio.Semaphore(1)
    rec = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem, task_timeout_s=0.05)
    assert rec.error and rec.error["stage"] == "timeout"
    assert rec.messages == []
    assert rec.error["partial_messages"] == 0


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


# ---------------------------------------------------------------------------
# Sequential mode
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_sequential_skips_poisson_and_marks_config(
    monkeypatch, tmp_path: Path
):
    """sequential=True must NOT touch poisson.arrivals / arrival_offsets and
    config.json must record sequential=True. This is the load-bearing
    guarantee: in sequential mode the Poisson model is bypassed entirely."""
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)
    cfg = _minimal_cfg(str(tmp_path / "ws"))
    fake_client = _FakeClient()

    def _boom(*a, **kw):
        raise AssertionError("poisson must not be called in sequential mode")

    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake_client)):
        with patch.object(runner.swebench, "load_samples", return_value=[_SAMPLE, _SAMPLE]):
            with patch.object(runner.poisson, "arrival_offsets", side_effect=_boom):
                with patch.object(runner.poisson, "arrivals", side_effect=_boom):
                    await runner.run(
                        cfg,
                        split="lite",
                        num_samples=2,
                        qps=1.0,
                        seed=42,
                        max_in_flight=16,   # ignored in sequential mode
                        out_dir=tmp_path / "out",
                        router_label="test",
                        task_timeout_s=None,
                        sequential=True,
                    )

    config_json = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config_json["sequential"] is True
    # qps and max_in_flight are still recorded (so reproducibility is
    # documented) but did not influence execution.
    assert config_json["qps"] == 1.0
    assert config_json["max_in_flight"] == 16


@pytest.mark.asyncio
async def test_run_sequential_executes_strictly_back_to_back(
    monkeypatch, tmp_path: Path
):
    """In sequential mode, task N+1's create_session MUST be called only
    AFTER task N's list_messages completed. Verified by tracking event
    order on a fake client whose send_message takes measurable time."""
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)
    cfg = _minimal_cfg(str(tmp_path / "ws"))

    # Two samples with distinct ids so directory names differ.
    samples = [
        {**_SAMPLE, "instance_id": "task_a"},
        {**_SAMPLE, "instance_id": "task_b"},
    ]
    events: list[str] = []

    class _OrderedFakeClient:
        async def create_session(self, directory: str) -> str:
            events.append(f"create:{directory.split('/')[-1]}")
            return "ses_" + directory.split("-")[-1]
        async def send_message(self, session_id, prompt, directory):
            events.append(f"send-start:{session_id}")
            # Yield to the loop so any incorrectly-pipelined create_session
            # call from the next task would race in here. In strict
            # sequential mode no such call should exist.
            await asyncio.sleep(0.01)
            events.append(f"send-end:{session_id}")
            return {"info": {}, "parts": []}
        async def list_messages(self, session_id, directory):
            events.append(f"list:{session_id}")
            return [{"info": {}, "parts": []}]

    fake_client = _OrderedFakeClient()
    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake_client)):
        with patch.object(runner.swebench, "load_samples", return_value=samples):
            await runner.run(
                cfg,
                split="lite",
                num_samples=2,
                qps=1.0,
                seed=42,
                max_in_flight=16,
                out_dir=tmp_path / "out",
                router_label="",
                task_timeout_s=None,
                sequential=True,
            )

    # Task A's full lifecycle must end before any event of task B begins.
    a_last = max(i for i, e in enumerate(events) if "task_a" in e or e.endswith("ses_a"))
    b_first = min(
        i for i, e in enumerate(events) if "task_b" in e or e.endswith("ses_b")
    )
    assert a_last < b_first, f"events interleaved: {events}"


@pytest.mark.asyncio
async def test_run_sequential_arrival_offset_grows_with_elapsed(
    monkeypatch, tmp_path: Path
):
    """arrival_offset_s in sequential mode = elapsed wall-clock when the
    task began. Task 0 starts at ~0; subsequent tasks' offsets must be
    strictly > prior task's offset (cumulative)."""
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)
    cfg = _minimal_cfg(str(tmp_path / "ws"))
    samples = [
        {**_SAMPLE, "instance_id": f"task_{i}"} for i in range(3)
    ]

    class _SlowFakeClient(_FakeClient):
        async def send_message(self, session_id, prompt, directory):
            await asyncio.sleep(0.02)
            return await super().send_message(session_id, prompt, directory)

    fake = _SlowFakeClient()
    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake)):
        with patch.object(runner.swebench, "load_samples", return_value=samples):
            await runner.run(
                cfg,
                split="lite", num_samples=3, qps=1.0, seed=42, max_in_flight=16,
                out_dir=tmp_path / "out", router_label="",
                task_timeout_s=None, sequential=True,
            )

    trace = [json.loads(l) for l in (tmp_path / "out" / "trace.jsonl").read_text().splitlines()]
    offsets = [t["arrival_offset_s"] for t in trace]
    assert offsets[0] >= 0.0
    # Each task's offset must exceed the prior's by at least the sleep
    # duration (per-task RTT includes the 20ms send sleep).
    for prev, curr in zip(offsets, offsets[1:]):
        assert curr > prev, f"offsets not monotonic: {offsets}"


# ---------------------------------------------------------------------------
# Progress output
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_prints_progress_to_stderr(monkeypatch, tmp_path, capsys):
    """run() emits a start banner, one line per completed task, and a
    final summary -- all to STDERR (stdout stays clean). The per-task
    line carries the running done/total counter."""
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)
    cfg = _minimal_cfg(str(tmp_path / "ws"))
    samples = [{**_SAMPLE, "instance_id": f"task_{i}"} for i in range(2)]
    fake = _FakeClient()

    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake)):
        with patch.object(runner.swebench, "load_samples", return_value=samples):
            await runner.run(
                cfg,
                split="lite", num_samples=2, qps=1.0, seed=42, max_in_flight=16,
                out_dir=tmp_path / "out", router_label="",
                task_timeout_s=None, sequential=True,
            )

    captured = capsys.readouterr()
    err = captured.err
    # Start banner names the mode and task count.
    assert "testbed run: 2 tasks (sequential)" in err
    # One progress line per task with the running counter.
    assert "[  1/2]" in err
    assert "[  2/2]" in err
    # Cumulative tally + final summary.
    assert "ok=2 fail=0" in err
    assert "done. 2 tasks, 2 ok / 0 fail" in err
    # Progress must NOT leak onto stdout.
    assert "testbed run" not in captured.out


@pytest.mark.asyncio
async def test_run_progress_marks_failed_task(monkeypatch, tmp_path, capsys):
    """A failed task shows FAIL:<stage> and increments the fail tally."""
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)
    cfg = _minimal_cfg(str(tmp_path / "ws"))
    samples = [{**_SAMPLE, "instance_id": "task_x"}]

    class _FailingClient(_FakeClient):
        async def create_session(self, directory: str) -> str:
            raise RuntimeError("boom")

    with patch.object(runner, "OpenCodeClient",
                      return_value=_FakeClientCtx(_FailingClient())):
        with patch.object(runner.swebench, "load_samples", return_value=samples):
            await runner.run(
                cfg,
                split="lite", num_samples=1, qps=1.0, seed=42, max_in_flight=16,
                out_dir=tmp_path / "out", router_label="",
                task_timeout_s=None, sequential=True,
            )

    err = capsys.readouterr().err
    assert "FAIL:session" in err          # create_session failure → stage "session"
    assert "done. 1 tasks, 0 ok / 1 fail" in err


# ---------------------------------------------------------------------------
# _directory_for
# ---------------------------------------------------------------------------

def test_directory_for_reset_returns_stable_name():
    """reset_workspace=True → exact 'session-<iid>', no uuid suffix."""
    name = _directory_for("django__django-1", reset_workspace=True)
    assert name == "session-django__django-1"


def test_directory_for_no_reset_has_8hex_suffix():
    """reset_workspace=False → 'session-<iid>-<8hexchars>' shape."""
    name = _directory_for("django__django-1", reset_workspace=False)
    assert name.startswith("session-django__django-1-")
    suffix = name[len("session-django__django-1-"):]
    assert len(suffix) == 8
    assert all(c in "0123456789abcdef" for c in suffix)


def test_directory_for_two_calls_differ():
    """Two non-reset calls for the same iid must produce distinct names."""
    a = _directory_for("django__django-1", reset_workspace=False)
    b = _directory_for("django__django-1", reset_workspace=False)
    assert a != b


# ---------------------------------------------------------------------------
# prepare_workspaces
# ---------------------------------------------------------------------------

async def test_prepare_workspaces_clones_all_samples(monkeypatch, tmp_path: Path):
    """Every sample must be cloned with dest=workspace_root/directories[iid]
    and the correct reset/cache_dir kwargs forwarded to _pre_clone."""
    samples = [
        {**_SAMPLE, "instance_id": "iid_a", "repo": "org/repo-a", "base_commit": "aaa"},
        {**_SAMPLE, "instance_id": "iid_b", "repo": "org/repo-b", "base_commit": "bbb"},
    ]
    directories = {"iid_a": "session-iid_a", "iid_b": "session-iid_b"}
    cache = tmp_path / "cache"
    calls: list[dict] = []

    async def _recording(repo, base_commit, dest, *, reset=False, cache_dir=None):
        calls.append({"repo": repo, "base_commit": base_commit,
                      "dest": dest, "reset": reset, "cache_dir": cache_dir})

    monkeypatch.setattr(runner, "_pre_clone", _recording)

    failures = await prepare_workspaces(
        samples, directories, tmp_path,
        reset_workspace=True, cache_dir=cache, concurrency=2,
    )

    assert failures == {}
    assert len(calls) == 2
    by_iid = {c["repo"].split("/")[1]: c for c in calls}   # "repo-a" / "repo-b"

    call_a = by_iid["repo-a"]
    assert call_a["base_commit"] == "aaa"
    assert call_a["dest"] == tmp_path / "session-iid_a"
    assert call_a["reset"] is True
    assert call_a["cache_dir"] == cache

    call_b = by_iid["repo-b"]
    assert call_b["base_commit"] == "bbb"
    assert call_b["dest"] == tmp_path / "session-iid_b"
    assert call_b["reset"] is True
    assert call_b["cache_dir"] == cache


async def test_prepare_workspaces_collects_failures_without_raising(monkeypatch, tmp_path: Path):
    """A _pre_clone failure for one iid is recorded in the returned dict;
    other samples succeed; no exception propagates out of prepare_workspaces."""
    samples = [
        {**_SAMPLE, "instance_id": "ok_task"},
        {**_SAMPLE, "instance_id": "bad_task"},
    ]
    directories = {"ok_task": "session-ok_task", "bad_task": "session-bad_task"}

    async def _selective(repo, base_commit, dest, *, reset=False, cache_dir=None):
        if "bad_task" in str(dest):
            raise RuntimeError("git clone exploded")

    monkeypatch.setattr(runner, "_pre_clone", _selective)

    failures = await prepare_workspaces(samples, directories, tmp_path)

    assert set(failures.keys()) == {"bad_task"}
    assert "RuntimeError" in failures["bad_task"]
    assert "git clone exploded" in failures["bad_task"]
    # ok_task must NOT appear in failures
    assert "ok_task" not in failures


# ---------------------------------------------------------------------------
# _run_one with explicit directory= kwarg
# ---------------------------------------------------------------------------

async def test_run_one_explicit_directory_used_verbatim(tmp_path: Path):
    """When directory= is supplied to _run_one, the TaskRecord.directory
    must be exactly that value — not a freshly generated uuid-suffixed name."""
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    rec = await _run_one(
        client, _SAMPLE, 0.0, tmp_path, sem,
        directory="session-django__django-1-cafebabe",
    )
    assert rec.directory == "session-django__django-1-cafebabe"
    # The absolute path sent to OpenCode must be derived from that name.
    assert client.create_calls[0] == str(tmp_path / "session-django__django-1-cafebabe")


async def test_run_one_explicit_directory_none_falls_back_to_directory_for(tmp_path: Path):
    """directory=None (default) must produce a name via _directory_for
    (i.e. the session-<iid>-<uuid8> prefix form)."""
    client = _FakeClient()
    sem = asyncio.Semaphore(1)
    rec = await _run_one(client, _SAMPLE, 0.0, tmp_path, sem, directory=None)
    assert rec.directory.startswith("session-django__django-1-")
    suffix = rec.directory[len("session-django__django-1-"):]
    assert len(suffix) == 8


# ---------------------------------------------------------------------------
# run() pre_clone_workspaces=True — call ordering and directory stability
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_pre_clone_workspaces_clones_before_session(
    monkeypatch, tmp_path: Path
):
    """With pre_clone_workspaces=True, ALL _pre_clone calls from prepare_workspaces
    must complete before the first create_session call fires. Verified via a
    shared event log that records 'clone:<iid>' and 'create:<iid>' in order."""
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)
    samples = [
        {**_SAMPLE, "instance_id": "task_a"},
        {**_SAMPLE, "instance_id": "task_b"},
    ]
    cfg = _minimal_cfg(str(tmp_path / "ws"))
    events: list[str] = []

    async def _recording_clone(repo, base_commit, dest, *, reset=False, cache_dir=None):
        iid = str(dest).split("/")[-1].replace("session-", "")
        events.append(f"clone:{iid}")

    monkeypatch.setattr(runner, "_pre_clone", _recording_clone)

    class _EventFakeClient(_FakeClient):
        async def create_session(self, directory: str) -> str:
            iid = directory.split("/")[-1].replace("session-", "")
            events.append(f"create:{iid}")
            return "ses_test"

    fake_client = _EventFakeClient()
    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake_client)):
        with patch.object(runner.swebench, "load_samples", return_value=samples):
            await runner.run(
                cfg,
                split="lite", num_samples=2, qps=1.0, seed=42,
                max_in_flight=2, out_dir=tmp_path / "out", router_label="",
                task_timeout_s=None, sequential=True,
                pre_clone_workspaces=True,
            )

    # All clone events must precede all create events.
    clone_indices = [i for i, e in enumerate(events) if e.startswith("clone:")]
    create_indices = [i for i, e in enumerate(events) if e.startswith("create:")]
    assert clone_indices, "no clone events recorded"
    assert create_indices, "no create events recorded"
    assert max(clone_indices) < min(create_indices), (
        f"some clone happened after first create — events: {events}"
    )


@pytest.mark.asyncio
async def test_run_pre_clone_workspaces_trace_directories_match_pre_assigned(
    monkeypatch, tmp_path: Path
):
    """With pre_clone_workspaces=True and reset_workspace=True, trace records'
    directory values are the stable 'session-<iid>' names assigned up front
    (not freshly generated uuid-suffixed names per task)."""
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)
    samples = [
        {**_SAMPLE, "instance_id": "task_a"},
        {**_SAMPLE, "instance_id": "task_b"},
    ]
    cfg = _minimal_cfg(str(tmp_path / "ws"))
    fake_client = _FakeClient()

    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake_client)):
        with patch.object(runner.swebench, "load_samples", return_value=samples):
            await runner.run(
                cfg,
                split="lite", num_samples=2, qps=1.0, seed=42,
                max_in_flight=2, out_dir=tmp_path / "out", router_label="",
                task_timeout_s=None, sequential=True,
                reset_workspace=True, pre_clone_workspaces=True,
            )

    trace = [json.loads(l) for l in (tmp_path / "out" / "trace.jsonl").read_text().splitlines()]
    dirs_by_iid = {t["instance_id"]: t["directory"] for t in trace}
    assert dirs_by_iid["task_a"] == "session-task_a"
    assert dirs_by_iid["task_b"] == "session-task_b"


# ---------------------------------------------------------------------------
# run() pre_clone_workspaces=False — _pre_clone called per-task, not in batch
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_pre_clone_workspaces_false_skips_batch_phase(
    monkeypatch, tmp_path: Path
):
    """With pre_clone_workspaces=False, prepare_workspaces must NOT be called.
    _pre_clone is still called exactly once per task inside _run_one."""
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)
    samples = [
        {**_SAMPLE, "instance_id": "task_a"},
        {**_SAMPLE, "instance_id": "task_b"},
    ]
    cfg = _minimal_cfg(str(tmp_path / "ws"))
    clone_calls: list[str] = []

    async def _recording_clone(repo, base_commit, dest, *, reset=False, cache_dir=None):
        clone_calls.append(str(dest).split("/")[-1])

    monkeypatch.setattr(runner, "_pre_clone", _recording_clone)

    prepare_ws_calls: list[int] = []
    original_prepare = runner.prepare_workspaces

    async def _spy_prepare(*args, **kwargs):
        prepare_ws_calls.append(1)
        return await original_prepare(*args, **kwargs)

    monkeypatch.setattr(runner, "prepare_workspaces", _spy_prepare)

    fake_client = _FakeClient()
    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake_client)):
        with patch.object(runner.swebench, "load_samples", return_value=samples):
            await runner.run(
                cfg,
                split="lite", num_samples=2, qps=1.0, seed=42,
                max_in_flight=2, out_dir=tmp_path / "out", router_label="",
                task_timeout_s=None, sequential=True,
                pre_clone_workspaces=False,
            )

    # prepare_workspaces must NOT have been called.
    assert prepare_ws_calls == [], "prepare_workspaces must not be called when pre_clone_workspaces=False"
    # _pre_clone IS called once per task (from _run_one).
    assert len(clone_calls) == 2


# ---------------------------------------------------------------------------
# config.json records pre_clone_workspaces flag
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_config_json_pre_clone_workspaces_true(monkeypatch, tmp_path: Path):
    """pre_clone_workspaces=True must be recorded in config.json."""
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)
    cfg = _minimal_cfg(str(tmp_path / "ws"))
    fake_client = _FakeClient()

    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake_client)):
        with patch.object(runner.swebench, "load_samples", return_value=[_SAMPLE]):
            with patch.object(runner.poisson, "arrival_offsets", return_value=[0.0]):
                with patch.object(runner.poisson, "arrivals", _fake_arrivals([0])):
                    await runner.run(
                        cfg,
                        split="lite", num_samples=1, qps=1.0, seed=42,
                        max_in_flight=1, out_dir=tmp_path / "out", router_label="",
                        task_timeout_s=None, pre_clone_workspaces=True,
                    )

    config_json = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config_json["pre_clone_workspaces"] is True


@pytest.mark.asyncio
async def test_run_config_json_pre_clone_workspaces_false(monkeypatch, tmp_path: Path):
    """pre_clone_workspaces=False must be recorded in config.json."""
    monkeypatch.delenv("OPENCODE_SERVER_PASSWORD", raising=False)
    cfg = _minimal_cfg(str(tmp_path / "ws"))
    fake_client = _FakeClient()

    with patch.object(runner, "OpenCodeClient", return_value=_FakeClientCtx(fake_client)):
        with patch.object(runner.swebench, "load_samples", return_value=[_SAMPLE]):
            with patch.object(runner.poisson, "arrival_offsets", return_value=[0.0]):
                with patch.object(runner.poisson, "arrivals", _fake_arrivals([0])):
                    await runner.run(
                        cfg,
                        split="lite", num_samples=1, qps=1.0, seed=42,
                        max_in_flight=1, out_dir=tmp_path / "out", router_label="",
                        task_timeout_s=None, pre_clone_workspaces=False,
                    )

    config_json = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config_json["pre_clone_workspaces"] is False
