from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from testbed import runner
from testbed.runner import TaskRecord, _run_one, _summary


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
    async def _ok(repo: str, base_commit: str, dest: Path) -> None:
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


async def test_clone_failure_marks_clone_stage(monkeypatch, tmp_path: Path):
    async def _boom(repo, base, dest):
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
