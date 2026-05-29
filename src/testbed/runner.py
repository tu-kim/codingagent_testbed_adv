"""Poisson workload driver: pre-clone, drive OpenCode, write trace.jsonl + summary.json."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import poisson, swebench
from .config import TestbedCfg, resolved_snapshot
from .opencode import OpenCodeClient


@dataclass
class TaskRecord:
    instance_id: str
    session_id: str | None
    directory: str
    arrival_offset_s: float
    rtt_s: float | None
    success: bool
    error: dict[str, Any] | None
    messages: list[Any] = field(default_factory=list)

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self)) + "\n"


def _err(stage: str, exc: BaseException) -> dict[str, Any]:
    return {"stage": stage, "type": type(exc).__name__, "msg": str(exc)}


def _env_truthy(v: str | None) -> bool:
    """Mirror the rule in opencode/.../profile/profile.ts: unset/empty/"0"/"false"
    (case-insensitive) is off; anything else is on."""
    if not v:
        return False
    return v.lower() not in ("0", "false")


async def _run_git(args: list[str]) -> None:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(args)} failed: {stderr.decode().strip()}")


async def _pre_clone(repo: str, base_commit: str, dest: Path,
                     *, reset: bool = False) -> None:
    """git clone <repo> <dest> && checkout <base_commit>. Fail-fast.

    With reset=True, an existing valid checkout is wiped back to
    base_commit (git reset --hard + git clean -fdx) instead of being
    reused as-is. Required for reproducible-workspace mode where the
    same instance_id maps to the same dir across runs -- otherwise
    artifacts from the previous agent loop (modified files, .opencode/
    state, etc.) leak into the next run and divergence cascades.

    With reset=False (legacy), an existing dest is reused untouched."""
    if dest.exists():
        if not reset:
            return
        if (dest / ".git").is_dir():
            # Fast reset path: keep the .git pack, just rewind working tree.
            await _run_git(["git", "-C", str(dest), "reset", "--hard", "--quiet", base_commit])
            # -fdx removes untracked + ignored (e.g. .opencode/ working files,
            # test runner caches). Same final state as a fresh clone+checkout
            # of base_commit, without the network round trip.
            await _run_git(["git", "-C", str(dest), "clean", "-fdxq"])
            return
        # Existed but not a usable git repo (interrupted clone, leftover
        # junk). Nuke and re-clone -- the alternative (silently using a
        # garbage dir) would just bury the failure deeper in the agent loop.
        import shutil
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{repo}.git"
    await _run_git(["git", "clone", "--quiet", url, str(dest)])
    await _run_git(["git", "-C", str(dest), "checkout", "--quiet", base_commit])


async def _run_one(
    client: OpenCodeClient,
    sample: dict[str, Any],
    arrival_offset_s: float,
    workspace_root: Path,
    sem: asyncio.Semaphore,
    task_timeout_s: float | None = None,
    reset_workspace: bool = False,
) -> TaskRecord:
    instance_id = sample["instance_id"]
    # reset_workspace=True: drop the uuid suffix so the same instance_id
    # always lands at the same absolute path. Combined with _pre_clone's
    # reset, this makes opencode's system prompt -- which embeds the
    # working directory -- byte-identical across reruns of the same
    # sample (key prerequisite for agent-loop reproducibility).
    if reset_workspace:
        directory = f"session-{instance_id}"
    else:
        directory = f"session-{instance_id}-{uuid.uuid4().hex[:8]}"
    dest = workspace_root / directory
    # OpenCode's InstanceMiddleware resolves `?directory=` via `path.resolve()`
    # against its own CWD (opencode/packages/opencode/src/server/routes/instance/middleware.ts).
    # We pre-clone to <workspace_root>/<directory>, so we MUST send the absolute
    # path or OpenCode will look in opencode/<directory> instead.
    abs_dir = str(dest)

    async with sem:
        # Stage 1: clone.
        try:
            await _pre_clone(sample["repo"], sample["base_commit"], dest,
                             reset=reset_workspace)
        except Exception as exc:
            return TaskRecord(
                instance_id=instance_id,
                session_id=None,
                directory=directory,
                arrival_offset_s=arrival_offset_s,
                rtt_s=None,
                success=False,
                error=_err("clone", exc),
                messages=[],
            )

        # Stage 2: create session.
        try:
            session_id = await client.create_session(directory=abs_dir)
        except Exception as exc:
            return TaskRecord(
                instance_id=instance_id,
                session_id=None,
                directory=directory,
                arrival_offset_s=arrival_offset_s,
                rtt_s=None,
                success=False,
                error=_err("session", exc),
                messages=[],
            )

        # Stage 3: send message (blocks for the whole agent loop). Time it.
        # Wrap ONLY this call in asyncio.wait_for: stages 1/2/4 are fast and
        # the hang we observe is always inside the agent loop (model stuck
        # in a tool cycle, vLLM KV-transfer stall, or OpenCode SSE never
        # closing — opencode.py uses read=None on purpose to allow long
        # legit runs). task_timeout_s=None disables the cap.
        prompt = swebench.render_prompt(sample)
        t0 = time.monotonic()
        try:
            send_coro = client.send_message(session_id, prompt, directory=abs_dir)
            if task_timeout_s is not None:
                await asyncio.wait_for(send_coro, timeout=task_timeout_s)
            else:
                await send_coro
        except asyncio.TimeoutError as exc:
            return TaskRecord(
                instance_id=instance_id,
                session_id=session_id,
                directory=directory,
                arrival_offset_s=arrival_offset_s,
                rtt_s=time.monotonic() - t0,
                success=False,
                error={
                    "stage": "timeout",
                    "type": "TimeoutError",
                    "msg": f"send_message exceeded task_timeout_s={task_timeout_s}",
                },
                messages=[],
            )
        except Exception as exc:
            return TaskRecord(
                instance_id=instance_id,
                session_id=session_id,
                directory=directory,
                arrival_offset_s=arrival_offset_s,
                rtt_s=time.monotonic() - t0,
                success=False,
                error=_err("message", exc),
                messages=[],
            )
        rtt_s = time.monotonic() - t0

        # Stage 4: list messages. RTT is already valid — failure here keeps success=True.
        try:
            messages = await client.list_messages(session_id, directory=abs_dir)
        except Exception as exc:
            return TaskRecord(
                instance_id=instance_id,
                session_id=session_id,
                directory=directory,
                arrival_offset_s=arrival_offset_s,
                rtt_s=rtt_s,
                success=True,
                error=_err("list", exc),
                messages=[],
            )

        return TaskRecord(
            instance_id=instance_id,
            session_id=session_id,
            directory=directory,
            arrival_offset_s=arrival_offset_s,
            rtt_s=rtt_s,
            success=True,
            error=None,
            messages=messages,
        )


def _summary(records: list[TaskRecord]) -> dict[str, Any]:
    count = len(records)
    if count == 0:
        return {"count": 0, "success_rate": None, "rtt_s": {"p50": None, "p95": None}}
    successes = [r for r in records if r.success]
    success_rate = len(successes) / count
    rtts = [r.rtt_s for r in successes if r.rtt_s is not None]
    if not rtts:
        return {"count": count, "success_rate": success_rate, "rtt_s": {"p50": None, "p95": None}}
    if len(rtts) == 1:
        return {"count": count, "success_rate": success_rate, "rtt_s": {"p50": rtts[0], "p95": rtts[0]}}
    qs = statistics.quantiles(rtts, n=100, method="inclusive")
    # quantiles(n=100) returns 99 cut points; index i is the (i+1)th percentile.
    return {
        "count": count,
        "success_rate": success_rate,
        "rtt_s": {"p50": qs[49], "p95": qs[94]},
    }


async def run(
    cfg: TestbedCfg,
    *,
    split: str,
    num_samples: int,
    qps: float,
    seed: int,
    max_in_flight: int,
    out_dir: Path,
    router_label: str,
    task_timeout_s: float | None = None,
    reset_workspace: bool = False,
    sequential: bool = False,
) -> None:
    """Drive `num_samples` SWE-bench tasks.

    Default mode: Poisson arrivals at `qps`, bounded concurrency at
    `max_in_flight`. arrival_offsets are computed up front and tasks fire
    at their offsets via `poisson.arrivals` -- so the system experiences
    realistic burst patterns and queueing behavior.

    sequential=True: ignore qps + max_in_flight; run tasks strictly
    back-to-back (task N+1 starts the moment task N's TaskRecord lands).
    Guarantees exactly one request in flight at all times -- no
    concurrent batching, no scheduler-induced reordering, no router
    cross-talk. Trades realism for reproducibility; pair with
    --reset-workspace for byte-stable workspace state. `arrival_offset_s`
    in TaskRecord becomes the elapsed wall-clock from run start at the
    moment that task started (i.e. the cumulative RTT of prior tasks).
    """
    samples = swebench.load_samples(split, seed, num_samples)

    out_dir.mkdir(parents=True, exist_ok=True)
    workspace_root = Path(cfg.workspace_root)
    workspace_root.mkdir(parents=True, exist_ok=True)

    # OpenCode profiling state is ENV-gated and resolved outside testbed.yaml
    # (deploy/testbed.sh up_opencode injects OPENCODE_PROFILE_DIR into the
    # OpenCode child env). Capture whatever is visible on the runner-side
    # process env at snapshot time so trace reproducibility includes the
    # profile knobs -- otherwise resolved_snapshot(cfg) is blind to them.
    opencode_profile = {
        "enabled": _env_truthy(os.environ.get("OPENCODE_PROFILE")),
        "raw": os.environ.get("OPENCODE_PROFILE"),
        "dir": os.environ.get("OPENCODE_PROFILE_DIR"),
        "messages": os.environ.get("OPENCODE_PROFILE_MESSAGES"),
    }

    invocation = {
        "split": split,
        "num_samples": num_samples,
        "qps": qps,
        "seed": seed,
        "max_in_flight": max_in_flight,
        "task_timeout_s": task_timeout_s,
        "reset_workspace": reset_workspace,
        "sequential": sequential,
        "router": router_label,
        "model": cfg.model.model_dump(mode="json"),
        "config": resolved_snapshot(cfg),
        "opencode_profile": opencode_profile,
    }
    (out_dir / "config.json").write_text(json.dumps(invocation, indent=2) + "\n")

    # Semaphore exists even in sequential mode (passed to _run_one as a
    # required positional) but is degenerate at size=1.
    sem = asyncio.Semaphore(1 if sequential else max_in_flight)
    password = os.environ.get("OPENCODE_SERVER_PASSWORD") or None

    records: list[TaskRecord] = []
    trace_path = out_dir / "trace.jsonl"

    async with OpenCodeClient(cfg.opencode, password=password) as client:
        with trace_path.open("w") as trace_fh:
            start = time.monotonic()
            if sequential:
                # Strictly serial: await each task fully before kicking off
                # the next. No Poisson distribution, no concurrent tasks.
                for i, sample in enumerate(samples):
                    elapsed = time.monotonic() - start
                    rec = await _run_one(
                        client,
                        sample,
                        elapsed,
                        workspace_root,
                        sem,
                        task_timeout_s=task_timeout_s,
                        reset_workspace=reset_workspace,
                    )
                    trace_fh.write(rec.to_jsonl())
                    trace_fh.flush()
                    records.append(rec)
            else:
                offsets = poisson.arrival_offsets(qps, num_samples, seed)
                tasks: list[asyncio.Task[TaskRecord]] = []
                async for i in poisson.arrivals(offsets, start_monotonic=start):
                    tasks.append(
                        asyncio.create_task(
                            _run_one(
                                client,
                                samples[i],
                                offsets[i],
                                workspace_root,
                                sem,
                                task_timeout_s=task_timeout_s,
                                reset_workspace=reset_workspace,
                            )
                        )
                    )

                for fut in asyncio.as_completed(tasks):
                    rec = await fut
                    trace_fh.write(rec.to_jsonl())
                    trace_fh.flush()
                    records.append(rec)

    (out_dir / "summary.json").write_text(json.dumps(_summary(records), indent=2) + "\n")
