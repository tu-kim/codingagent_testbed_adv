"""Poisson workload driver: pre-clone, drive OpenCode, write trace.jsonl + summary.json."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
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


def _print_progress(rec: TaskRecord, records: list[TaskRecord],
                    total: int, elapsed_s: float) -> None:
    """One line per completed task → stderr (stdout stays clean for any
    downstream capture). Shows running done/total, this task's status +
    RTT, and the cumulative ok/fail tally."""
    done = len(records)
    n_ok = sum(1 for r in records if r.success)
    n_fail = done - n_ok
    if rec.success:
        status = "ok"
    else:
        status = "FAIL:" + str((rec.error or {}).get("stage", "?"))
    rtt = f"{rec.rtt_s:6.1f}s" if rec.rtt_s is not None else "     -  "
    print(
        f"[{done:>3}/{total}] {rec.instance_id:<34.34} {status:<11} "
        f"rtt={rtt}  ok={n_ok} fail={n_fail}  elapsed={elapsed_s:5.0f}s",
        file=sys.stderr,
        flush=True,
    )


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


async def _run_git_retry(args: list[str], *, retries: int = 3,
                         base_delay: float = 2.0) -> None:
    """_run_git with exponential backoff -- for NETWORK git ops (clone /
    fetch) which fail transiently under GitHub rate-limiting. Local ops
    use plain _run_git (a failure there is deterministic, not transient)."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            await _run_git(args)
            return
        except RuntimeError as exc:
            last = exc
            if attempt < retries - 1:
                await asyncio.sleep(base_delay * (2 ** attempt))
    assert last is not None
    raise last


def _repo_url(repo: str) -> str:
    """GitHub HTTPS URL for a SWE-bench `repo` field (e.g. 'django/django').
    Factored out so tests can point clones at a local file:// repo."""
    return f"https://github.com/{repo}.git"


async def _is_valid_repo(dest: Path) -> bool:
    """True iff <dest> is a checkout git itself accepts.

    `(dest / ".git").is_dir()` is NOT enough: an interrupted clone can
    leave a .git directory whose contents are incomplete, and every
    subsequent git command then dies with "fatal: not a git repository".
    Ask git directly instead."""
    try:
        await _run_git(["git", "-C", str(dest), "rev-parse", "--git-dir"])
        return True
    except RuntimeError:
        return False


async def _pre_clone(repo: str, base_commit: str, dest: Path,
                     *, reset: bool = False) -> None:
    """Prepare <dest> at <repo>@<base_commit>. Fail-fast.

    Direct network clone, retried with exponential backoff
    (_run_git_retry) to ride out transient network failures. Idempotent:
    an existing VALID checkout is a no-op (reset=False) so retrying a
    partially failed pre-clone pass only re-clones what's missing. A
    broken checkout (interrupted clone, half-written .git) is detected
    via `git rev-parse --git-dir` -- not just a .git presence check --
    and nuked + re-cloned in BOTH modes.

    With reset=True, an existing valid checkout is wiped back to
    base_commit (git reset --hard + git clean -fdx) instead of re-cloned;
    if even the reset fails (repo valid enough for rev-parse but missing
    objects), it falls through to a full re-clone. Required for
    reproducible-workspace mode (same instance_id → same dir across runs)
    -- otherwise prior agent-loop artifacts leak in.
    With reset=False (legacy), an existing valid dest is reused untouched."""
    if dest.exists():
        if await _is_valid_repo(dest):
            if not reset:
                return
            try:
                await _run_git(["git", "-C", str(dest), "reset", "--hard", "--quiet", base_commit])
                await _run_git(["git", "-C", str(dest), "clean", "-fdxq"])
                return
            except RuntimeError:
                # Valid-looking repo that still can't reset (missing
                # objects, corrupt index). Fall through to re-clone.
                pass
        # Broken / unresettable checkout: nuke and re-clone fresh.
        import shutil
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    await _run_git_retry(["git", "clone", "--quiet", _repo_url(repo), str(dest)])
    await _run_git(["git", "-C", str(dest), "checkout", "--quiet", base_commit])


async def _try_list_partial(client: OpenCodeClient, session_id: str,
                            abs_dir: str, *, timeout_s: float = 30.0) -> list[Any]:
    """Best-effort fetch of the turns completed BEFORE a timeout abort.
    opencode persists each turn as it finishes, so the list endpoint
    returns partial progress even after we cancelled the message call.
    Returns [] on any error / its own timeout so it never re-hangs the
    task or masks the original timeout."""
    try:
        return await asyncio.wait_for(
            client.list_messages(session_id, directory=abs_dir),
            timeout=timeout_s,
        )
    except Exception:
        return []


def _directory_for(instance_id: str, reset_workspace: bool) -> str:
    """Workspace folder NAME for a task (not the OpenCode session id).

    reset_workspace=True: drop the uuid suffix so the same instance_id
    always lands at the same absolute path. Combined with _pre_clone's
    reset, this makes opencode's system prompt -- which embeds the
    working directory -- byte-identical across reruns of the same
    sample (key prerequisite for agent-loop reproducibility)."""
    if reset_workspace:
        return f"session-{instance_id}"
    return f"session-{instance_id}-{uuid.uuid4().hex[:8]}"


async def prepare_workspaces(
    samples: list[dict[str, Any]],
    directories: dict[str, str],
    workspace_root: Path,
    *,
    reset_workspace: bool = False,
    concurrency: int = 8,
) -> dict[str, str]:
    """Clone EVERY task workspace up front, before the workload starts.

    Moves every (retried) network clone BEFORE the first request, so the
    workload phase performs zero clones: `_pre_clone` at task time sees
    the existing checkout and returns immediately (or, with
    reset_workspace=True, just resets it -- still no clone). On a flaky
    network nothing can fail a task at arrival time.

    Returns {instance_id: error_msg} for workspaces that could NOT be
    prepared; those tasks retry the clone at their arrival (legacy path)
    and fail-fast into an error.stage="clone" TaskRecord if it still fails.
    """
    sem = asyncio.Semaphore(concurrency)
    failures: dict[str, str] = {}
    done = 0
    total = len(samples)

    async def _one(sample: dict[str, Any]) -> None:
        nonlocal done
        iid = sample["instance_id"]
        dest = workspace_root / directories[iid]
        async with sem:
            try:
                await _pre_clone(sample["repo"], sample["base_commit"], dest,
                                 reset=reset_workspace)
            except Exception as exc:  # noqa: BLE001 -- collected, not fatal here
                failures[iid] = f"{type(exc).__name__}: {exc}"
            done += 1
            if done % 25 == 0 or done == total:
                print(f"  workspaces: {done}/{total} prepared",
                      file=sys.stderr, flush=True)

    await asyncio.gather(*(_one(s) for s in samples))
    return failures


def workspace_manifest_path(workspace_root: Path, split: str, seed: int,
                            num_samples: int) -> Path:
    """Manifest written by `pre-clone` and consumed (single-use) by `run`.

    Keyed by the deterministic sample-selection tuple so a run only picks
    up workspaces prepared for exactly its sample set."""
    return workspace_root / f".workspaces-{split}-s{seed}-n{num_samples}.json"


async def pre_clone_run(
    cfg: TestbedCfg,
    *,
    split: str,
    num_samples: int,
    seed: int,
    reset_workspace: bool = False,
    concurrency: int = 8,
) -> dict[str, str]:
    """Standalone conservative prepare step (`python -m testbed pre-clone`):
    clone EVERY task workspace, write the manifest.

    Run this before `run` with the SAME (--split, --num-samples, --seed,
    --reset-workspace). `run` then finds the manifest, reuses the exact
    workspace directories, and skips its own clone phase entirely -- the
    workload performs zero git operations over the network.

    Resumable: if a manifest already exists for this (split, seed, n), the
    previously assigned directory names are reused, so re-running after a
    flaky-network partial failure retries ONLY the missing workspaces
    (`_pre_clone` is a no-op on an existing checkout).

    Returns {instance_id: error_msg} for workspaces that failed; empty
    means everything is ready.
    """
    samples = swebench.load_samples(split, seed, num_samples)
    # expanduser+resolve: a relative workspace_root in testbed.yaml would
    # otherwise anchor git -C / OpenCode ?directory= on whatever CWD the
    # process happens to run from ("fatal: not a git repository ...").
    workspace_root = Path(cfg.workspace_root).expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    manifest = workspace_manifest_path(workspace_root, split, seed, num_samples)
    directories: dict[str, str] = {}
    if manifest.exists():
        try:
            prior = json.loads(manifest.read_text())
            if prior.get("reset_workspace") == reset_workspace:
                directories = dict(prior.get("directories", {}))
                print(f"resuming from existing manifest: {manifest}",
                      file=sys.stderr, flush=True)
        except (json.JSONDecodeError, AttributeError):
            pass  # corrupt manifest -- assign fresh below
    for s in samples:
        directories.setdefault(s["instance_id"],
                               _directory_for(s["instance_id"], reset_workspace))

    print(f"pre-cloning {len(samples)} workspaces → {workspace_root}",
          file=sys.stderr, flush=True)
    failures = await prepare_workspaces(
        samples, directories, workspace_root,
        reset_workspace=reset_workspace, concurrency=concurrency,
    )

    manifest.write_text(json.dumps({
        "split": split,
        "seed": seed,
        "num_samples": num_samples,
        "reset_workspace": reset_workspace,
        "directories": directories,
    }, indent=2) + "\n")
    return failures


async def _run_one(
    client: OpenCodeClient,
    sample: dict[str, Any],
    arrival_offset_s: float,
    workspace_root: Path,
    sem: asyncio.Semaphore,
    task_timeout_s: float | None = None,
    reset_workspace: bool = False,
    directory: str | None = None,
) -> TaskRecord:
    instance_id = sample["instance_id"]
    # run() pre-assigns directory names so it can pre-clone all workspaces
    # before the workload starts; smoke (cli.py) passes None and gets the
    # same name _directory_for would have produced inside run().
    if directory is None:
        directory = _directory_for(instance_id, reset_workspace)
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
        except asyncio.TimeoutError:
            rtt_timeout = time.monotonic() - t0
            # Best-effort: pull the turns opencode persisted before the
            # abort so the trace shows how far the agent got (where it
            # stalled), instead of an empty messages list.
            partial = await _try_list_partial(client, session_id, abs_dir)
            return TaskRecord(
                instance_id=instance_id,
                session_id=session_id,
                directory=directory,
                arrival_offset_s=arrival_offset_s,
                rtt_s=rtt_timeout,
                success=False,
                error={
                    "stage": "timeout",
                    "type": "TimeoutError",
                    "msg": f"send_message exceeded task_timeout_s={task_timeout_s}",
                    "partial_messages": len(partial),
                },
                messages=partial,
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
    pre_clone_workspaces: bool = True,
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
    # expanduser+resolve: a relative workspace_root in testbed.yaml would
    # otherwise anchor git -C / OpenCode ?directory= on whatever CWD the
    # process happens to run from ("fatal: not a git repository ...").
    # Must match pre_clone_run's normalization or the manifest lookup and
    # the pre-cloned checkouts land in different places.
    workspace_root = Path(cfg.workspace_root).expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)

    # Preferred path: `python -m testbed pre-clone` already cloned every
    # workspace and left a manifest for this exact (split, seed, n).
    # Consume it (single-use: a later run must NOT silently reuse dirty
    # non-reset workspaces) and skip all cloning here -- the workload then
    # starts with zero git/network work.
    directories: dict[str, str] | None = None
    manifest_used: str | None = None
    if pre_clone_workspaces:
        manifest = workspace_manifest_path(workspace_root, split, seed, num_samples)
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text())
            except json.JSONDecodeError:
                data = None
                print(f"ignoring corrupt workspace manifest: {manifest}",
                      file=sys.stderr, flush=True)
            wanted = {s["instance_id"] for s in samples}
            if data is not None and data.get("reset_workspace") != reset_workspace:
                print(f"ignoring workspace manifest {manifest}: it was built "
                      f"with reset_workspace={data.get('reset_workspace')} but "
                      f"this run uses {reset_workspace}",
                      file=sys.stderr, flush=True)
            elif data is not None and wanted <= set(data.get("directories", {})):
                directories = {iid: data["directories"][iid] for iid in wanted}
                manifest_used = str(manifest)
                manifest.unlink()
                print(f"using pre-cloned workspaces from {manifest} "
                      f"(manifest consumed -- run `pre-clone` again before the "
                      f"next run)", file=sys.stderr, flush=True)
            elif data is not None:
                print(f"ignoring workspace manifest {manifest}: it does not "
                      f"cover all {len(wanted)} sample(s)",
                      file=sys.stderr, flush=True)

    if directories is None:
        # No usable manifest: clone all workspaces inline, before the
        # first request fires. Workspace directory names are assigned UP
        # FRONT (not inside _run_one) so every task workspace can be
        # cloned before the first request fires.
        directories = {s["instance_id"]: _directory_for(s["instance_id"], reset_workspace)
                       for s in samples}

        if pre_clone_workspaces:
            print(f"pre-cloning {len(samples)} workspaces → {workspace_root}",
                  file=sys.stderr, flush=True)
            clone_failures = await prepare_workspaces(
                samples, directories, workspace_root,
                reset_workspace=reset_workspace,
            )
            if clone_failures:
                print(f"  {len(clone_failures)} workspace(s) FAILED to pre-clone "
                      f"(will retry at task start):", file=sys.stderr, flush=True)
                for iid, msg in sorted(clone_failures.items()):
                    print(f"    {iid}: {msg}", file=sys.stderr, flush=True)
            print(f"workspaces ready: {len(samples) - len(clone_failures)}/{len(samples)}",
                  file=sys.stderr, flush=True)

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
        "pre_clone_workspaces": pre_clone_workspaces,
        "workspace_manifest": manifest_used,
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

    total = len(samples)
    mode = "sequential" if sequential else f"poisson qps={qps} max_in_flight={max_in_flight}"
    print(f"testbed run: {total} tasks ({mode}) → {out_dir}",
          file=sys.stderr, flush=True)

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
                        directory=directories[sample["instance_id"]],
                    )
                    trace_fh.write(rec.to_jsonl())
                    trace_fh.flush()
                    records.append(rec)
                    _print_progress(rec, records, total, time.monotonic() - start)
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
                                directory=directories[samples[i]["instance_id"]],
                            )
                        )
                    )

                for fut in asyncio.as_completed(tasks):
                    rec = await fut
                    trace_fh.write(rec.to_jsonl())
                    trace_fh.flush()
                    records.append(rec)
                    _print_progress(rec, records, total, time.monotonic() - start)

    summary = _summary(records)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    n_ok = sum(1 for r in records if r.success)
    print(f"testbed run: done. {summary['count']} tasks, "
          f"{n_ok} ok / {summary['count'] - n_ok} fail, "
          f"elapsed={time.monotonic() - start:.0f}s → {out_dir}",
          file=sys.stderr, flush=True)
