"""Poisson workload driver: pre-clone, drive OpenCode, write trace.jsonl + summary.json."""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import sys
import time
import uuid
from collections import defaultdict
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
    Factored out so tests can point the cache at a local file:// repo."""
    return f"https://github.com/{repo}.git"


def _repo_cache_path(cache_dir: Path, repo: str) -> Path:
    return cache_dir / repo.replace("/", "__")


async def _ensure_repo_cached(repo: str, base_commits: set[str],
                              cache_dir: Path) -> Path | None:
    """Ensure a full clone of `repo` exists in the cache and contains every
    needed base_commit. Returns the cache path, or None on failure (caller
    then falls back to a direct per-task clone). Network ops are retried."""
    cache = _repo_cache_path(cache_dir, repo)
    try:
        if not (cache / ".git").is_dir():
            if cache.exists():
                import shutil
                shutil.rmtree(cache)
            cache.parent.mkdir(parents=True, exist_ok=True)
            await _run_git_retry(["git", "clone", "--quiet", _repo_url(repo), str(cache)])
        missing = []
        for c in base_commits:
            try:
                await _run_git(["git", "-C", str(cache), "cat-file", "-e", f"{c}^{{commit}}"])
            except RuntimeError:
                missing.append(c)
        if missing:
            # A shallow/old clone may lack some base_commits; pull full history.
            await _run_git_retry(["git", "-C", str(cache), "fetch", "--quiet", "origin"])
        return cache
    except RuntimeError:
        return None


async def warm_repo_cache(samples: list[dict[str, Any]], cache_dir: Path,
                          *, concurrency: int = 4) -> dict[str, Path]:
    """Pre-clone every UNIQUE repo once into the cache before the workload
    starts. SWE-bench draws many samples from the same repos, so this turns
    N per-task network clones into U (unique-repo) clones and removes the
    rate-limit / bandwidth pressure that makes mid-run clones flaky.
    Returns {repo: cache_path} for repos that cached successfully; repos
    absent from the map fall back to a direct per-task clone."""
    by_repo: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        by_repo[s["repo"]].add(s["base_commit"])
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, Path] = {}

    async def _one(repo: str, commits: set[str]) -> None:
        async with sem:
            path = await _ensure_repo_cached(repo, commits, cache_dir)
            if path is not None:
                results[repo] = path

    await asyncio.gather(*(_one(r, c) for r, c in by_repo.items()))
    return results


async def _pre_clone(repo: str, base_commit: str, dest: Path,
                     *, reset: bool = False, cache_dir: Path | None = None) -> None:
    """Prepare <dest> at <repo>@<base_commit>. Fail-fast.

    Clone source: if `cache_dir` holds a cached clone of `repo`, copy from
    it LOCALLY (git clone --local -- no network, immune to rate limits);
    otherwise fall back to a direct (retried) network clone.

    With reset=True, an existing valid checkout is wiped back to
    base_commit (git reset --hard + git clean -fdx) instead of re-cloned.
    Required for reproducible-workspace mode (same instance_id → same dir
    across runs) -- otherwise prior agent-loop artifacts leak in.
    With reset=False (legacy), an existing dest is reused untouched."""
    if dest.exists():
        if not reset:
            return
        if (dest / ".git").is_dir():
            await _run_git(["git", "-C", str(dest), "reset", "--hard", "--quiet", base_commit])
            await _run_git(["git", "-C", str(dest), "clean", "-fdxq"])
            return
        # Existed but not a usable git repo (interrupted clone). Re-create.
        import shutil
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cache = _repo_cache_path(cache_dir, repo) if cache_dir is not None else None
    if cache is not None and (cache / ".git").is_dir():
        # Local clone from the warmed cache -- no network. Hardlinks the
        # (immutable) object store, so it's fast and cheap on disk.
        await _run_git(["git", "clone", "--quiet", "--local", str(cache), str(dest)])
        await _run_git(["git", "-C", str(dest), "checkout", "--quiet", base_commit])
        return
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


async def _run_one(
    client: OpenCodeClient,
    sample: dict[str, Any],
    arrival_offset_s: float,
    workspace_root: Path,
    sem: asyncio.Semaphore,
    task_timeout_s: float | None = None,
    reset_workspace: bool = False,
    repo_cache_dir: Path | None = None,
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
                             reset=reset_workspace, cache_dir=repo_cache_dir)
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
    repo_cache: bool = True,
    repo_cache_dir: str | None = None,
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

    # Pre-clone each unique repo once into a local cache BEFORE firing any
    # task. Per-task clones then copy from the cache locally (no network),
    # which removes the GitHub rate-limiting / bandwidth pressure that
    # makes mid-run clones fail when many tasks clone the same repos.
    cache_dir: Path | None = None
    if repo_cache:
        cache_dir = (Path(repo_cache_dir) if repo_cache_dir
                     else workspace_root / ".repo-cache")
        n_repos = len({s["repo"] for s in samples})
        print(f"warming repo cache: {n_repos} unique repos → {cache_dir}",
              file=sys.stderr, flush=True)
        cached = await warm_repo_cache(samples, cache_dir)
        if len(cached) < n_repos:
            print(f"  {n_repos - len(cached)} repo(s) failed to cache -- those "
                  f"tasks fall back to direct clone", file=sys.stderr, flush=True)
        print(f"repo cache ready: {len(cached)}/{n_repos} repos", file=sys.stderr, flush=True)

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
        "repo_cache": repo_cache,
        "repo_cache_dir": str(cache_dir) if cache_dir else None,
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
                        repo_cache_dir=cache_dir,
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
                                repo_cache_dir=cache_dir,
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
