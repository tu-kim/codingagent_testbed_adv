"""Click entrypoint for the testbed: `run` and `smoke`."""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from pathlib import Path

import click

from . import config as config_mod
from . import runner as runner_mod
from .opencode import OpenCodeClient


def _resolve_split(workload: str, split: str | None) -> str:
    """Fill in the workload's default split and reject splits the workload
    doesn't have (click.Choice can't express per-workload choices)."""
    wl = runner_mod.get_workload(workload)
    if split is None:
        return wl.default_split
    if split not in wl.splits:
        raise click.BadParameter(
            f"workload {workload!r} has splits {list(wl.splits)}, not {split!r}",
            param_hint="--split",
        )
    return split


_WORKLOAD_OPT = click.option(
    "--workload", default="swebench", show_default=True,
    type=click.Choice(sorted(runner_mod.WORKLOADS)),
    help="Benchmark driving the workload: swebench (git checkout + issue "
         "fix) or apps (materialized PROBLEM.md + solution.py, no git).",
)

_SPLIT_OPT = click.option(
    "--split", default=None,
    help="Sample split. Default: the workload's default (swebench: lite; "
         "apps: test). swebench: lite|verified|full. apps: train|test plus "
         "the difficulty pseudo-splits introductory|interview|competition "
         "(= test filtered to that difficulty).",
)


@click.group()
def main() -> None:
    """testbed CLI."""


@main.command("run")
@_WORKLOAD_OPT
@_SPLIT_OPT
@click.option("--num-samples", default=10, show_default=True, type=int)
@click.option("--qps", default=0.5, show_default=True, type=float)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--max-in-flight", default=16, show_default=True, type=int)
@click.option(
    "--task-timeout-s",
    default=300.0,
    show_default=True,
    type=float,
    help=(
        "Per-task wall-clock cap on the agent loop (only stage 3, "
        "POST /session/:id/message). On expiry the TaskRecord lands with "
        "error.stage='timeout'. Pass <=0 to disable."
    ),
)
@click.option("--router", default="", show_default=True, help="Recorded into config.json only.")
@click.option(
    "--reset-workspace",
    is_flag=True,
    default=False,
    help=(
        "Use a deterministic per-instance_id workspace dir "
        "(session-<instance_id>, no uuid suffix) and wipe it back to "
        "base_commit (git reset --hard + git clean -fdx) before each "
        "task. Makes the absolute workspace path stable across reruns "
        "of the same sample -- required for reproducible system prompts "
        "in opencode (which embeds the cwd). Default off preserves the "
        "legacy UUID-suffix + accumulate behavior. For full agent-loop "
        "reproducibility pair with --sequential."
    ),
)
@click.option(
    "--sequential",
    is_flag=True,
    default=False,
    help=(
        "Strictly back-to-back execution: task N+1 starts the moment "
        "task N finishes. Ignores --qps and --max-in-flight. Guarantees "
        "exactly one request in flight at all times -- no concurrent "
        "batching, no router cross-talk, no scheduler reordering. "
        "Use for reproducibility comparisons; pair with --reset-workspace. "
        "Default off preserves the Poisson-arrival workload model."
    ),
)
@click.option(
    "--pre-clone-workspaces/--no-pre-clone-workspaces",
    default=True,
    show_default=True,
    help=(
        "Clone EVERY task workspace before the workload starts, so zero "
        "clones happen mid-run -- on a flaky network this removes clone "
        "failures entirely. Workspaces that still fail to pre-clone are "
        "retried at their task's arrival. --no-pre-clone-workspaces "
        "reverts to cloning each workspace at task arrival time."
    ),
)
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--config", "config_path", default=None, type=click.Path(dir_okay=False, exists=True, path_type=Path))
def run_cmd(
    workload: str,
    split: str | None,
    num_samples: int,
    qps: float,
    seed: int,
    max_in_flight: int,
    task_timeout_s: float,
    router: str,
    reset_workspace: bool,
    sequential: bool,
    pre_clone_workspaces: bool,
    out: Path,
    config_path: Path | None,
) -> None:
    """Run a Poisson workload through OpenCode."""
    split = _resolve_split(workload, split)
    cfg = config_mod.load(config_path)
    asyncio.run(
        runner_mod.run(
            cfg,
            workload=workload,
            split=split,
            num_samples=num_samples,
            qps=qps,
            seed=seed,
            max_in_flight=max_in_flight,
            out_dir=out,
            router_label=router,
            task_timeout_s=task_timeout_s if task_timeout_s > 0 else None,
            reset_workspace=reset_workspace,
            sequential=sequential,
            pre_clone_workspaces=pre_clone_workspaces,
        )
    )


@main.command("pre-clone")
@_WORKLOAD_OPT
@_SPLIT_OPT
@click.option("--num-samples", default=10, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--reset-workspace", is_flag=True, default=False,
              help="Must match the upcoming run's --reset-workspace "
                   "(controls workspace dir naming; a mismatched manifest "
                   "is ignored by `run`).")
@click.option("--concurrency", default=8, show_default=True, type=int,
              help="Concurrent workspace clones.")
@click.option("--config", "config_path", default=None, type=click.Path(dir_okay=False, exists=True, path_type=Path))
def pre_clone_cmd(
    workload: str,
    split: str | None,
    num_samples: int,
    seed: int,
    reset_workspace: bool,
    concurrency: int,
    config_path: Path | None,
) -> None:
    """Clone ALL task workspaces for an upcoming run, before starting it.

    Conservative two-step flow for flaky networks: do every git operation
    here, verify everything is ready, THEN start the workload (which does
    zero cloning). Writes a manifest that `run` consumes to reuse the
    exact same directories:

        python -m testbed pre-clone --split lite --num-samples 300 --seed 42 \\
          && python -m testbed run --split lite --num-samples 300 --seed 42 --out results/run1

    Use the SAME --split/--num-samples/--seed (sample selection is
    deterministic) and the same --reset-workspace as the run.

    Resumable: re-running retries only the workspaces that failed.
    Exits 1 if any workspace could not be prepared.
    """
    split = _resolve_split(workload, split)
    cfg = config_mod.load(config_path)
    failures = asyncio.run(
        runner_mod.pre_clone_run(
            cfg,
            workload=workload,
            split=split,
            num_samples=num_samples,
            seed=seed,
            reset_workspace=reset_workspace,
            concurrency=concurrency,
        )
    )
    ready = num_samples - len(failures)
    click.echo(f"workspaces ready: {ready}/{num_samples}")
    if failures:
        click.echo("FAILED workspaces (re-run pre-clone to retry just these):",
                   err=True)
        for iid, msg in sorted(failures.items()):
            click.echo(f"  {iid}: {msg}", err=True)
        raise SystemExit(1)


@main.command("smoke")
@_WORKLOAD_OPT
@_SPLIT_OPT
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--task-timeout-s", default=300.0, show_default=True, type=float,
              help="Per-task wall-clock cap; <=0 disables.")
@click.option("--reset-workspace", is_flag=True, default=False,
              help="Same semantics as `run --reset-workspace` -- deterministic "
                   "dir name + reset to base_commit before the task.")
@click.option("--config", "config_path", default=None, type=click.Path(dir_okay=False, exists=True, path_type=Path))
def smoke_cmd(workload: str, split: str | None, seed: int, task_timeout_s: float,
              reset_workspace: bool, config_path: Path | None) -> None:
    """Run a single end-to-end task and print the resulting TaskRecord."""
    split = _resolve_split(workload, split)
    wl = runner_mod.get_workload(workload)
    cfg = config_mod.load(config_path)
    sample = wl.load_samples(split, seed, 1)[0]
    # Same normalization as runner.run/pre_clone_run: a relative
    # workspace_root must not anchor on the process CWD.
    workspace_root = Path(cfg.workspace_root).expanduser().resolve()
    workspace_root.mkdir(parents=True, exist_ok=True)
    password = os.environ.get("OPENCODE_SERVER_PASSWORD") or None
    timeout = task_timeout_s if task_timeout_s > 0 else None

    async def _go() -> None:
        sem = asyncio.Semaphore(1)
        async with OpenCodeClient(cfg.opencode, password=password) as client:
            rec = await runner_mod._run_one(
                client, sample, 0.0, workspace_root, sem,
                task_timeout_s=timeout,
                reset_workspace=reset_workspace,
                workload=wl,
            )
            click.echo(json.dumps(asdict(rec), indent=2))

    asyncio.run(_go())


if __name__ == "__main__":  # pragma: no cover
    main()
