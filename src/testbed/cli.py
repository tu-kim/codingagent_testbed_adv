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
from . import swebench
from .opencode import OpenCodeClient


@click.group()
def main() -> None:
    """testbed CLI."""


@main.command("run")
@click.option("--split", default="lite", show_default=True, type=click.Choice(["lite", "verified", "full"]))
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
    "--repo-cache/--no-repo-cache",
    default=True,
    show_default=True,
    help=(
        "Pre-clone each unique SWE-bench repo ONCE into a local cache "
        "(<workspace_root>/.repo-cache) before the run, then make each "
        "task clone locally from the cache instead of hitting GitHub. "
        "Removes the rate-limit / bandwidth clone failures you get when "
        "many tasks clone the same repos concurrently. --no-repo-cache "
        "reverts to a direct network clone per task."
    ),
)
@click.option("--repo-cache-dir", default=None,
              type=click.Path(file_okay=False, path_type=Path),
              help="Override the repo cache location (default "
                   "<workspace_root>/.repo-cache).")
@click.option("--out", required=True, type=click.Path(file_okay=False, path_type=Path))
@click.option("--config", "config_path", default=None, type=click.Path(dir_okay=False, exists=True, path_type=Path))
def run_cmd(
    split: str,
    num_samples: int,
    qps: float,
    seed: int,
    max_in_flight: int,
    task_timeout_s: float,
    router: str,
    reset_workspace: bool,
    sequential: bool,
    repo_cache: bool,
    repo_cache_dir: Path | None,
    out: Path,
    config_path: Path | None,
) -> None:
    """Run a Poisson workload through OpenCode."""
    cfg = config_mod.load(config_path)
    asyncio.run(
        runner_mod.run(
            cfg,
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
            repo_cache=repo_cache,
            repo_cache_dir=str(repo_cache_dir) if repo_cache_dir else None,
        )
    )


@main.command("smoke")
@click.option("--split", default="lite", show_default=True, type=click.Choice(["lite", "verified", "full"]))
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--task-timeout-s", default=300.0, show_default=True, type=float,
              help="Per-task wall-clock cap; <=0 disables.")
@click.option("--reset-workspace", is_flag=True, default=False,
              help="Same semantics as `run --reset-workspace` -- deterministic "
                   "dir name + reset to base_commit before the task.")
@click.option("--config", "config_path", default=None, type=click.Path(dir_okay=False, exists=True, path_type=Path))
def smoke_cmd(split: str, seed: int, task_timeout_s: float,
              reset_workspace: bool, config_path: Path | None) -> None:
    """Run a single end-to-end task and print the resulting TaskRecord."""
    cfg = config_mod.load(config_path)
    sample = swebench.load_samples(split, seed, 1)[0]
    workspace_root = Path(cfg.workspace_root)
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
            )
            click.echo(json.dumps(asdict(rec), indent=2))

    asyncio.run(_go())


if __name__ == "__main__":  # pragma: no cover
    main()
