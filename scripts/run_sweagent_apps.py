#!/usr/bin/env python3
"""Drive APPS tasks through the SWE-agent scaffold against the SAME Dynamo
backend the OpenCode runner uses -- the scaffold-comparison counterpart of
`python -m testbed run --workload apps`.

Why this exists: tool-time share is a joint property of (scaffold
granularity x execution environment x model/serving speed), not of the
benchmark. OpenCode on APPS measures ~1% tool share on this testbed while
SWE-agent-style scaffolds are reported around 35% in the literature. This
runner holds the benchmark samples (identical deterministic selection via
testbed.apps.load_samples), the model, and the serving stack constant, and
swaps ONLY the scaffold -- so the residual difference is attributable to
the scaffold + execution environment. `--deployment docker` (default) is
the upstream-supported path; `local` was meant to split the container
overhead out of the scaffold factor, but upstream copies the repo to the
DEPLOYMENT ROOT `/{repo_name}` (fine inside a container, PermissionError
13 on a host for non-root users; the configurable-base-dir PR #1132 was
closed unmerged with "use mini-swe-agent for fully local runs") -- so the
scaffold-only arm needs a root-writable / or a different minimal scaffold.

Flow per task (strictly sequential -- matches the opencode comparison mode):
  1. materialize the APPS workspace (PROBLEM.md + solution.py via
     apps.prepare_workspace, reset semantics) and `git init` + commit it
     (SWE-agent environments operate on a git repo; the commit is the
     base state its patch is diffed against).
  2. invoke `sweagent run` as a subprocess:
       model  -> litellm openai-compatible: --agent.model.name
                 openai/<served_name> + --agent.model.api_base <dynamo>/v1
       env    -> --env.repo.path <workspace> +
                 --env.deployment.type {local|docker}
       task   -> --problem_statement.text <apps.render_prompt(sample)>
                 (+ --problem_statement.id <instance_id> so output dirs
                 are stable)
     Wall-clock the subprocess; record task_start/end unix timestamps so
     the Dynamo frontend log can be joined per-task afterwards.
  3. locate the produced patch (<traj_dir>/<id>/<id>.patch) and
     `git apply` it onto the pristine workspace -- the workspace then
     holds SWE-agent's final solution.py, so scripts/evaluate_apps.py
     works on this run directory EXACTLY as it does for opencode runs.
  4. append a TaskRecord-shaped line to trace.jsonl.

Version pin / drift note: written against the `sweagent` 1.x CLI
(install FROM SOURCE: `pip install "git+https://github.com/SWE-agent/SWE-agent.git"`
-- the PyPI name `sweagent` is an unrelated squatted 0.0.1 package whose
`togetherunidiff` dep doesn't even resolve; verify with `sweagent run --help`).
ALL sweagent
flag names live in build_sweagent_cmd() -- if the CLI surface drifts, fix
that one function. Use --dry-run to print the exact commands without
executing anything (cheap flag-surface validation on the GPU host).

Outputs under --out:
  config.json     workload="apps", agent="swe-agent", sample-selection
                  tuple, sweagent knobs, workspace_root snapshot --
                  evaluate_apps.py-compatible.
  trace.jsonl     one record per task: instance_id, directory, rtt_s,
                  success, error, task_start_unix_s, task_end_unix_s,
                  traj_dir, patch_applied.
  trajs/<id>/     raw SWE-agent output (trajectory .traj, patch, logs).

Usage:
  scripts/run_sweagent_apps.py --split competition --num-samples 20 \
      --seed 42 --out results/sweagent-apps1 [--deployment local] \
      [--max-steps 50] [--task-timeout-s 1800] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from testbed import apps  # noqa: E402
from testbed import config as config_mod  # noqa: E402


# ---------- workspace ----------


def _git(ws: Path, *args: str) -> None:
    """Run git in <ws> with a pinned identity (no dependency on host
    git config; commits must succeed on a bare CI-like host)."""
    cmd = ["git", "-C", str(ws),
           "-c", "user.email=testbed@local", "-c", "user.name=testbed",
           *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {proc.stderr.strip()}")


def prepare_git_workspace(sample: dict[str, Any], ws: Path) -> None:
    """Materialize the APPS task files (same two-file layout the opencode
    workload uses) and turn the directory into a one-commit git repo.
    Always reset=True: each SWE-agent task starts from the pristine base
    commit its patch will be diffed against."""
    import asyncio
    asyncio.run(apps.prepare_workspace(sample, ws, reset=True))
    _git(ws, "init", "--quiet")
    _git(ws, "add", "-A")
    _git(ws, "commit", "--quiet", "-m", "apps task base")


# ---------- sweagent invocation ----------


def build_sweagent_cmd(*, model_name: str, api_base: str, api_key: str,
                       ws: Path, prompt: str, instance_id: str,
                       traj_root: Path, deployment: str, max_steps: int,
                       extra_args: list[str]) -> list[str]:
    """The ONE place that knows sweagent 1.x CLI flag names. If
    `sweagent run --help` on the host disagrees, fix it here.

    Cost limits are zeroed (litellm cannot price a local model and
    sweagent aborts on unpriced models otherwise); the step budget is
    enforced via per_instance_call_limit instead."""
    return [
        "sweagent", "run",
        "--agent.model.name", f"openai/{model_name}",
        "--agent.model.api_base", api_base,
        "--agent.model.api_key", api_key,
        "--agent.model.per_instance_cost_limit", "0",
        "--agent.model.total_cost_limit", "0",
        "--agent.model.per_instance_call_limit", str(max_steps),
        "--env.repo.path", str(ws),
        "--env.deployment.type", deployment,
        "--problem_statement.text", prompt,
        "--problem_statement.id", instance_id,
        "--output_dir", str(traj_root),
        *extra_args,
    ]


def find_patch(traj_root: Path, instance_id: str) -> Path | None:
    """sweagent writes <output_dir>/<id>/<id>.patch on submit. Fall back
    to a glob in case the layout shifts between minor versions."""
    exact = traj_root / instance_id / f"{instance_id}.patch"
    if exact.is_file():
        return exact
    hits = sorted(traj_root.glob(f"**/{instance_id}*.patch"))
    return hits[0] if hits else None


def apply_patch(ws: Path, patch: Path) -> bool:
    # resolve(): `git -C <ws>` chdirs BEFORE interpreting its arguments, so a
    # relative patch path (e.g. from a relative --out) would be looked up
    # inside the workspace and fail with "can't open patch".
    proc = subprocess.run(
        ["git", "-C", str(ws), "apply", "--whitespace=nowarn",
         str(Path(patch).resolve())],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"  git apply failed for {patch}: {proc.stderr.strip()}",
              file=sys.stderr)
        return False
    return True


# ---------- main ----------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--split", default="test", choices=sorted(apps.SPLITS))
    ap.add_argument("--num-samples", default=10, type=int)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workspace-root", default=None, type=Path,
                    help="Default: <testbed.yaml workspace_root>/sweagent")
    ap.add_argument("--api-base", default=None,
                    help="OpenAI-compatible base URL. Default: "
                         "http://<dynamo.host>:<dynamo.port>/v1 from testbed.yaml")
    ap.add_argument("--model", default=None,
                    help="Served model name. Default: model.served_name from testbed.yaml")
    ap.add_argument("--api-key", default="dummy",
                    help="Dynamo ignores it; litellm requires something.")
    ap.add_argument("--deployment", default="docker", choices=["docker", "local"],
                    help="SWE-agent execution environment. `docker` (default) "
                         "is the upstream-supported path and matches the "
                         "container round-trip the SWE-agent literature "
                         "numbers include. `local` would isolate the pure "
                         "scaffold factor BUT upstream hardcodes copying the "
                         "repo to the deployment root `/{repo_name}` -- on a "
                         "host that means writing to / (PermissionError 13 "
                         "for non-root; the repo_base_dir fix, upstream PR "
                         "#1132, was closed unmerged). Only use `local` if "
                         "your / is writable; upstream recommends "
                         "mini-swe-agent for fully-local runs instead.")
    ap.add_argument("--max-steps", default=50, type=int,
                    help="per_instance_call_limit (agent step budget).")
    ap.add_argument("--task-timeout-s", default=1800.0, type=float,
                    help="Wall-clock cap per sweagent subprocess; <=0 disables.")
    ap.add_argument("--sweagent-extra-arg", action="append", default=[],
                    dest="extra_args", metavar="ARG",
                    help="Extra raw argv appended to `sweagent run` "
                         "(repeatable) -- escape hatch for CLI drift.")
    ap.add_argument("--config", dest="config_path", default=None, type=Path,
                    help="testbed.yaml path (default: deploy/testbed.yaml)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the sweagent command per task and exit "
                         "without preparing workspaces or executing.")
    args = ap.parse_args(argv)

    cfg = config_mod.load(args.config_path)
    api_base = args.api_base or f"http://{cfg.dynamo.host}:{cfg.dynamo.port}/v1"
    model_name = args.model or cfg.model.served_name
    workspace_root = (args.workspace_root
                      or Path(cfg.workspace_root).expanduser().resolve() / "sweagent")
    workspace_root = Path(workspace_root).expanduser().resolve()

    samples = apps.load_samples(args.split, args.seed, args.num_samples)
    # NOTE: no mkdir before the dry-run branch -- --dry-run must be fully
    # side-effect-free (it only prints the would-be commands).
    # resolve(): a relative --out would otherwise flow into trace.jsonl's
    # traj_dir (breaking analyzers run from another CWD) and into git-apply
    # arguments that `git -C <ws>` interprets relative to the WORKSPACE.
    traj_root = (args.out / "trajs").resolve()

    if args.dry_run:
        for s in samples:
            iid = s["instance_id"]
            cmd = build_sweagent_cmd(
                model_name=model_name, api_base=api_base, api_key=args.api_key,
                ws=workspace_root / f"session-{iid}",
                prompt=f"<prompt for {iid}: {len(apps.render_prompt(s))} chars>",
                instance_id=iid, traj_root=traj_root,
                deployment=args.deployment, max_steps=args.max_steps,
                extra_args=args.extra_args,
            )
            print(shlex.join(cmd))
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)
    traj_root.mkdir(parents=True, exist_ok=True)

    invocation = {
        "workload": "apps",
        "agent": "swe-agent",
        "split": args.split,
        "num_samples": args.num_samples,
        "seed": args.seed,
        "sequential": True,
        "sweagent": {
            "deployment": args.deployment,
            "max_steps": args.max_steps,
            "task_timeout_s": args.task_timeout_s,
            "api_base": api_base,
            "model": model_name,
            "extra_args": args.extra_args,
        },
        # evaluate_apps.py reads config.workspace_root from here.
        "config": {"workspace_root": str(workspace_root)},
    }
    (args.out / "config.json").write_text(json.dumps(invocation, indent=2) + "\n")

    trace_path = args.out / "trace.jsonl"
    n_ok = 0
    with trace_path.open("w") as trace_fh:
        run_start = time.monotonic()
        for i, sample in enumerate(samples):
            iid = sample["instance_id"]
            directory = f"session-{iid}"
            ws = workspace_root / directory
            rec: dict[str, Any] = {
                "instance_id": iid,
                "session_id": None,          # no opencode session; schema parity
                "directory": directory,
                "arrival_offset_s": time.monotonic() - run_start,
                "rtt_s": None,
                "success": False,
                "error": None,
                "traj_dir": str(traj_root / iid),
                "patch_applied": False,
            }
            try:
                prepare_git_workspace(sample, ws)
            except Exception as exc:  # noqa: BLE001 -- per-task fail-fast
                rec["error"] = {"stage": "clone", "type": type(exc).__name__,
                                "msg": str(exc)}
                trace_fh.write(json.dumps(rec) + "\n"); trace_fh.flush()
                print(f"[{i+1:>3}/{len(samples)}] {iid} FAIL:clone",
                      file=sys.stderr, flush=True)
                continue

            cmd = build_sweagent_cmd(
                model_name=model_name, api_base=api_base, api_key=args.api_key,
                ws=ws, prompt=apps.render_prompt(sample), instance_id=iid,
                traj_root=traj_root, deployment=args.deployment,
                max_steps=args.max_steps, extra_args=args.extra_args,
            )
            timeout = args.task_timeout_s if args.task_timeout_s > 0 else None
            rec["task_start_unix_s"] = time.time()
            t0 = time.monotonic()
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True,
                                      timeout=timeout)
                rec["rtt_s"] = time.monotonic() - t0
                rec["task_end_unix_s"] = time.time()
                if proc.returncode != 0:
                    rec["error"] = {"stage": "message", "type": "NonZeroExit",
                                    "msg": (proc.stderr or proc.stdout)[-2000:]}
                else:
                    rec["success"] = True
            except subprocess.TimeoutExpired:
                rec["rtt_s"] = time.monotonic() - t0
                rec["task_end_unix_s"] = time.time()
                rec["error"] = {"stage": "timeout", "type": "TimeoutError",
                                "msg": f"sweagent exceeded {args.task_timeout_s}s"}

            # Patch extraction is best-effort even on timeout/failure --
            # a partial patch still lets evaluate_apps score what exists.
            patch = find_patch(traj_root, iid)
            if patch is not None:
                rec["patch_applied"] = apply_patch(ws, patch)

            if rec["success"]:
                n_ok += 1
            trace_fh.write(json.dumps(rec) + "\n"); trace_fh.flush()
            status = "ok" if rec["success"] else \
                "FAIL:" + str((rec["error"] or {}).get("stage", "?"))
            # rtt_s is always set on paths reaching here today, but guard the
            # format anyway: a future error path that falls through with
            # rtt_s=None must not crash the whole run's progress loop.
            rtt = f"{rec['rtt_s']:.1f}s" if rec["rtt_s"] is not None else "-"
            print(f"[{i+1:>3}/{len(samples)}] {iid:<16} {status:<12} "
                  f"rtt={rtt} patch={rec['patch_applied']}",
                  file=sys.stderr, flush=True)

    print(f"done: {n_ok}/{len(samples)} ok → {args.out}", file=sys.stderr)
    print(f"next: scripts/evaluate_apps.py --run {args.out}   # correctness",
          file=sys.stderr)
    print(f"      scripts/analyze_sweagent_traj.py --run {args.out} "
          f"--frontend logs/frontend.log   # llm/tool/others share",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
