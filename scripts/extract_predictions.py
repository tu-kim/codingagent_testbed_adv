#!/usr/bin/env python3
"""Extract SWE-bench predictions (model_patch) from a finished testbed run.

The runner leaves each task's workspace at <workspace_root>/<directory>
(pre-cloned at base_commit; the agent edited files in place and did NOT
commit). This script turns each workspace into the official SWE-bench
predictions.jsonl format:

    {"instance_id": ..., "model_name_or_path": ..., "model_patch": "<unified diff>"}

so true resolve/fail can be judged with the official evaluation harness
(see scripts/evaluate_predictions.sh). The testbed's trace.jsonl `success`
field is only HTTP-level (the agent loop completed) -- it says nothing
about whether the fix is correct.

Patch extraction per task:
    git -C <ws> add -A                      # stage edits + new files + deletions
    git -C <ws> diff --cached <base_commit> # one unified diff vs base
This mutates the workspace INDEX only (working tree untouched). Workspaces
are post-run scratch, so this is safe; re-running the script is idempotent.

base_commit resolution (in order):
  1. --base-commits-json FILE   explicit {"instance_id": "<sha>", ...} map
  2. dataset (default)          re-derive the exact sample set with
                                testbed.swebench.load_samples(split, seed, n)
                                from the run's config.json -- deterministic,
                                needs `datasets` installed (the runner host
                                already has it)
  3. --head-as-base             fallback: diff against the workspace's HEAD.
                                Only correct if the agent never committed.

Failed tasks (clone/session errors, missing workspace) get an EMPTY
model_patch -- the harness counts them as unresolved, which is what we
want: an agent that never produced a fix did not resolve the instance.

Usage:
  scripts/extract_predictions.py --run results/run1
  # -> results/run1/predictions.jsonl

  scripts/extract_predictions.py --run results/run1 \
      --model-name testbed-qwen3-coder --out /tmp/preds.jsonl
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Pathspec excludes appended to the diff command: agent-generated junk that
# must not ride into model_patch (it would be applied inside the eval
# container and can break test collection).
DEFAULT_EXCLUDES = [
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    "*.egg-info",
    ".opencode",
]


def _git(ws: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(ws), *args],
        capture_output=True,
        text=True,
    )


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    config = json.loads((run_dir / "config.json").read_text())
    records = []
    with (run_dir / "trace.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return config, records


def base_commits_from_dataset(config: dict) -> dict[str, str]:
    """Re-derive the run's exact sample set; load_samples is deterministic
    given (split, seed, n), so this returns precisely the instances the
    runner used."""
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "src"))
    from testbed import swebench  # noqa: PLC0415

    samples = swebench.load_samples(
        config["split"], config["seed"], config["num_samples"]
    )
    return {s["instance_id"]: s["base_commit"] for s in samples}


def extract_patch(ws: Path, base_commit: str | None,
                  excludes: list[str]) -> tuple[str, str | None]:
    """Returns (patch, error). patch='' on any failure."""
    if not (ws / ".git").is_dir():
        return "", f"not a git repo: {ws}"

    if base_commit is None:  # --head-as-base
        r = _git(ws, "rev-parse", "HEAD")
        if r.returncode != 0:
            return "", f"rev-parse HEAD failed: {r.stderr.strip()}"
        base_commit = r.stdout.strip()

    r = _git(ws, "add", "-A")
    if r.returncode != 0:
        return "", f"git add -A failed: {r.stderr.strip()}"

    pathspec = [".", *[f":(exclude){pat}" for pat in excludes]]
    r = _git(ws, "diff", "--cached", base_commit, "--", *pathspec)
    if r.returncode != 0:
        return "", f"git diff failed: {r.stderr.strip()}"
    return r.stdout, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path,
                    help="Run output dir containing config.json + trace.jsonl")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output predictions.jsonl (default <run>/predictions.jsonl)")
    ap.add_argument("--workspace-root", type=Path, default=None,
                    help="Override workspace root (default: from config.json snapshot)")
    ap.add_argument("--model-name", default="testbed",
                    help="model_name_or_path recorded in each prediction")
    ap.add_argument("--base-commits-json", type=Path, default=None,
                    help='Explicit {"instance_id": "<sha>"} map; skips dataset load')
    ap.add_argument("--head-as-base", action="store_true",
                    help="Diff against each workspace's HEAD instead of the "
                         "dataset base_commit (only correct if the agent never "
                         "committed)")
    ap.add_argument("--exclude", action="append", default=None,
                    metavar="PATHSPEC",
                    help=f"Extra git pathspec exclude (repeatable). "
                         f"Defaults always applied: {DEFAULT_EXCLUDES}")
    args = ap.parse_args()

    config, records = load_run(args.run)

    workspace_root = args.workspace_root
    if workspace_root is None:
        # config.json records the RAW yaml value; the runner normalizes it
        # with expanduser().resolve() before cloning. Mirror that here or a
        # relative / ~-prefixed workspace_root makes every ws.is_dir() fail
        # ("workspace missing" -> empty patches across the board).
        workspace_root = Path(config["config"]["workspace_root"]).expanduser().resolve()

    base_commits: dict[str, str] = {}
    if not args.head_as_base:
        if args.base_commits_json:
            base_commits = json.loads(args.base_commits_json.read_text())
        else:
            base_commits = base_commits_from_dataset(config)

    excludes = list(DEFAULT_EXCLUDES) + (args.exclude or [])
    out_path = args.out or (args.run / "predictions.jsonl")

    n_patch, n_empty = 0, 0
    with out_path.open("w") as f:
        for rec in records:
            iid = rec["instance_id"]
            ws = workspace_root / rec["directory"]
            patch, err = "", None

            if not ws.is_dir():
                err = f"workspace missing: {ws}"
            elif not args.head_as_base and iid not in base_commits:
                err = "no base_commit known (not in dataset map)"
            else:
                base = None if args.head_as_base else base_commits[iid]
                patch, err = extract_patch(ws, base, excludes)

            if patch:
                n_patch += 1
            else:
                n_empty += 1
                detail = err or "no changes in workspace"
                print(f"  empty patch: {iid:<40} ({detail})", file=sys.stderr)

            f.write(json.dumps({
                "instance_id": iid,
                "model_name_or_path": args.model_name,
                "model_patch": patch,
            }) + "\n")

    print(f"wrote {out_path}: {n_patch} with patch, {n_empty} empty "
          f"(of {len(records)} tasks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
