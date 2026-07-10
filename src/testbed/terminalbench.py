"""Terminal-Bench 2.0 sample loading, prompt rendering, and workspace
preparation.

Mirrors swebench.py/apps.py's module contract -- load_samples(split, seed, n)
is deterministic the same way (sort by stable id, then Random(seed).sample)
and render_prompt(sample) returns the full agent prompt. prepare_workspace()
is the workspace materializer (TASK.md only -- see below). Samples returned
by load_samples carry an injected "instance_id" (f"terminalbench-{task_id}")
so the runner's directory/manifest/trace plumbing works unchanged.

Dataset: harborframework/terminal-bench-2.0 (HF *dataset repo*, pinned at
_HF_REVISION). It is NOT a parquet/`datasets`-loadable dataset -- the repo is
a flat collection of Harbor-format task directories (89 at the pinned
revision), so we `huggingface_hub.snapshot_download` the repo (hf_hub is a
hard dependency of `datasets`, already in our tree; HF_HOME caching and
HF_HUB_OFFLINE behave as with the other workloads) and read per task:

    <task-id>/
      instruction.md   agent-facing task instruction (becomes TASK.md + prompt)
      task.toml        [metadata] difficulty/category/tags (+ timeouts, image)
      environment/     Dockerfile + challenge-construction scripts
      solution/        oracle solution
      tests/           verifier

Only instruction.md is materialized into the workspace (as TASK.md).
environment/, solution/ and tests/ are deliberately NOT copied: solution/ and
tests/ would leak the oracle into the agent's view, and environment/ is the
Docker build context -- its challenge-setup scripts CONSTRUCT the task state,
which is equally answer-adjacent. The real benchmark executes each task inside
its Docker image; this testbed does not reproduce that container, so absolute
paths named by instructions (/app/...) don't exist on the host. render_prompt
therefore tells the agent to treat the workspace as the task's root filesystem
and re-anchor absolute paths inside it -- keeping the agent loop inside the
workspace (measurement + host hygiene; out-of-tree writes would ride the
external_directory permission, allowed but noisy). The trade-off is explicit:
this workload exists to shape realistic terminal-command-heavy agent traffic
for router/scheduling measurement, not to score Terminal-Bench correctness
(that requires the official Harbor harness + per-task containers).

Split names accepted here: "test" (all tasks) plus the difficulty
pseudo-splits "easy" | "medium" | "hard" (= tasks whose task.toml
[metadata].difficulty equals that value, filtered BEFORE the sort+sample so
selection stays deterministic).
"""

from __future__ import annotations

import random
import shutil
import tomllib
from pathlib import Path
from typing import Any

_HF_REPO = "harborframework/terminal-bench-2.0"
# Pinned so the task pool -- and therefore Random(seed).sample selection --
# cannot drift if upstream adds/renames tasks. 2026-04-24, 89 tasks.
_HF_REVISION = "f2e8c75e23add71613117eecc9498f53bcd7e04e"

# split name -> difficulty filter (None = all tasks)
_SPLIT_MAP: dict[str, str | None] = {
    "test": None,
    "easy": "easy",
    "medium": "medium",
    "hard": "hard",
}

SPLITS: tuple[str, ...] = tuple(_SPLIT_MAP)

TASK_FILE = "TASK.md"


def _snapshot_dataset() -> Path:
    """Download (or reuse the HF-cached copy of) the pinned dataset repo and
    return its local root. Indirection so tests can point _load_tasks at a
    tmp_path fake tree without network/huggingface_hub."""
    from huggingface_hub import snapshot_download  # type: ignore[import-not-found]

    return Path(snapshot_download(repo_id=_HF_REPO, repo_type="dataset",
                                  revision=_HF_REVISION))


def _load_tasks() -> list[dict[str, Any]]:
    """Parse every task directory (= a root-level dir holding task.toml AND
    instruction.md) into a sample dict. Indirection so selection tests can
    monkeypatch this with an in-memory pool.

    sorted() on the glob because directory iteration order is
    filesystem-dependent -- load_samples re-sorts anyway, but a deterministic
    base list keeps this layer reproducible on its own. A malformed task.toml
    (or one without [metadata]) degrades to metadata-less fields rather than
    failing the whole load; a dir missing instruction.md is skipped (there is
    nothing to prompt or materialize)."""
    root = _snapshot_dataset()
    tasks: list[dict[str, Any]] = []
    for toml_path in sorted(root.glob("*/task.toml")):
        task_dir = toml_path.parent
        instruction_path = task_dir / "instruction.md"
        if not instruction_path.exists():
            continue
        try:
            meta = tomllib.loads(toml_path.read_text(encoding="utf-8")).get("metadata", {})
        except tomllib.TOMLDecodeError:
            meta = {}
        if not isinstance(meta, dict):
            meta = {}
        tasks.append({
            "task_id": task_dir.name,
            "instruction": instruction_path.read_text(encoding="utf-8"),
            "difficulty": meta.get("difficulty"),
            "category": meta.get("category"),
            "tags": list(meta.get("tags") or []),
        })
    return tasks


def instance_id_for(sample: dict[str, Any]) -> str:
    """Stable, path-safe id (task dir names are kebab-case by construction)."""
    return f"terminalbench-{sample['task_id']}"


def load_samples(split: str, seed: int, n: int) -> list[dict[str, Any]]:
    """Deterministic given (split, seed, n): filter difficulty (when the
    split names one), sort by task_id, then Random(seed).sample.
    Each returned sample carries an injected "instance_id"."""
    if split not in _SPLIT_MAP:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(_SPLIT_MAP)}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    difficulty = _SPLIT_MAP[split]
    samples = _load_tasks()
    if difficulty is not None:
        samples = [s for s in samples if s.get("difficulty") == difficulty]
    for s in samples:
        s["instance_id"] = instance_id_for(s)
    samples.sort(key=lambda s: str(s["task_id"]))
    if n > len(samples):
        raise ValueError(f"requested n={n} but split {split!r} has only "
                         f"{len(samples)} samples")
    return random.Random(seed).sample(samples, n)


_PROMPT_TEMPLATE = """\
You are working in a prepared workspace directory. It contains:

  - TASK.md   -- the full task description (identical to the one below)

The task below comes from Terminal-Bench: complete it by running shell
commands and editing files from inside the workspace. Treat the workspace
directory as the task's root filesystem -- when the task references an
absolute path (e.g. /app/data.txt), create and use the corresponding path
INSIDE the workspace (./app/data.txt) instead of the real system path, and do
not write outside the workspace. When you are confident the task is complete,
stop.

# Task

{instruction}
"""


def render_prompt(sample: dict[str, Any]) -> str:
    """Render a Terminal-Bench sample into the prompt the OpenCode agent
    receives."""
    return _PROMPT_TEMPLATE.format(instruction=str(sample["instruction"]).strip())


async def prepare_workspace(sample: dict[str, Any], dest: Path, *,
                            reset: bool = False) -> None:
    """Materialize the Terminal-Bench task workspace at <dest>: TASK.md only
    (see the module docstring for why environment/solution/tests stay out).
    The async signature matches runner._pre_clone so the runner's workload
    dispatch treats all workloads uniformly; the body is tiny synchronous
    file I/O (no subprocess, no network), so nothing here actually awaits.

    Idempotent like _pre_clone / apps.prepare_workspace: an existing
    workspace with TASK.md present is a no-op when reset=False (agent
    artifacts from a prior run are preserved for inspection). With
    reset=True the directory is wiped and re-materialized so the workspace
    state -- and therefore opencode's cwd-embedding system prompt -- is
    byte-stable across reruns. A dest that exists but is missing TASK.md
    (interrupted materialization) gets the task file rewritten in place in
    both modes."""
    dest = Path(dest)
    marker = dest / TASK_FILE
    if dest.exists():
        if not dest.is_dir():
            dest.unlink()
        elif reset:
            shutil.rmtree(dest)
        elif marker.exists():
            return
    dest.mkdir(parents=True, exist_ok=True)
    (dest / TASK_FILE).write_text(
        str(sample["instruction"]).strip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    # Helper for smoke tests: print the first test sample's prompt.
    print(render_prompt(load_samples("test", 0, 1)[0]))
