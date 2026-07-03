"""APPS (Automated Programming Progress Standard) sample loading, prompt
rendering, and workspace preparation.

Mirrors swebench.py's module contract -- load_samples(split, seed, n) is
deterministic the same way (sort by stable id, then Random(seed).sample)
and render_prompt(sample) returns the full agent prompt. Adds
prepare_workspace(), the APPS counterpart of runner._pre_clone: APPS
problems have no git repo, so a workspace is materialized locally
(PROBLEM.md + solution.py) instead of cloned. Samples returned by
load_samples carry an injected "instance_id" (f"apps-{problem_id:05d}")
so the runner's directory/manifest/trace plumbing works unchanged.

Dataset: codeparrot/apps (HF). 10,000 problems, HF splits train/test
(5,000 each), difficulty introductory/interview/competition. Fields:
  problem_id: int      unique across BOTH splits
  question: str        full problem statement (includes example I/O)
  starter_code: str    LeetCode-style stub for call-based problems ("" for stdio)
  input_output: str    JSON: {"inputs": [...], "outputs": [...], "fn_name"?: str}
                       fn_name present -> call-based (judge imports + calls it)
                       fn_name absent  -> stdio (judge pipes stdin, reads stdout)
  solutions: str       JSON list of reference solutions (unused here)
  difficulty: str, url: str

Split names accepted here: the two HF splits ("train", "test") plus the
three difficulty names ("introductory", "interview", "competition"),
which select the TEST split filtered to that difficulty. Filtering
happens before the sort+sample so selection stays deterministic.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

_HF_ID = "codeparrot/apps"

# split name -> (hf_split, difficulty filter or None)
_SPLIT_MAP: dict[str, tuple[str, str | None]] = {
    "train": ("train", None),
    "test": ("test", None),
    "introductory": ("test", "introductory"),
    "interview": ("test", "interview"),
    "competition": ("test", "competition"),
}

SPLITS: tuple[str, ...] = tuple(_SPLIT_MAP)

PROBLEM_FILE = "PROBLEM.md"
SOLUTION_FILE = "solution.py"


def _load_dataset(hf_id: str, split: str) -> list[dict[str, Any]]:
    """Indirection so tests can monkeypatch this without importing `datasets`.

    The "all" config is the full dataset (the difficulty-named configs are
    subsets); we always load "all" and filter difficulty in Python so the
    deterministic-selection contract has a single code path.

    codeparrot/apps is a legacy SCRIPT-based hub dataset (it ships an
    apps.py loading script). datasets >= 4.0 removed script support
    entirely ("Dataset scripts are no longer supported"), and 2.20+ gates
    it behind trust_remote_code (ValueError). Fallback: read the Hub's
    auto-converted parquet export (refs/convert/parquet branch) through
    the packaged parquet builder -- same rows/fields, no script."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    try:
        ds = load_dataset(hf_id, "all", split=split)
    except (RuntimeError, ValueError):
        # The convert branch lays shards out as <config>/<split>/*.parquet.
        # With explicit data_files the parquet builder names its single
        # split "train" regardless of the source split -- request that.
        files = f"hf://datasets/{hf_id}@refs/convert/parquet/all/{split}/*.parquet"
        ds = load_dataset("parquet", data_files=files, split="train")
    return [dict(s) for s in ds]


def instance_id_for(sample: dict[str, Any]) -> str:
    """Stable, path-safe id: zero-padded so lexicographic == numeric order."""
    return f"apps-{int(sample['problem_id']):05d}"


def load_samples(split: str, seed: int, n: int) -> list[dict[str, Any]]:
    """Deterministic given (split, seed, n): filter difficulty (when the
    split names one), sort by problem_id, then Random(seed).sample.
    Each returned sample carries an injected "instance_id"."""
    if split not in _SPLIT_MAP:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(_SPLIT_MAP)}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    hf_split, difficulty = _SPLIT_MAP[split]
    samples = _load_dataset(_HF_ID, hf_split)
    if difficulty is not None:
        samples = [s for s in samples if s.get("difficulty") == difficulty]
    for s in samples:
        s["instance_id"] = instance_id_for(s)
    samples.sort(key=lambda s: int(s["problem_id"]))
    if n > len(samples):
        raise ValueError(f"requested n={n} but split {split!r} has only "
                         f"{len(samples)} samples")
    return random.Random(seed).sample(samples, n)


def parse_input_output(sample: dict[str, Any]) -> dict[str, Any]:
    """Parse the sample's input_output JSON string.

    Returns {"inputs": [...], "outputs": [...], "fn_name": str | None};
    all-empty on missing/malformed JSON (a handful of train problems have
    an empty string here)."""
    raw = sample.get("input_output") or ""
    try:
        io = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        io = None
    if not isinstance(io, dict):
        return {"inputs": [], "outputs": [], "fn_name": None}
    fn = io.get("fn_name")
    return {
        "inputs": list(io.get("inputs") or []),
        "outputs": list(io.get("outputs") or []),
        "fn_name": str(fn) if fn else None,
    }


_PROMPT_TEMPLATE = """\
You are working in a prepared workspace directory. It contains:

  - PROBLEM.md   -- the full problem statement (identical to the one below)
  - solution.py  -- the file you must complete with your final Python 3 solution

{mode_block}

Edit solution.py in place; do not create a different entry-point file. When you
are confident the solution is correct, stop.

# Problem

{question}
"""

_STDIO_MODE_BLOCK = """\
The problem is judged in STDIN/STDOUT mode: solution.py is executed as
`python3 solution.py`, receives the test input on standard input, and must
print exactly the expected output to standard output -- nothing else."""

_CALL_MODE_BLOCK = """\
The problem is judged in FUNCTION-CALL mode: the judge imports solution.py and
calls `{fn_name}` with the test arguments. Complete the starter code already
present in solution.py, keeping the existing class/function signature."""


def render_prompt(sample: dict[str, Any]) -> str:
    """Render an APPS sample into the prompt the OpenCode agent receives."""
    fn_name = parse_input_output(sample)["fn_name"]
    if fn_name:
        mode_block = _CALL_MODE_BLOCK.format(fn_name=fn_name)
    else:
        mode_block = _STDIO_MODE_BLOCK
    return _PROMPT_TEMPLATE.format(
        mode_block=mode_block,
        question=str(sample["question"]).strip(),
    )


_STDIO_SCAFFOLD = """\
# Write your Python 3 solution for the problem in PROBLEM.md here.
# Read the test input from stdin and print the answer to stdout.
"""


def _initial_solution(sample: dict[str, Any]) -> str:
    """starter_code verbatim for call-based problems; a comment scaffold
    for stdio problems (starter_code is "" there)."""
    starter = (sample.get("starter_code") or "").rstrip()
    if starter:
        return starter + "\n"
    return _STDIO_SCAFFOLD


async def prepare_workspace(sample: dict[str, Any], dest: Path, *,
                            reset: bool = False) -> None:
    """Materialize the APPS task workspace at <dest>. The async signature
    matches runner._pre_clone so the runner's workload dispatch treats both
    uniformly; the body is tiny synchronous file I/O (no subprocess, no
    network), so nothing here actually awaits.

    Idempotent like _pre_clone: an existing workspace with PROBLEM.md
    present is a no-op when reset=False (agent artifacts from a prior run
    are preserved for inspection). With reset=True the directory is wiped
    and re-materialized so the workspace state -- and therefore opencode's
    cwd-embedding system prompt -- is byte-stable across reruns. A dest
    that exists but is missing PROBLEM.md (interrupted materialization)
    gets its two task files rewritten in place in both modes."""
    dest = Path(dest)
    marker = dest / PROBLEM_FILE
    if dest.exists():
        if not dest.is_dir():
            dest.unlink()
        elif reset:
            shutil.rmtree(dest)
        elif marker.exists():
            return
    dest.mkdir(parents=True, exist_ok=True)
    (dest / PROBLEM_FILE).write_text(
        str(sample["question"]).strip() + "\n", encoding="utf-8")
    (dest / SOLUTION_FILE).write_text(
        _initial_solution(sample), encoding="utf-8")


if __name__ == "__main__":
    # Helper for smoke tests: print the first test sample's prompt.
    print(render_prompt(load_samples("test", 0, 1)[0]))
