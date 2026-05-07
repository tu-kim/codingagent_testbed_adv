"""SWE-bench sample loading and prompt rendering."""

from __future__ import annotations

import random
from typing import Any


_DATASETS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "verified": "princeton-nlp/SWE-bench_Verified",
    "full": "princeton-nlp/SWE-bench",
}


def _load_dataset(hf_id: str, split: str = "test") -> list[dict[str, Any]]:
    """Indirection so tests can monkeypatch this without importing `datasets`."""
    from datasets import load_dataset  # type: ignore[import-not-found]

    ds = load_dataset(hf_id, split=split)
    return [dict(s) for s in ds]


def load_samples(split: str, seed: int, n: int) -> list[dict[str, Any]]:
    """Deterministic given (split, seed, n): sort by instance_id, then Random(seed).sample."""
    if split not in _DATASETS:
        raise ValueError(f"unknown split {split!r}; choose from {sorted(_DATASETS)}")
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    samples = _load_dataset(_DATASETS[split])
    samples.sort(key=lambda s: s["instance_id"])
    if n > len(samples):
        raise ValueError(f"requested n={n} but split has only {len(samples)} samples")
    return random.Random(seed).sample(samples, n)


_PROMPT_TEMPLATE = """\
You are working in a git checkout that has already been cloned and checked out at the
correct base commit. Your working directory is the repository root.

Apply a fix in place that resolves the issue described below. Edit files directly,
do not output a patch. When you are confident the fix is correct, stop.

# Issue

{problem_statement}
"""

_HINTS_BLOCK = """

# Hints

{hints}
"""


def render_prompt(sample: dict[str, Any]) -> str:
    """Render a SWE-bench sample into the prompt the OpenCode agent receives."""
    body = _PROMPT_TEMPLATE.format(problem_statement=sample["problem_statement"].strip())
    hints = (sample.get("hints_text") or "").strip()
    if hints:
        body += _HINTS_BLOCK.format(hints=hints)
    return body


if __name__ == "__main__":
    # Helper for scripts/curl_smoke.sh swebench: print the first lite sample's prompt.
    print(render_prompt(load_samples("lite", 0, 1)[0]))
