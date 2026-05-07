from __future__ import annotations

import pytest

from testbed import swebench


def _fake_samples():
    return [
        {"instance_id": "z__z-1", "repo": "z/z", "base_commit": "deadbeef", "problem_statement": "Z", "hints_text": ""},
        {"instance_id": "a__a-1", "repo": "a/a", "base_commit": "cafef00d", "problem_statement": "A", "hints_text": "h"},
        {"instance_id": "m__m-1", "repo": "m/m", "base_commit": "01234567", "problem_statement": "M", "hints_text": ""},
    ]


def test_load_samples_sorts_then_samples_deterministically(monkeypatch):
    monkeypatch.setattr(swebench, "_load_dataset", lambda hf_id, split="test": _fake_samples())
    out = swebench.load_samples("lite", seed=0, n=2)
    out2 = swebench.load_samples("lite", seed=0, n=2)
    assert out == out2  # determinism
    ids = sorted(s["instance_id"] for s in _fake_samples())
    # Sampling pool is the sorted list, so picks must be from sorted ids.
    assert all(o["instance_id"] in ids for o in out)


def test_load_samples_unknown_split():
    with pytest.raises(ValueError):
        swebench.load_samples("nope", 0, 1)


def test_render_prompt_includes_hints_when_present():
    s = {"problem_statement": "fix the bug", "hints_text": "look at foo.py"}
    out = swebench.render_prompt(s)
    assert "fix the bug" in out
    assert "look at foo.py" in out
    assert "# Hints" in out


def test_render_prompt_skips_hints_section_when_empty():
    s = {"problem_statement": "fix the bug", "hints_text": ""}
    out = swebench.render_prompt(s)
    assert "fix the bug" in out
    assert "# Hints" not in out


def test_render_prompt_handles_missing_hints_field():
    s = {"problem_statement": "fix"}
    out = swebench.render_prompt(s)
    assert "# Hints" not in out
