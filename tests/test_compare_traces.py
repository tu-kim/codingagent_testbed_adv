"""Tests for scripts/compare_traces.py.

N-way reproducibility comparison of trace.jsonl runs, keyed by
instance_id. Covers per-session metric extraction, the status
classification (REPRODUCIBLE / TRAJ_DIFF_SAME_ANSWER / ANSWER_DIFF /
INSUFFICIENT), N>2 comparison, and the matplotlib-gated figures.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_traces.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("compare_traces", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["compare_traces"] = module
    spec.loader.exec_module(module)
    return module


def _assistant(text=None, tools=None, steps=1, out_tokens=0, in_tokens=0):
    """Build an assistant message dict. tools = list of (name, input)."""
    parts = []
    for _ in range(steps):
        parts.append({"type": "step-start"})
    if text is not None:
        parts.append({"type": "text", "text": text})
    for name, inp in (tools or []):
        parts.append({"type": "tool", "tool": name, "state": {"input": inp}})
    return {
        "info": {"role": "assistant",
                 "tokens": {"input": in_tokens, "output": out_tokens,
                            "cache": {"read": 0}}},
        "parts": parts,
    }


def _user_with_diffs(diffs):
    """User message carrying summary.diffs (opencode attaches it there)."""
    return {"info": {"role": "user",
                     "summary": {"diffs": diffs}},
            "parts": [{"type": "text", "text": "task"}]}


def _record(instance_id, *, session_id="ses_x", success=True, rtt_s=10.0,
            messages=None):
    return {
        "instance_id": instance_id,
        "session_id": session_id,
        "directory": f"session-{instance_id}",
        "arrival_offset_s": 0.0,
        "rtt_s": rtt_s,
        "success": success,
        "error": None,
        "messages": messages or [],
    }


def _write_trace(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


# ---------- extraction ----------


def test_extract_session_counts_turns_tools_tokens(mod):
    rec = _record("inst-1", messages=[
        _user_with_diffs([{"file": "a.py", "additions": 3, "deletions": 1}]),
        _assistant(text="let me look", tools=[("read", {"f": "a.py"})],
                   steps=1, out_tokens=12),
        _assistant(text="fixing", tools=[("edit", {"f": "a.py"})],
                   steps=1, out_tokens=20),
    ])
    sm = mod.extract_session(rec)
    assert sm.instance_id == "inst-1"
    assert sm.n_turns == 2                 # two step-start parts
    assert sm.n_tool_calls == 2
    assert sm.output_tokens == 32          # 12 + 20
    assert sm.tool_sequence == ["read", "edit"]
    assert sm.diff_signature == [("a.py", 3, 1)]
    assert sm.final_text == "fixing"       # last assistant text


def test_extract_session_turn_fallback_to_assistant_count(mod):
    """If there are no step-start parts (older trace), n_turns falls back
    to the assistant-message count."""
    rec = _record("inst-1", messages=[
        {"info": {"role": "assistant", "tokens": {"output": 5}},
         "parts": [{"type": "text", "text": "hi"}]},
    ])
    sm = mod.extract_session(rec)
    assert sm.n_turns == 1


def test_session_id_not_used_as_key(mod, tmp_path):
    """Two runs of the same instance_id have DIFFERENT session_ids; the
    loader must key by instance_id so they join."""
    p = tmp_path / "trace.jsonl"
    _write_trace(p, [_record("inst-1", session_id="ses_aaa")])
    d = mod.load_trace(p)
    assert set(d) == {"inst-1"}
    assert d["inst-1"].session_id == "ses_aaa"


# ---------- status classification ----------


def _sm(mod, iid, traj_text, diffs, out_tokens=10, turns=1, rtt=10.0):
    rec = _record(iid, rtt_s=rtt, messages=[
        _user_with_diffs(diffs),
        _assistant(text=traj_text, tools=[("edit", {"f": "x"})],
                   steps=turns, out_tokens=out_tokens),
    ])
    return mod.extract_session(rec)


def test_status_reproducible(mod):
    a = _sm(mod, "i", "same", [{"file": "x", "additions": 1, "deletions": 0}])
    b = _sm(mod, "i", "same", [{"file": "x", "additions": 1, "deletions": 0}])
    c = mod.InstanceComparison("i", ["r1", "r2"], [a, b], n_runs=2)
    assert c.status == "REPRODUCIBLE"
    assert c.distinct_trajectories == 1
    assert c.complete is True


def test_status_answer_diff(mod):
    """Different final answer text => ANSWER_DIFF."""
    a = _sm(mod, "i", "answer one", [{"file": "x", "additions": 1, "deletions": 0}])
    b = _sm(mod, "i", "answer TWO", [{"file": "x", "additions": 1, "deletions": 0}])
    c = mod.InstanceComparison("i", ["r1", "r2"], [a, b], n_runs=2)
    assert c.status == "ANSWER_DIFF"
    assert c.distinct_answers == 2


def test_status_answer_diff_on_different_diff(mod):
    """Same final text but DIFFERENT code diff => still ANSWER_DIFF (the
    agent changed different files)."""
    a = _sm(mod, "i", "done", [{"file": "x.py", "additions": 1, "deletions": 0}])
    b = _sm(mod, "i", "done", [{"file": "y.py", "additions": 2, "deletions": 0}])
    c = mod.InstanceComparison("i", ["r1", "r2"], [a, b], n_runs=2)
    assert c.status == "ANSWER_DIFF"


def test_status_traj_diff_same_answer(mod):
    """Trajectory differs (different tool input) but final text AND diff
    identical => TRAJ_DIFF_SAME_ANSWER (converged despite different path)."""
    a = _record("i", messages=[
        _user_with_diffs([{"file": "x", "additions": 1, "deletions": 0}]),
        _assistant(text="done", tools=[("read", {"f": "a"})], out_tokens=10),
    ])
    b = _record("i", messages=[
        _user_with_diffs([{"file": "x", "additions": 1, "deletions": 0}]),
        _assistant(text="done", tools=[("read", {"f": "DIFFERENT"})], out_tokens=10),
    ])
    sa, sb = mod.extract_session(a), mod.extract_session(b)
    assert sa.trajectory != sb.trajectory          # tool input differs
    assert sa.answer_key() == sb.answer_key()       # but answer identical
    c = mod.InstanceComparison("i", ["r1", "r2"], [sa, sb], n_runs=2)
    assert c.status == "TRAJ_DIFF_SAME_ANSWER"


def test_status_insufficient_when_present_in_one_run(mod):
    a = _sm(mod, "i", "x", [])
    c = mod.InstanceComparison("i", ["r1"], [a], n_runs=3)
    assert c.status == "INSUFFICIENT"
    assert c.complete is False


# ---------- N-way (3+) ----------


def test_compare_three_runs_all_reproducible(mod, tmp_path):
    msgs = [
        _user_with_diffs([{"file": "x", "additions": 1, "deletions": 0}]),
        _assistant(text="done", tools=[("edit", {"f": "x"})], out_tokens=10),
    ]
    paths = []
    for i in range(3):
        p = tmp_path / f"run{i}" / "trace.jsonl"
        p.parent.mkdir()
        _write_trace(p, [_record("inst-1", session_id=f"ses_{i}", messages=msgs)])
        paths.append(p)
    runs = [(mod._label_for(p), mod.load_trace(p)) for p in paths]
    comps = mod.compare(runs)
    assert len(comps) == 1
    c = comps[0]
    assert c.n_runs == 3
    assert len(c.metrics) == 3
    assert c.complete is True
    assert c.status == "REPRODUCIBLE"


def test_compare_four_runs_one_diverges(mod):
    base = [
        _user_with_diffs([{"file": "x", "additions": 1, "deletions": 0}]),
        _assistant(text="done", tools=[("edit", {"f": "x"})], out_tokens=10),
    ]
    diverged = [
        _user_with_diffs([{"file": "x", "additions": 1, "deletions": 0}]),
        _assistant(text="DIFFERENT", tools=[("edit", {"f": "x"})], out_tokens=99),
    ]
    runs = []
    for i in range(3):
        runs.append((f"r{i}", {"inst-1": mod.extract_session(_record("inst-1", messages=base))}))
    runs.append(("r3", {"inst-1": mod.extract_session(_record("inst-1", messages=diverged))}))
    comps = mod.compare(runs)
    c = comps[0]
    assert len(c.metrics) == 4
    assert c.distinct_answers == 2     # 3 same + 1 different
    assert c.status == "ANSWER_DIFF"


def test_compare_missing_from_some_runs(mod):
    a = mod.extract_session(_record("only-a", messages=[]))
    shared1 = mod.extract_session(_record("shared", messages=[]))
    shared2 = mod.extract_session(_record("shared", messages=[]))
    runs = [("r1", {"only-a": a, "shared": shared1}),
            ("r2", {"shared": shared2})]
    comps = {c.instance_id: c for c in mod.compare(runs)}
    assert comps["only-a"].status == "INSUFFICIENT"   # present in 1 run
    assert comps["only-a"].complete is False
    assert comps["shared"].complete is True           # present in both


# ---------- CSV / main ----------


def test_main_three_runs_exit_zero_when_reproducible(mod, tmp_path, capsys):
    msgs = [
        _user_with_diffs([{"file": "x", "additions": 1, "deletions": 0}]),
        _assistant(text="done", tools=[("edit", {"f": "x"})], out_tokens=10),
    ]
    paths = []
    for i in range(3):
        p = tmp_path / f"trace{i}.jsonl"
        _write_trace(p, [_record("inst-1", session_id=f"ses_{i}", messages=msgs)])
        paths.append(str(p))
    out = tmp_path / "cmp"
    rc = mod.main(["--traces", *paths, "--output", str(out)])
    assert rc == 0      # reproducible → exit 0

    per = list(csv.DictReader((out / "per_instance.csv").open()))
    assert len(per) == 1
    assert per[0]["status"] == "REPRODUCIBLE"
    assert per[0]["n_runs_present"] == "3"

    runs = list(csv.DictReader((out / "runs_summary.csv").open()))
    assert len(runs) == 3
    assert "REPRODUCIBLE" in capsys.readouterr().out


def test_main_exit_3_on_answer_diff(mod, tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_trace(a, [_record("i", messages=[
        _user_with_diffs([]), _assistant(text="one", out_tokens=5)])])
    _write_trace(b, [_record("i", messages=[
        _user_with_diffs([]), _assistant(text="two", out_tokens=9)])])
    out = tmp_path / "cmp"
    rc = mod.main(["--traces", str(a), str(b), "--output", str(out)])
    assert rc == 3      # answer divergence → exit 3 (CI gate signal)


def test_main_requires_two_traces(mod, tmp_path):
    a = tmp_path / "a.jsonl"
    _write_trace(a, [_record("i")])
    rc = mod.main(["--traces", str(a), "--output", str(tmp_path / "o")])
    assert rc == 2


def test_main_labels_count_mismatch(mod, tmp_path):
    a = tmp_path / "a.jsonl"; _write_trace(a, [_record("i")])
    b = tmp_path / "b.jsonl"; _write_trace(b, [_record("i")])
    rc = mod.main(["--traces", str(a), str(b),
                   "--labels", "only-one", "--output", str(tmp_path / "o")])
    assert rc == 2


def test_label_for_uses_parent_dir_for_trace_jsonl(mod, tmp_path):
    p = tmp_path / "run7" / "trace.jsonl"
    p.parent.mkdir()
    p.write_text("")
    assert mod._label_for(p) == "run7"
    other = tmp_path / "myrun.jsonl"
    other.write_text("")
    assert mod._label_for(other) == "myrun"


# ---------- figures (matplotlib-gated) ----------


def test_make_figures_writes_pdfs(mod, tmp_path):
    pytest.importorskip("matplotlib")
    a = mod.extract_session(_record("i1", rtt_s=10.0, messages=[
        _user_with_diffs([]), _assistant(text="x", out_tokens=5, steps=2)]))
    b = mod.extract_session(_record("i1", rtt_s=12.0, messages=[
        _user_with_diffs([]), _assistant(text="y", out_tokens=9, steps=4)]))
    runs = [("r1", {"i1": a}), ("r2", {"i1": b})]
    comps = mod.compare(runs)
    out = tmp_path / "figs"
    out.mkdir()
    paths = mod.make_figures(comps, runs, out)
    names = {p.name for p in paths}
    assert "fig_status_breakdown.pdf" in names
    assert "fig_turns_spread.pdf" in names      # turns differ (2 vs 4 steps)
    assert "fig_rtt_by_run.pdf" in names
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_main_figures_flag(mod, tmp_path):
    pytest.importorskip("matplotlib")
    paths = []
    for i in range(2):
        p = tmp_path / f"t{i}.jsonl"
        _write_trace(p, [_record("i", session_id=f"s{i}", rtt_s=10.0 + i, messages=[
            _user_with_diffs([]), _assistant(text="x", out_tokens=5)])])
        paths.append(str(p))
    out = tmp_path / "cmp"
    rc = mod.main(["--traces", *paths, "--output", str(out), "--figures"])
    assert rc == 0
    assert (out / "fig_status_breakdown.pdf").exists()
