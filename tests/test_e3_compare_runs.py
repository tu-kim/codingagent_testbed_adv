"""Tests for scripts/arm/e3_compare_runs.py.

No network, no GPU. Loaded via importlib (script, not a package module),
matching this repo's convention for scripts/ tests.
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "arm" / "e3_compare_runs.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e3():
    return _load_module("e3_compare_runs", _SCRIPT_PATH)


# ---------- resolve_run ----------


def test_resolve_run_profiles_subdir(e3, tmp_path):
    root = tmp_path / "runA"
    (root / "profiles").mkdir(parents=True)
    label, inp, trace = e3.resolve_run(str(root))
    assert label == "runA"
    assert inp == root / "profiles"
    assert trace is None


def test_resolve_run_label_equals_dir_syntax(e3, tmp_path):
    root = tmp_path / "some_dir"
    (root / "profiles").mkdir(parents=True)
    label, inp, trace = e3.resolve_run(f"custom_label={root}")
    assert label == "custom_label"
    assert inp == root / "profiles"


def test_resolve_run_profiles_jsonl_file(e3, tmp_path):
    root = tmp_path / "runB"
    root.mkdir()
    (root / "profiles.jsonl").write_text("")
    label, inp, trace = e3.resolve_run(str(root))
    assert label == "runB"
    assert inp == root / "profiles.jsonl"


def test_resolve_run_missing_path_raises(e3, tmp_path):
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        e3.resolve_run(str(missing))


def test_resolve_run_trace_detected(e3, tmp_path):
    root = tmp_path / "runC"
    (root / "profiles").mkdir(parents=True)
    (root / "trace.jsonl").write_text("")
    label, inp, trace = e3.resolve_run(str(root))
    assert trace == root / "trace.jsonl"


def test_resolve_run_trace_absent(e3, tmp_path):
    root = tmp_path / "runD"
    (root / "profiles").mkdir(parents=True)
    label, inp, trace = e3.resolve_run(str(root))
    assert trace is None


def test_resolve_run_bare_dir_no_profiles_subdir_or_file(e3, tmp_path):
    # A dir that is neither <root>/profiles nor <root>/profiles.jsonl, but
    # exists itself -- falls through to "already a profiles dir/file".
    root = tmp_path / "already_profiles_dir"
    root.mkdir()
    (root / "sesX.jsonl").write_text("")
    label, inp, trace = e3.resolve_run(str(root))
    assert label == "already_profiles_dir"
    assert inp == root


# ---------- summarize ----------


def test_summarize_hand_computed(e3):
    rows = [(12, 8, 1, 3), (3.5, 3, 0.2, 0.3)]
    summ = e3.summarize(rows)

    llm = summ["llm"]
    assert llm["mean_s"] == pytest.approx(5.5)
    assert llm["p90_s"] == pytest.approx(7.5)
    assert llm["mean_share"] == pytest.approx(0.762, abs=1e-3)


def test_summarize_zero_wall_row_skipped_for_shares(e3):
    # A zero-wall row contributes to mean_s/p90_s but must be excluded from
    # mean_share (division by wall would be undefined/misleading).
    rows = [(12, 8, 1, 3), (0.0, 5, 0, 0)]
    summ = e3.summarize(rows)

    llm = summ["llm"]
    # mean_s over both rows: (8+5)/2 = 6.5
    assert llm["mean_s"] == pytest.approx(6.5)
    # mean_share only over the non-zero-wall row: 8/12
    assert llm["mean_share"] == pytest.approx(8 / 12)


def test_summarize_components_present(e3):
    rows = [(12, 8, 1, 3)]
    summ = e3.summarize(rows)
    assert set(summ.keys()) == {"llm", "tool", "scaffold"}
    assert summ["tool"]["mean_s"] == pytest.approx(1)
    assert summ["scaffold"]["mean_s"] == pytest.approx(3)


# ---------- main() integration ----------


def _write_jsonl(path: Path, records: list[dict]):
    with path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _synthetic_session_events(sid: str) -> list[dict]:
    """Two turns for one session engineered to produce canonical
    (wall, llm, tool, scaffold) = (12, 8, 1, 3) and (3.5, 3, 0.2, 0.3),
    matching the hand-computed summarize() case above."""
    return [
        {"ev": "turn.start", "sessionID": sid, "step": 0, "ts": 0},
        {"ev": "llm.end", "sessionID": sid, "step": 0, "ts": 8,
         "dynamo": {"elapsed_s": 8}},
        {"ev": "turn.end", "sessionID": sid, "step": 0, "ts": 9,
         "duration_s": 9, "llm_wall_s": 7, "tool_wall_s": 1,
         "post_overhead_s": 1},
        {"ev": "turn.start", "sessionID": sid, "step": 1, "ts": 12},
        {"ev": "llm.end", "sessionID": sid, "step": 1, "ts": 15,
         "dynamo": {"elapsed_s": 3}},
        {"ev": "turn.end", "sessionID": sid, "step": 1, "ts": 15.5,
         "duration_s": 3.5, "llm_wall_s": 3, "tool_wall_s": 0.2,
         "post_overhead_s": 0.3},
    ]


def _make_run_dir(tmp_path: Path, name: str, *, session_ids: list[str],
                   trace_session_ids: list[str] | None) -> Path:
    root = tmp_path / name
    profiles = root / "profiles"
    profiles.mkdir(parents=True)
    for sid in session_ids:
        _write_jsonl(profiles / f"{sid}.jsonl", _synthetic_session_events(sid))
    if trace_session_ids is not None:
        trace_records = [
            {"instance_id": f"inst-{sid}", "session_id": sid, "success": True}
            for sid in trace_session_ids
        ]
        _write_jsonl(root / "trace.jsonl", trace_records)
    return root


def test_main_no_figures_writes_comparison_csv(e3, tmp_path):
    run_dir = _make_run_dir(tmp_path, "runA", session_ids=["sesA"],
                             trace_session_ids=["sesA"])
    out_dir = tmp_path / "out"

    rc = e3.main([str(run_dir), "--out", str(out_dir), "--no-figures"])

    assert rc == 0
    csv_path = out_dir / "comparison.csv"
    assert csv_path.is_file()
    # no figure should have been produced
    assert not (out_dir / "fig_share_by_run.pdf").exists()

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3  # llm, tool, scaffold
    by_component = {r["component"]: r for r in rows}
    assert set(by_component) == {"llm", "tool", "scaffold"}

    llm_row = by_component["llm"]
    assert llm_row["run"] == "runA"
    assert float(llm_row["mean_s"]) == pytest.approx(5.5, abs=1e-4)
    assert float(llm_row["p90_s"]) == pytest.approx(7.5, abs=1e-4)
    assert float(llm_row["mean_share"]) == pytest.approx(0.762, abs=1e-3)
    assert int(llm_row["n_turns"]) == 2

    tool_row = by_component["tool"]
    assert float(tool_row["mean_s"]) == pytest.approx((1 + 0.2) / 2, abs=1e-4)

    scaffold_row = by_component["scaffold"]
    assert float(scaffold_row["mean_s"]) == pytest.approx((3 + 0.3) / 2, abs=1e-4)


def test_main_trace_filters_to_kept_sessions(e3, tmp_path):
    # Two sessions in profiles/, but trace.jsonl only names one of them --
    # the other (e.g. a title-gen or nested `task` session) must be dropped.
    run_dir = _make_run_dir(
        tmp_path, "runFiltered",
        session_ids=["sesA", "sesB"],
        trace_session_ids=["sesA"],
    )
    out_dir = tmp_path / "out_filtered"

    rc = e3.main([str(run_dir), "--out", str(out_dir), "--no-figures"])
    assert rc == 0

    with (out_dir / "comparison.csv").open() as f:
        rows = list(csv.DictReader(f))

    # Only sesA's 2 turns should be counted; sesB is filtered out entirely.
    for r in rows:
        assert int(r["n_turns"]) == 2

    by_component = {r["component"]: r for r in rows}
    assert float(by_component["llm"]["mean_s"]) == pytest.approx(5.5, abs=1e-4)


def test_main_missing_run_returns_error_rc(e3, tmp_path):
    missing = tmp_path / "nope"
    rc = e3.main([str(missing), "--out", str(tmp_path / "out2"),
                  "--no-figures"])
    assert rc == 2
