"""Tests for scripts/analyze_eval_results.py.

Pure JSON fixtures in tmp_path. No network, no GPU.
Covers: load_report verdict precedence, both harness key styles,
find_report (skips config.json/summary.json, picks newest json),
not_in_report verdict, csv output, exit 1 when no report found.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_eval_results.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_eval_results", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_eval_results"] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_report(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def _write_trace(run_dir: Path, records: list[dict]) -> None:
    (run_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )


def _make_run(tmp_path: Path, records: list[dict]) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    _write_trace(run_dir, records)
    return run_dir


# ---------------------------------------------------------------------------
# load_report — both key styles
# ---------------------------------------------------------------------------

def test_load_report_resolved_ids_style(mod, tmp_path):
    """New-style harness keys (resolved_ids, unresolved_ids, …) are parsed."""
    report = tmp_path / "report.json"
    _write_report(report, {
        "resolved_ids": ["a__a-1", "b__b-2"],
        "unresolved_ids": ["c__c-3"],
        "error_ids": [],
    })
    verdicts = mod.load_report(report)
    assert verdicts["a__a-1"] == "resolved"
    assert verdicts["b__b-2"] == "resolved"
    assert verdicts["c__c-3"] == "unresolved"


def test_load_report_resolved_style(mod, tmp_path):
    """Old-style harness keys (resolved, unresolved, …) are parsed."""
    report = tmp_path / "report.json"
    _write_report(report, {
        "resolved": ["x__x-1"],
        "unresolved": ["x__x-2"],
        "empty_patch": ["x__x-3"],
        "incomplete": [],
    })
    verdicts = mod.load_report(report)
    assert verdicts["x__x-1"] == "resolved"
    assert verdicts["x__x-2"] == "unresolved"
    assert verdicts["x__x-3"] == "empty_patch"


def test_load_report_precedence_resolved_beats_error(mod, tmp_path):
    """If an instance_id appears in both error and resolved lists, resolved wins
    (applied last in the precedence loop)."""
    report = tmp_path / "report.json"
    _write_report(report, {
        "error_ids": ["iid-1"],
        "resolved_ids": ["iid-1"],   # same instance in both
    })
    verdicts = mod.load_report(report)
    assert verdicts["iid-1"] == "resolved"


def test_load_report_precedence_resolved_beats_unresolved(mod, tmp_path):
    report = tmp_path / "report.json"
    _write_report(report, {
        "unresolved_ids": ["iid-2"],
        "resolved_ids": ["iid-2"],
    })
    verdicts = mod.load_report(report)
    assert verdicts["iid-2"] == "resolved"


def test_load_report_precedence_resolved_beats_incomplete(mod, tmp_path):
    report = tmp_path / "report.json"
    _write_report(report, {
        "incomplete_ids": ["iid-3"],
        "resolved_ids": ["iid-3"],
    })
    verdicts = mod.load_report(report)
    assert verdicts["iid-3"] == "resolved"


def test_load_report_precedence_resolved_beats_empty_patch(mod, tmp_path):
    report = tmp_path / "report.json"
    _write_report(report, {
        "empty_patch_ids": ["iid-4"],
        "resolved_ids": ["iid-4"],
    })
    verdicts = mod.load_report(report)
    assert verdicts["iid-4"] == "resolved"


def test_load_report_precedence_unresolved_beats_error(mod, tmp_path):
    """Precedence order from the code: incomplete < empty_patch < error <
    unresolved < resolved. So unresolved wins over error."""
    report = tmp_path / "report.json"
    _write_report(report, {
        "error_ids": ["iid-5"],
        "unresolved_ids": ["iid-5"],
    })
    verdicts = mod.load_report(report)
    assert verdicts["iid-5"] == "unresolved"


def test_load_report_precedence_error_beats_empty_patch(mod, tmp_path):
    report = tmp_path / "report.json"
    _write_report(report, {
        "empty_patch_ids": ["iid-6"],
        "error_ids": ["iid-6"],
    })
    verdicts = mod.load_report(report)
    assert verdicts["iid-6"] == "error"


def test_load_report_empty_report(mod, tmp_path):
    report = tmp_path / "report.json"
    _write_report(report, {})
    verdicts = mod.load_report(report)
    assert verdicts == {}


def test_load_report_mixed_key_styles_in_same_file(mod, tmp_path):
    """Some harness versions emit a mix; both should be consumed."""
    report = tmp_path / "report.json"
    _write_report(report, {
        "resolved_ids": ["new__style-1"],
        "unresolved": ["old__style-2"],  # old-style key
    })
    verdicts = mod.load_report(report)
    assert verdicts["new__style-1"] == "resolved"
    assert verdicts["old__style-2"] == "unresolved"


# ---------------------------------------------------------------------------
# find_report
# ---------------------------------------------------------------------------

def test_find_report_skips_config_json(mod, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # config.json must be ignored even if it happens to contain a known key
    _write_report(run_dir / "config.json",
                  {"resolved_ids": ["a__a-1"]})
    result = mod.find_report(run_dir)
    assert result is None


def test_find_report_skips_summary_json(mod, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_report(run_dir / "summary.json",
                  {"resolved_ids": ["a__a-1"]})
    result = mod.find_report(run_dir)
    assert result is None


def test_find_report_picks_report_json(mod, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    report = run_dir / "testbed.run1.json"
    _write_report(report, {"resolved_ids": ["a__a-1"]})
    result = mod.find_report(run_dir)
    assert result == report


def test_find_report_picks_newest_when_multiple(mod, tmp_path):
    """When multiple candidate json files exist, find_report returns the
    newest by mtime."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    old = run_dir / "old.json"
    _write_report(old, {"resolved_ids": ["a__a-1"]})
    # Ensure mtime difference is detectable.
    time.sleep(0.05)
    new = run_dir / "new.json"
    _write_report(new, {"resolved_ids": ["b__b-2"]})

    result = mod.find_report(run_dir)
    assert result == new


def test_find_report_ignores_malformed_json(mod, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "broken.json").write_text("this is not json {{{{")
    result = mod.find_report(run_dir)
    assert result is None


def test_find_report_ignores_json_without_known_keys(mod, tmp_path):
    """A valid JSON file that lacks any harness id-list key is not a report."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "other.json").write_text(json.dumps({"foo": "bar"}))
    result = mod.find_report(run_dir)
    assert result is None


def test_find_report_returns_none_for_empty_dir(mod, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert mod.find_report(run_dir) is None


# ---------------------------------------------------------------------------
# main() integration
# ---------------------------------------------------------------------------

def _make_full_run(tmp_path: Path, records: list[dict],
                   report_data: dict) -> tuple[Path, Path]:
    """Build a run dir with trace.jsonl and a harness report json.
    Returns (run_dir, report_path)."""
    run_dir = tmp_path / "run"
    run_dir.mkdir(exist_ok=True)
    _write_trace(run_dir, records)
    report = run_dir / "testbed.runX.json"
    _write_report(report, report_data)
    return run_dir, report


def test_main_happy_path_prints_table(mod, tmp_path, capsys):
    records = [
        {"instance_id": "a__a-1", "success": True, "rtt_s": 10.5, "error": None},
        {"instance_id": "b__b-2", "success": False, "rtt_s": None,
         "error": {"stage": "clone", "type": "OSError", "msg": "fail"}},
    ]
    run_dir, _ = _make_full_run(tmp_path, records,
                                {"resolved_ids": ["a__a-1"],
                                 "unresolved_ids": ["b__b-2"]})

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_eval_results", "--run", str(run_dir)]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    out = capsys.readouterr().out
    assert "a__a-1" in out
    assert "resolved" in out
    assert "b__b-2" in out
    assert "unresolved" in out


def test_main_not_in_report_verdict(mod, tmp_path, capsys):
    """An instance present in trace but absent from report gets
    verdict=not_in_report."""
    records = [
        {"instance_id": "z__z-99", "success": True, "rtt_s": 5.0, "error": None},
    ]
    run_dir, _ = _make_full_run(tmp_path, records,
                                {"resolved_ids": []})

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_eval_results", "--run", str(run_dir)]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    out = capsys.readouterr().out
    assert "not_in_report" in out


def test_main_resolve_rates_computed(mod, tmp_path, capsys):
    """resolve_rate_all and resolve_rate_http_ok are both printed."""
    records = [
        {"instance_id": "p1", "success": True,  "rtt_s": 5.0, "error": None},
        {"instance_id": "p2", "success": True,  "rtt_s": 6.0, "error": None},
        {"instance_id": "p3", "success": False, "rtt_s": None,
         "error": {"stage": "clone", "type": "E", "msg": ""}},
    ]
    run_dir, _ = _make_full_run(tmp_path, records,
                                {"resolved_ids": ["p1"],
                                 "unresolved_ids": ["p2", "p3"]})

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_eval_results", "--run", str(run_dir)]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    out = capsys.readouterr().out
    # 1 resolved out of 3 total → 33.3%
    assert "resolve_rate_all" in out
    assert "1/3" in out
    # 1 resolved out of 2 http-ok → 50.0%
    assert "resolve_rate_http_ok" in out
    assert "1/2" in out


def test_main_no_http_ok_tasks(mod, tmp_path, capsys):
    """When all tasks failed at the HTTP level, resolve_rate_http_ok prints n/a."""
    records = [
        {"instance_id": "f1", "success": False, "rtt_s": None,
         "error": {"stage": "session"}},
    ]
    run_dir, _ = _make_full_run(tmp_path, records, {"resolved_ids": []})

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_eval_results", "--run", str(run_dir)]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    out = capsys.readouterr().out
    assert "n/a" in out


def test_main_csv_output(mod, tmp_path, capsys):
    """--csv writes per-instance rows with all expected columns."""
    records = [
        {"instance_id": "r1", "success": True,  "rtt_s": 12.3, "error": None},
        {"instance_id": "r2", "success": False, "rtt_s": 2.1,
         "error": {"stage": "message", "type": "TimeoutError", "msg": ""}},
    ]
    run_dir, _ = _make_full_run(tmp_path, records,
                                {"resolved_ids": ["r1"],
                                 "error_ids": ["r2"]})
    csv_out = tmp_path / "out.csv"

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_eval_results", "--run", str(run_dir),
                    "--csv", str(csv_out)]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    assert csv_out.exists()
    rows = list(csv.DictReader(csv_out.open()))
    assert len(rows) == 2
    by_id = {r["instance_id"]: r for r in rows}
    assert by_id["r1"]["verdict"] == "resolved"
    assert by_id["r1"]["http_success"] == "True"
    assert by_id["r2"]["verdict"] == "error"
    assert by_id["r2"]["error_stage"] == "message"


def test_main_explicit_report_flag(mod, tmp_path, capsys):
    """--report overrides auto-discovery of the report file."""
    records = [
        {"instance_id": "q1", "success": True, "rtt_s": 4.0, "error": None},
    ]
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_trace(run_dir, records)
    # Put the real report outside the run dir to confirm auto-discovery is bypassed.
    external_report = tmp_path / "external.json"
    _write_report(external_report, {"resolved_ids": ["q1"]})

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_eval_results", "--run", str(run_dir),
                    "--report", str(external_report)]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 0
    out = capsys.readouterr().out
    assert "resolved" in out


def test_main_returns_1_when_no_report_found(mod, tmp_path, capsys):
    """When no harness report is auto-discoverable and --report is not given,
    main must return exit code 1 and print an error to stderr."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    records = [{"instance_id": "x__x-1", "success": True, "rtt_s": 1.0,
                "error": None}]
    _write_trace(run_dir, records)
    # No report json in run_dir.

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_eval_results", "--run", str(run_dir)]
        rc = mod.main()
    finally:
        sys.argv = saved_argv

    assert rc == 1
    err = capsys.readouterr().err
    assert "no harness report found" in err


def test_main_verdict_counts_in_stdout(mod, tmp_path, capsys):
    """The summary line that lists per-verdict counts is printed."""
    records = [
        {"instance_id": "a", "success": True,  "rtt_s": 1.0, "error": None},
        {"instance_id": "b", "success": True,  "rtt_s": 2.0, "error": None},
        {"instance_id": "c", "success": False, "rtt_s": None,
         "error": {"stage": "clone"}},
    ]
    run_dir, _ = _make_full_run(tmp_path, records,
                                {"resolved_ids": ["a"],
                                 "unresolved_ids": ["b"],
                                 "error_ids": ["c"]})

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_eval_results", "--run", str(run_dir)]
        mod.main()
    finally:
        sys.argv = saved_argv

    out = capsys.readouterr().out
    # "tasks: 3   verdicts: ..."
    assert "tasks: 3" in out
    assert "resolved=1" in out
    assert "unresolved=1" in out
    assert "error=1" in out


def test_main_rtt_displayed_as_dash_when_null(mod, tmp_path, capsys):
    """Tasks with rtt_s=null should show '-' in the table, not 'None'."""
    records = [
        {"instance_id": "no-rtt", "success": False, "rtt_s": None,
         "error": {"stage": "clone"}},
    ]
    run_dir, _ = _make_full_run(tmp_path, records,
                                {"unresolved_ids": ["no-rtt"]})

    saved_argv = sys.argv
    try:
        sys.argv = ["analyze_eval_results", "--run", str(run_dir)]
        mod.main()
    finally:
        sys.argv = saved_argv

    out = capsys.readouterr().out
    # The dash placeholder must appear; raw 'None' must not.
    lines = [l for l in out.splitlines() if "no-rtt" in l]
    assert lines, "expected table row for no-rtt instance"
    assert "-" in lines[0]
    assert "None" not in lines[0]
