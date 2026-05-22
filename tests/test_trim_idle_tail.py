"""Tests for scripts/trim_idle_tail.py.

Pure file-based filtering on tmp_path. No network or external deps.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "trim_idle_tail.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("trim_idle_tail", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["trim_idle_tail"] = module
    spec.loader.exec_module(module)
    return module


def _write_profile(path: Path, sessions: list[tuple[str, float, float]]) -> None:
    """sessions = [(sid, start_ts, end_ts), ...]. Writes ONE NDJSON file
    per session under path/."""
    path.mkdir(exist_ok=True)
    for sid, s, e in sessions:
        with (path / f"{sid}.jsonl").open("w") as f:
            f.write(json.dumps({"ev": "query.start", "ts": s, "sessionID": sid}) + "\n")
            f.write(json.dumps({"ev": "turn.start", "ts": s + 1, "sessionID": sid, "step": 1}) + "\n")
            f.write(json.dumps({"ev": "query.end", "ts": e, "sessionID": sid,
                                 "duration_s": e - s, "steps": 1}) + "\n")


def _write_ndjson(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


# ---------- window detection ----------


def test_detect_window_spans_all_sessions(mod, tmp_path):
    pdir = tmp_path / "profiles"
    _write_profile(pdir, [
        ("ses_a", 100.0, 200.0),
        ("ses_b", 150.0, 250.0),
        ("ses_c",  50.0, 120.0),  # earliest start, latest end shouldn't be this
    ])
    start, end = mod.detect_window(pdir)
    assert start == 50.0
    assert end == 250.0


def test_detect_window_falls_back_to_last_seen_when_query_end_missing(mod, tmp_path):
    """A mid-flight session whose monitor was killed has query.start but
    no query.end. Use the last ts seen on that session as the end."""
    pdir = tmp_path / "profiles"
    pdir.mkdir()
    with (pdir / "ses_x.jsonl").open("w") as f:
        f.write(json.dumps({"ev": "query.start", "ts": 10.0, "sessionID": "ses_x"}) + "\n")
        f.write(json.dumps({"ev": "turn.start",  "ts": 12.0, "sessionID": "ses_x", "step": 1}) + "\n")
        f.write(json.dumps({"ev": "llm.start",   "ts": 15.0, "sessionID": "ses_x", "step": 1}) + "\n")
        # no query.end -- treat 15.0 as end
    start, end = mod.detect_window(pdir)
    assert start == 10.0
    assert end == 15.0


def test_detect_window_errors_when_no_sessions(mod, tmp_path):
    pdir = tmp_path / "profiles"
    pdir.mkdir()
    with pytest.raises(SystemExit, match="no query.start"):
        mod.detect_window(pdir)


# ---------- trim_file ----------


def test_trim_file_keeps_rows_in_window_drops_idle_tail(mod, tmp_path):
    target = tmp_path / "resource.ndjson"
    _write_ndjson(target, [
        {"ts":  50.0, "host": {"cpu": 5}},   # before window — drop
        {"ts": 100.0, "host": {"cpu": 30}},  # inside
        {"ts": 150.0, "host": {"cpu": 50}},  # inside
        {"ts": 200.0, "host": {"cpu": 60}},  # at end — inside (inclusive)
        {"ts": 300.0, "host": {"cpu": 10}},  # after window — drop (the "dummy tail")
        {"ts": 400.0, "host": {"cpu":  8}},  # also after
    ])
    kept, total, out = mod.trim_file(target, 100.0, 200.0, in_place=False)
    assert total == 6
    assert kept == 3
    rows = [json.loads(l) for l in out.read_text().splitlines() if l]
    assert [r["ts"] for r in rows] == [100.0, 150.0, 200.0]


def test_trim_file_output_path_default_appends_trimmed(mod, tmp_path):
    target = tmp_path / "resource.ndjson"
    _write_ndjson(target, [{"ts": 100.0, "host": {}}])
    _, _, out = mod.trim_file(target, 50.0, 200.0, in_place=False)
    assert out.name == "resource.trimmed.ndjson"
    assert out.parent == target.parent
    # Original is untouched
    assert target.exists()


def test_trim_file_in_place_saves_bak(mod, tmp_path):
    target = tmp_path / "resource.ndjson"
    _write_ndjson(target, [
        {"ts": 100.0, "host": {"cpu": 30}},
        {"ts": 999.0, "host": {"cpu":  8}},
    ])
    original_bytes = target.read_bytes()
    kept, total, out = mod.trim_file(target, 50.0, 200.0, in_place=True)
    assert out == target  # overwritten
    assert kept == 1
    bak = target.with_suffix(target.suffix + ".bak")
    assert bak.exists()
    assert bak.read_bytes() == original_bytes
    # Trimmed content
    rows = [json.loads(l) for l in target.read_text().splitlines() if l]
    assert len(rows) == 1
    assert rows[0]["ts"] == 100.0


def test_trim_file_skips_rows_without_ts(mod, tmp_path):
    """Defensive: a row with no `ts` field (shouldn't happen, but
    monitor/scrape might evolve) is dropped silently rather than
    bouncing into either the kept or excluded bucket arbitrarily."""
    target = tmp_path / "data.ndjson"
    _write_ndjson(target, [
        {"ts": 100.0, "x": 1},
        {"x": 2},               # no ts
        {"ts": 150.0, "x": 3},
    ])
    kept, total, out = mod.trim_file(target, 50.0, 200.0, in_place=False)
    assert total == 3
    assert kept == 2
    rows = [json.loads(l) for l in out.read_text().splitlines() if l]
    assert [r["ts"] for r in rows] == [100.0, 150.0]


def test_trim_file_preserves_malformed_json_lines(mod, tmp_path):
    """A truncated last line shouldn't be silently dropped by trim;
    we don't pretend to know its ts, so keep it for the user to
    inspect rather than masking the corruption."""
    target = tmp_path / "data.ndjson"
    target.write_text(
        json.dumps({"ts": 100.0, "x": 1}) + "\n"
        + "garbage line not json\n"
        + json.dumps({"ts": 150.0, "x": 2}) + "\n"
    )
    kept, total, out = mod.trim_file(target, 50.0, 200.0, in_place=False)
    lines = out.read_text().splitlines()
    assert len(lines) == 3   # 2 in-window + 1 malformed kept
    assert "garbage line not json" in out.read_text()


# ---------- main end-to-end ----------


def test_main_with_profile_dir_trims_both_targets(mod, tmp_path, capsys):
    pdir = tmp_path / "profiles"
    _write_profile(pdir, [("ses_a", 100.0, 200.0)])
    res = tmp_path / "resource.ndjson"
    vllm = tmp_path / "vllm_metrics.ndjson"
    _write_ndjson(res, [
        {"ts":  90.0}, {"ts": 100.0}, {"ts": 200.0}, {"ts": 300.0},
    ])
    _write_ndjson(vllm, [
        {"ts":  95.0}, {"ts": 150.0}, {"ts": 250.0},
    ])
    rc = mod.main([
        "--profile-dir", str(pdir),
        "--target", str(res),
        "--target", str(vllm),
    ])
    assert rc == 0
    res_trimmed = res.parent / "resource.trimmed.ndjson"
    vllm_trimmed = vllm.parent / "vllm_metrics.trimmed.ndjson"
    res_rows = [json.loads(l) for l in res_trimmed.read_text().splitlines() if l]
    vllm_rows = [json.loads(l) for l in vllm_trimmed.read_text().splitlines() if l]
    assert [r["ts"] for r in res_rows] == [100.0, 200.0]
    assert [r["ts"] for r in vllm_rows] == [150.0]
    captured = capsys.readouterr().out
    assert "window from profile dir" in captured


def test_main_with_explicit_window(mod, tmp_path, capsys):
    res = tmp_path / "resource.ndjson"
    _write_ndjson(res, [{"ts": 100.0}, {"ts": 250.0}, {"ts": 400.0}])
    rc = mod.main([
        "--window", "200.0", "300.0",
        "--target", str(res),
    ])
    assert rc == 0
    rows = [json.loads(l) for l in
            (tmp_path / "resource.trimmed.ndjson").read_text().splitlines() if l]
    assert [r["ts"] for r in rows] == [250.0]
