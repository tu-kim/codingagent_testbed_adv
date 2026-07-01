"""Verify the wall-clock-anchored LLM decomposition in analyze_profiles.py.

Hypothesis under test (observed on ses_1018bbe7...): the profiler's
stream-event ``llm_wall_s`` (start-step -> firstTool/lastText) collapses on a
BUFFERED large tool-call turn, because ``start-step``/``llm.start`` fires only
when opencode begins CONSUMING the response -- for a big edit tool-call the
whole response arrives at the END of generation, so ``llm.start`` is late and
the real LLM time lands in the ``turn.start -> llm.start`` gap, misattributed
to ``post_overhead`` ("others").  A STREAMED turn does not have this problem.

The correction ``llm_wall_true_s = (llm.end.ts - turn.start.ts) - tool_wall``
must:
  * recover the hidden LLM time on the buffered turn (llm_true >> stream
    llm_wall_s; others_true << others_s), and
  * leave a streamed turn essentially unchanged (llm_true ~= stream llm_wall_s),

so the amount recovered is the discriminator between the two regimes.

Pure mock: synthetic profile NDJSON, no network / GPU.  matplotlib is forced
to the Agg backend and import-skipped if absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless: force a non-interactive backend before pyplot loads

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_profiles.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_profiles", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_profiles"] = module
    spec.loader.exec_module(module)
    return module


# --- synthetic turns -------------------------------------------------------
#
# Numbers mirror the real trace: step 20 (buffered, ~75s hidden in others) and
# step 17 (streamed, measured correctly).  A third turn omits llm.end to
# exercise the None fallback.

SID = "ses_mocktest"

# (step, kind, turn_start_ts, llm_end_ts_or_None, tool_wall, dur, stream_llm_wall, post_overhead)
_TURNS = [
    # buffered large edit: llm.start fires ~75s late -> stream llm_wall ~= 0
    ("buffered", 1, 1000.000, 1075.433, 0.016, 75.535, 0.007, 75.512),
    # streamed: llm.start early -> stream llm_wall correct
    ("streamed", 2, 2000.000, 2007.231, 0.011, 7.331, 6.583, 0.737),
    # no llm.end -> corrected fields fall back to None
    ("no_llm_end", 3, 3000.000, None, 0.500, 5.000, 4.000, 0.500),
]


def _write_profiles(path: Path) -> None:
    lines = []
    for _kind, step, tstart, lend, tw, dur, lw, po in _TURNS:
        lines.append({"ev": "turn.start", "sessionID": SID, "step": step, "ts": tstart})
        # a single edit tool call (name != "task", so the turn is kept)
        lines.append({
            "ev": "tool.end", "sessionID": SID, "step": step,
            "name": "edit", "ok": True, "duration_s": tw, "output_chars": 10,
        })
        if lend is not None:
            lines.append({
                "ev": "llm.end", "sessionID": SID, "step": step, "ts": lend,
                "finish": "tool-calls", "step_duration_s": 0.25,
                "tokens": {"total": 1000, "input": 1722, "output": 732,
                           "reasoning": 0, "cache": {"write": 0, "read": 35344}},
            })
        lines.append({
            "ev": "turn.end", "sessionID": SID, "step": step, "ts": tstart + dur,
            "duration_s": dur, "llm_wall_s": lw, "tool_wall_s": tw,
            "post_overhead_s": po,
        })
    with path.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


@pytest.fixture()
def rows(mod, tmp_path):
    p = tmp_path / "profiles.jsonl"
    _write_profiles(p)
    sessions = mod.load_sessions(p)
    out = mod._collect_turn_decomposition(sessions)
    # index by step for readable assertions
    return {r[1]: r for r in out}


# column layout of a decomposition row:
# (sid, step, duration, llm_wall_stream, tool_wall, others_stream,
#  llm_wall_true, others_true)
_D, _LW, _TW, _PO, _LT, _OT = 2, 3, 4, 5, 6, 7


def test_all_three_turns_collected(rows):
    assert set(rows) == {1, 2, 3}


def test_buffered_turn_recovers_hidden_llm_time(rows):
    r = rows[1]
    # stream-based split badly under-measures LLM, dumps it into "others"
    assert r[_LW] == pytest.approx(0.007, abs=1e-6)
    assert r[_PO] == pytest.approx(75.512, abs=1e-6)
    # corrected: llm_true = llm.end.ts - turn.start.ts - tool_wall
    assert r[_LT] == pytest.approx(1075.433 - 1000.000 - 0.016, abs=1e-6)  # 75.417
    assert r[_OT] == pytest.approx(75.535 - 75.417 - 0.016, abs=1e-3)      # 0.102
    # the correction moved ~75s from others into LLM
    assert r[_LT] > 75.0
    assert r[_OT] < 0.2


def test_streamed_turn_essentially_unchanged(rows):
    r = rows[2]
    # corrected llm_true = 2007.231 - 2000.000 - 0.011 = 7.220
    assert r[_LT] == pytest.approx(7.220, abs=1e-6)
    # differs from the stream-based llm_wall only by the small
    # turn.start->llm.start gap (here ~0.64s), NOT by tens of seconds
    assert abs(r[_LT] - r[_LW]) < 1.0
    assert r[_OT] == pytest.approx(0.100, abs=1e-3)


def test_recovery_is_the_discriminator(rows):
    """The whole point: buffered turns hide huge LLM time in 'others',
    streamed turns do not."""
    recovered_buffered = rows[1][_LT] - rows[1][_LW]
    recovered_streamed = rows[2][_LT] - rows[2][_LW]
    assert recovered_buffered > 50.0
    assert recovered_streamed < 1.0
    assert recovered_buffered > 50 * recovered_streamed


def test_corrected_decomposition_is_complete(rows):
    """llm_true + tool_wall + others_true == duration (no time lost or
    double-counted) for every turn that carries timestamps."""
    for step in (1, 2):
        r = rows[step]
        assert r[_LT] + r[_TW] + r[_OT] == pytest.approx(r[_D], abs=1e-6)


def test_exact_arithmetic_matches_definition(rows):
    for step, tstart, lend, tw in ((1, 1000.000, 1075.433, 0.016),
                                   (2, 2000.000, 2007.231, 0.011)):
        r = rows[step]
        assert r[_LT] == pytest.approx(lend - tstart - tw, abs=1e-6)
        assert r[_OT] == pytest.approx(r[_D] - r[_LT] - tw, abs=1e-6)


def test_missing_llm_end_falls_back_to_none(rows):
    r = rows[3]
    assert r[_LT] is None
    assert r[_OT] is None
    # stream-based fields still present
    assert r[_LW] == pytest.approx(4.000, abs=1e-6)


def test_per_turn_csv_has_corrected_columns(mod, tmp_path):
    """plot_turn_decomposition emits llm_wall_true_s / others_true_s columns
    with the recovered values, and blanks for the fallback turn."""
    import csv

    p = tmp_path / "profiles.jsonl"
    _write_profiles(p)
    sessions = mod.load_sessions(p)
    outdir = tmp_path / "figs"
    outdir.mkdir()
    mod.plot_turn_decomposition(sessions, outdir)

    csv_path = outdir / "fig6_turn_decomposition_per_turn.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        recs = {int(row["step"]): row for row in csv.DictReader(f)}

    assert "llm_wall_true_s" in recs[1]
    assert "others_true_s" in recs[1]
    assert float(recs[1]["llm_wall_true_s"]) == pytest.approx(75.417, abs=1e-3)
    assert float(recs[2]["llm_wall_true_s"]) == pytest.approx(7.220, abs=1e-3)
    # fallback turn: blank corrected columns
    assert recs[3]["llm_wall_true_s"] == ""
    assert recs[3]["others_true_s"] == ""
