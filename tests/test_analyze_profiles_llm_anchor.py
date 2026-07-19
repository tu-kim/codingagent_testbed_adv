"""Verify the canonical per-turn decomposition in analyze_profiles.py
(``_collect_turn_decomposition`` / ``plot_turn_decomposition`` /
``analyze_latency_composition``).

Row shape returned by ``_collect_turn_decomposition``:
    (session_id, step, wall_s, llm_wall_s_stream, tool_wall_s,
     others_s_stream, llm_canon_s, others_canon_s)

  * ``wall_s`` (position 2) = turn.start(N+1).ts - turn.start(N).ts for all
    but the last turn of a session; the last turn falls back to
    turn.end.ts - turn.start.ts, then to turn_duration_s. This is the TRUE
    wall bracket a turn occupies (inter-turn scaffold/queue/prefill time is
    captured instead of dropped) -- it is NOT the old turn.end duration_s.
  * ``llm_canon_s`` (position 6) prefers ``llm.end.dynamo.elapsed_s``
    (server HTTP receipt -> last chunk: queue wait + prefill + decode);
    falling back to the wall-clock-anchored ``llm.end.ts - turn.start.ts -
    tool_wall_s`` when dynamo timing is absent; falling back further to the
    stream-based ``llm_wall_s`` when even the anchor timestamps are
    missing. It is therefore ALWAYS populated (no more `None` fallback).
  * ``others_canon_s`` (position 7) = ``max(0, wall - llm_canon - tool)``.
    The identity ``llm_canon + tool + others_canon == wall`` holds exactly
    whenever the clamp doesn't fire (checked below) -- the old identity
    against ``duration_s`` no longer applies in general because ``wall`` is
    no longer defined as ``llm + tool + post``.

Original hypothesis this file still protects (BUFFERED vs STREAMED turns):
the profiler's stream-event ``llm_wall_s`` (start-step -> firstTool/lastText)
collapses on a BUFFERED large tool-call turn, because ``start-step`` /
``llm.start`` fires only when opencode begins CONSUMING the response -- for
a big edit tool-call the whole response arrives at the END of generation,
so ``llm.start`` is late and the real LLM time is hidden. The anchored
correction (``llm.end.ts - turn.start.ts - tool_wall``) recovers it. A
STREAMED turn does not have this problem, so the "amount recovered"
(``llm_canon - llm_wall_s(stream)``) is still the discriminator between the
two regimes -- that part of the arithmetic is unaffected by the wall-bracket
redefinition and is asserted unchanged.

Pure mock: synthetic profile NDJSON, no network / GPU. matplotlib is forced
to the Agg backend and import-skipped if absent.
"""

from __future__ import annotations

import csv
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
# Numbers mirror the real trace: step 1 (buffered, ~75s hidden in others) and
# step 2 (streamed, measured correctly), both 1000s apart in turn_start_ts
# (that spacing was chosen only to keep the llm.end - turn.start anchor
# calculation unambiguous -- it now ALSO becomes the turn's `wall_s` via the
# next-turn-start rule, which is why wall/others below are large; see the
# hand-traced arithmetic in each test). Step 3 omits llm.end to exercise the
# final fallback-to-stream-value rung of the llm_canon chain.

SID = "ses_mocktest"

# (kind, step, turn_start_ts, llm_end_ts_or_None, tool_wall, dur, stream_llm_wall, post_overhead)
_TURNS = [
    # buffered large edit: llm.start fires ~75s late -> stream llm_wall ~= 0
    ("buffered", 1, 1000.000, 1075.433, 0.016, 75.535, 0.007, 75.512),
    # streamed: llm.start early -> stream llm_wall correct
    ("streamed", 2, 2000.000, 2007.231, 0.011, 7.331, 6.583, 0.737),
    # no llm.end -> llm_canon falls back all the way to the stream value
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
# (sid, step, wall_s, llm_wall_s_stream, tool_wall_s, others_s_stream,
#  llm_canon_s, others_canon_s)
_W, _LW, _TW, _PO, _LC, _OC = 2, 3, 4, 5, 6, 7


def test_all_three_turns_collected(rows):
    assert set(rows) == {1, 2, 3}


def test_wall_is_next_turn_start_gap(rows):
    """wall_s = turn.start(N+1) - turn.start(N) for non-last turns; the last
    turn falls back to turn.end.ts - turn.start.ts."""
    # step1 -> step2 gap: 2000.000 - 1000.000
    assert rows[1][_W] == pytest.approx(2000.000 - 1000.000, abs=1e-6)
    # step2 -> step3 gap: 3000.000 - 2000.000
    assert rows[2][_W] == pytest.approx(3000.000 - 2000.000, abs=1e-6)
    # step3 is last: turn.end.ts (=turn_start + dur) - turn.start.ts = dur
    assert rows[3][_W] == pytest.approx(5.000, abs=1e-6)


def test_buffered_turn_recovers_hidden_llm_time(rows):
    r = rows[1]
    # stream-based split badly under-measures LLM, dumps it into "others"
    # (legacy stream fields are untouched by the wall/others redefinition)
    assert r[_LW] == pytest.approx(0.007, abs=1e-6)
    assert r[_PO] == pytest.approx(75.512, abs=1e-6)
    # canonical: llm_canon = llm.end.ts - turn.start.ts - tool_wall
    assert r[_LC] == pytest.approx(1075.433 - 1000.000 - 0.016, abs=1e-6)  # 75.417
    # canonical others = wall - llm_canon - tool; wall here is the 1000s
    # next-turn-start gap, so others_canon absorbs the remainder of that
    # gap (inter-turn idle/scaffold time), not just the old ~0.1s slice.
    wall = 2000.000 - 1000.000
    assert r[_OC] == pytest.approx(wall - r[_LC] - 0.016, abs=1e-6)  # 924.567
    # the correction still moved ~75s from the stream "others" into LLM
    assert r[_LC] > 75.0
    assert r[_LC] - r[_LW] > 50.0


def test_streamed_turn_essentially_unchanged_llm(rows):
    r = rows[2]
    # canonical llm_canon = 2007.231 - 2000.000 - 0.011 = 7.220
    assert r[_LC] == pytest.approx(7.220, abs=1e-6)
    # differs from the stream-based llm_wall only by the small
    # turn.start->llm.start gap (here ~0.64s), NOT by tens of seconds
    assert abs(r[_LC] - r[_LW]) < 1.0
    wall = 3000.000 - 2000.000
    assert r[_OC] == pytest.approx(wall - r[_LC] - 0.011, abs=1e-6)  # 992.769


def test_recovery_is_the_discriminator(rows):
    """The whole point: buffered turns hide huge LLM time in the stream
    'others' bucket relative to the anchored llm_canon, streamed turns do
    not. This delta is independent of the wall-bracket redefinition."""
    recovered_buffered = rows[1][_LC] - rows[1][_LW]
    recovered_streamed = rows[2][_LC] - rows[2][_LW]
    assert recovered_buffered > 50.0
    assert recovered_streamed < 1.0
    assert recovered_buffered > 50 * recovered_streamed


def test_canonical_identity_holds_when_unclamped(rows):
    """llm_canon + tool_wall + others_canon == wall for every turn here
    (none of them hit the max(0, ...) clamp)."""
    for step in (1, 2, 3):
        r = rows[step]
        assert r[_LC] + r[_TW] + r[_OC] == pytest.approx(r[_W], abs=1e-6)


def test_exact_arithmetic_matches_definition(rows):
    for step, tstart, lend, tw in ((1, 1000.000, 1075.433, 0.016),
                                   (2, 2000.000, 2007.231, 0.011)):
        r = rows[step]
        assert r[_LC] == pytest.approx(lend - tstart - tw, abs=1e-6)
        assert r[_OC] == pytest.approx(r[_W] - r[_LC] - tw, abs=1e-6)


def test_missing_llm_end_falls_back_to_stream_value(rows):
    """No llm.end and no dynamo timing -> llm_canon falls all the way back
    to the stream-based llm_wall_s (it is NEVER None any more)."""
    r = rows[3]
    assert r[_LC] == pytest.approx(4.000, abs=1e-6)
    assert r[_LC] == pytest.approx(r[_LW], abs=1e-6)
    # wall is the turn.end fallback (turn_start_ts absent a next turn and
    # turn_end_ts = turn_start + dur = 3005.000) -> wall == dur == 5.000
    assert r[_W] == pytest.approx(5.000, abs=1e-6)
    assert r[_OC] == pytest.approx(5.000 - 4.000 - 0.500, abs=1e-6)  # 0.500


def test_dynamo_elapsed_s_wins_over_anchored_fallback(mod, tmp_path):
    """When llm.end carries dynamo.elapsed_s, llm_canon uses THAT value,
    not the wall-clock-anchored (llm.end.ts - turn.start.ts - tool) figure
    -- even though the anchored figure is available and would give a very
    different number."""
    sid = "ses_dynamo_test"
    tstart, lend, tw, dyn_elapsed = 5000.0, 5010.0, 0.02, 3.5
    dur, lw, po = 12.0, 9.5, 2.48
    lines = [
        {"ev": "turn.start", "sessionID": sid, "step": 1, "ts": tstart},
        {"ev": "tool.end", "sessionID": sid, "step": 1, "name": "edit",
         "ok": True, "duration_s": tw, "output_chars": 5},
        {"ev": "llm.end", "sessionID": sid, "step": 1, "ts": lend,
         "finish": "tool-calls", "step_duration_s": 0.25,
         "dynamo": {"elapsed_s": dyn_elapsed},
         "tokens": {"total": 100, "input": 80, "output": 20,
                    "reasoning": 0, "cache": {"write": 0, "read": 0}}},
        {"ev": "turn.end", "sessionID": sid, "step": 1, "ts": tstart + dur,
         "duration_s": dur, "llm_wall_s": lw, "tool_wall_s": tw,
         "post_overhead_s": po},
    ]
    p = tmp_path / "dynamo_profiles.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")

    sessions = mod.load_sessions(p)
    out = mod._collect_turn_decomposition(sessions)
    assert len(out) == 1
    r = out[0]

    # sanity: the anchored (non-dynamo) figure WOULD have been very
    # different, so this is a real discriminator, not a coincidence.
    anchored_would_be = lend - tstart - tw  # 9.98
    assert anchored_would_be != pytest.approx(dyn_elapsed, abs=0.1)

    assert r[_LC] == pytest.approx(dyn_elapsed, abs=1e-6)  # 3.5, not 9.98
    # single-turn session: wall falls back to turn.end.ts - turn.start.ts
    wall = dur
    assert r[_W] == pytest.approx(wall, abs=1e-6)
    assert r[_OC] == pytest.approx(wall - dyn_elapsed - tw, abs=1e-6)  # 8.48


def test_per_turn_csv_has_canonical_columns(mod, tmp_path):
    """plot_turn_decomposition emits llm_canon_s / others_canon_s columns
    (renamed from llm_wall_true_s / others_true_s) with the recovered
    values. llm_canon_s is populated for EVERY turn now, including the
    fallback turn (no more blank cells)."""
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

    assert "llm_canon_s" in recs[1]
    assert "others_canon_s" in recs[1]
    assert float(recs[1]["llm_canon_s"]) == pytest.approx(75.417, abs=1e-3)
    assert float(recs[2]["llm_canon_s"]) == pytest.approx(7.220, abs=1e-3)
    # fallback turn: llm_canon_s is now populated (fell back to stream lw),
    # not blank.
    assert float(recs[3]["llm_canon_s"]) == pytest.approx(4.000, abs=1e-3)
    assert float(recs[3]["others_canon_s"]) == pytest.approx(0.500, abs=1e-3)


def test_turn_share_distribution_figure_created(mod, tmp_path):
    """plot_turn_decomposition also emits the overlaid per-turn share
    distribution figure."""
    p = tmp_path / "profiles.jsonl"
    _write_profiles(p)
    sessions = mod.load_sessions(p)
    outdir = tmp_path / "figs"
    outdir.mkdir()
    mod.plot_turn_decomposition(sessions, outdir)
    fig = outdir / "fig6b_turn_share_distribution.png"
    assert fig.exists() and fig.stat().st_size > 0


# --------------------------------------------------------------------------
# analyze_latency_composition (merged from the former analyze_latency_breakdown)
# --------------------------------------------------------------------------
#
# analyze_latency_composition consumes the SAME canonical rows as
# _collect_turn_decomposition (dur == wall_s at position 2, llm == llm_canon_s
# at position 6 with stream fallback, others == others_canon_s at position 7
# with stream fallback). With the new wall-bracket definition:
#
#   step1 (buffered):  wall 1000.000  llm  75.417  tool 0.016  others 924.567
#   step2 (streamed):  wall 1000.000  llm   7.220  tool 0.011  others 992.769
#   step3 (no llm.end):wall    5.000  llm   4.000  tool 0.500  others   0.500
#
# Sums: Σwall=2005.000  Σllm=86.637  Σtool=0.527  Σothers=1917.836
#
# Because two of the three wall values are equal (1000.000) and dominate the
# quantile computation, np.quantile([5,1000,1000], (0.5,0.9,0.99)) all
# collapse to 1000.0 -- every turn (including the 5.000s one) falls in the
# "<=p50" bucket, so the other three buckets are empty. That's a genuine
# artifact of this fixture's turn spacing under the new wall semantics, not
# a bug; it's asserted explicitly below.


def _run_latency(mod, tmp_path):
    p = tmp_path / "profiles.jsonl"
    _write_profiles(p)
    sessions = mod.load_sessions(p)
    outdir = tmp_path / "lat"
    outdir.mkdir()
    ret = mod.analyze_latency_composition(sessions, outdir)
    assert ret == outdir
    return outdir


def test_latency_outputs_created(mod, tmp_path):
    outdir = _run_latency(mod, tmp_path)
    for name in ("latency_pooled_share.csv", "latency_per_request_share.csv",
                 "latency_conditional_by_bucket.csv", "latency_share_violin.png",
                 "latency_sorted_stacked.png", "latency_bucket_stacked.png"):
        f = outdir / name
        assert f.exists() and f.stat().st_size > 0, name


def test_latency_pooled_share_uses_canonical_components(mod, tmp_path):
    outdir = _run_latency(mod, tmp_path)
    with (outdir / "latency_pooled_share.csv").open() as f:
        recs = {row["component"]: row for row in csv.DictReader(f)}
    grand = 1000.000 + 1000.000 + 5.000  # 2005.000
    llm_sum = 75.417 + 7.220 + 4.000       # 86.637
    tool_sum = 0.016 + 0.011 + 0.500        # 0.527
    others_sum = 924.567 + 992.769 + 0.500  # 1917.836
    assert float(recs["llm_wall_s"]["pooled_share"]) == pytest.approx(llm_sum / grand, abs=1e-4)
    assert float(recs["tool_wall_s"]["pooled_share"]) == pytest.approx(tool_sum / grand, abs=1e-4)
    assert float(recs["others"]["pooled_share"]) == pytest.approx(others_sum / grand, abs=1e-4)
    # three component shares sum to 1
    tot = sum(float(recs[c]["pooled_share"]) for c in ("llm_wall_s", "tool_wall_s", "others"))
    assert tot == pytest.approx(1.0, abs=1e-3)


def test_latency_per_request_table_structure(mod, tmp_path):
    outdir = _run_latency(mod, tmp_path)
    with (outdir / "latency_per_request_share.csv").open() as f:
        recs = list(csv.DictReader(f))
    assert [r["component"] for r in recs] == ["llm_wall_s", "tool_wall_s", "others"]
    for r in recs:
        assert int(r["n_requests"]) == 3
        assert 0.0 <= float(r["mean"]) <= 1.0


def test_latency_buckets_all_collapse_into_le_p50(mod, tmp_path):
    """With wall values [1000.000, 1000.000, 5.000], np.quantile at
    0.5/0.9/0.99 all evaluate to 1000.000 (the two duplicated max values
    dominate), so every turn's wall <= p50 and lands in the same bucket."""
    outdir = _run_latency(mod, tmp_path)
    with (outdir / "latency_conditional_by_bucket.csv").open() as f:
        recs = {row["bucket"]: row for row in csv.DictReader(f)}
    assert int(recs["<=p50"]["n_requests"]) == 3
    assert int(recs["p50-p90"]["n_requests"]) == 0
    assert int(recs["p90-p99"]["n_requests"]) == 0
    assert int(recs[">p99"]["n_requests"]) == 0
    # mean shares in the sole populated bucket sum to ~1
    tot = sum(float(recs["<=p50"][f"{c}_mean_share"])
              for c in ("llm_wall_s", "tool_wall_s", "others"))
    assert tot == pytest.approx(1.0, abs=1e-3)
    # the "others" component (dominated by the huge inter-turn wall gaps
    # of the two 1000s turns) is now the overwhelming majority share
    assert float(recs["<=p50"]["others_mean_share"]) > 0.5


# --------------------------------------------------------------------------
# analyze_tool_dominated_turns
# --------------------------------------------------------------------------

_TSID = "ses_tooltest"


_BASH_CMD = "python -m pytest tests/ -x"


def _write_tool_heavy(path: Path) -> None:
    """Two turns: a bash-dominated one (tool_wall 9/10) and a normal read turn.
    tool.start carries the command preview (args_head) + callID; tool.end matches
    by callID."""
    turns = [
        # step, tool, dur, out_chars, callID, args_head, llm_end_ts, in, out, cache,
        #   turn_start, dur, llm_wall, tool_wall, post
        (1, "bash", 9.0, 500, "c1", json.dumps({"command": _BASH_CMD, "description": "run"}),
         1000.5, 5000, 200, 1000, 1000.0, 10.0, 0.5, 9.0, 0.5),
        (2, "read", 0.1, 50, "c2", json.dumps({"filePath": "/repo/foo.py"}),
         2004.0, 2000, 800, 0, 2000.0, 5.0, 4.0, 0.1, 0.9),
    ]
    lines = []
    for (step, tname, tdur, oc, cid, ah, lend, itk, otk, cr,
         tstart, dur, lw, tw, po) in turns:
        lines.append({"ev": "turn.start", "sessionID": _TSID, "step": step, "ts": tstart})
        lines.append({"ev": "tool.start", "sessionID": _TSID, "step": step,
                      "callID": cid, "name": tname, "kind": "builtin", "args_head": ah})
        lines.append({"ev": "tool.end", "sessionID": _TSID, "step": step, "callID": cid,
                      "name": tname, "ok": True, "duration_s": tdur, "output_chars": oc})
        lines.append({"ev": "llm.end", "sessionID": _TSID, "step": step, "ts": lend,
                      "finish": "tool-calls", "step_duration_s": 0.2,
                      "tokens": {"total": itk + otk, "input": itk, "output": otk,
                                 "reasoning": 0, "cache": {"write": 0, "read": cr}}})
        lines.append({"ev": "turn.end", "sessionID": _TSID, "step": step, "ts": tstart + dur,
                      "duration_s": dur, "llm_wall_s": lw, "tool_wall_s": tw,
                      "post_overhead_s": po})
    with path.open("w", encoding="utf-8") as f:
        for obj in lines:
            f.write(json.dumps(obj) + "\n")


def _run_tool_dominated(mod, tmp_path):
    p = tmp_path / "profiles.jsonl"
    _write_tool_heavy(p)
    sessions = mod.load_sessions(p)
    outdir = tmp_path / "td"
    outdir.mkdir()
    csv_path = mod.analyze_tool_dominated_turns(sessions, outdir, top_n=20)
    assert csv_path == outdir / "tool_dominated_turns.csv"
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def test_tool_dominated_ranks_by_share(mod, tmp_path):
    recs = _run_tool_dominated(mod, tmp_path)
    assert len(recs) == 2
    # sorted by tool_share desc -> bash turn (0.9) first, read turn (0.02) second
    assert recs[0]["dominant_tool"] == "bash"
    assert float(recs[0]["tool_share"]) == pytest.approx(0.9, abs=1e-6)
    assert recs[1]["dominant_tool"] == "read"
    assert float(recs[1]["tool_share"]) == pytest.approx(0.02, abs=1e-6)


def test_tool_dominated_reports_tokens_and_breakdown(mod, tmp_path):
    recs = _run_tool_dominated(mod, tmp_path)
    bash = recs[0]
    assert int(bash["input_tokens"]) == 5000
    assert int(bash["output_tokens"]) == 200
    assert int(bash["cache_read"]) == 1000
    assert int(bash["n_tools"]) == 1
    assert int(bash["tool_output_chars"]) == 500
    assert float(bash["dominant_tool_s"]) == pytest.approx(9.0, abs=1e-6)
    assert bash["tools"] == "bash:9.000"
    # the actual command run under bash is surfaced from tool.start.args_head
    assert bash["dominant_tool_cmd"] == _BASH_CMD


def test_tool_dominated_command_extracted_from_args_head(mod, tmp_path):
    recs = _run_tool_dominated(mod, tmp_path)
    by_tool = {r["dominant_tool"]: r for r in recs}
    assert by_tool["bash"]["dominant_tool_cmd"] == _BASH_CMD
    # the read turn's file path is pulled from the filePath key
    assert by_tool["read"]["dominant_tool_cmd"] == "/repo/foo.py"
