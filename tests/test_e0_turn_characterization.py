"""Tests for scripts/arm/e0_turn_characterization.py (data functions only;
matplotlib paths are not exercised).

Covers: chronological ordering + session-boundary detection, bottom-
percentile subsetting with prev/cur tool distribution + %, hit/KV/gap
series extraction, and main() CSV outputs with --no-figures.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = (Path(__file__).resolve().parents[1]
                / "scripts" / "arm" / "e0_turn_characterization.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("e0_turn_characterization", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["e0_turn_characterization"] = module
    spec.loader.exec_module(module)
    return module


# minimal stand-in matching the TurnRec attributes the script reads.
# hit_tokens=None means "no usage data" (input_tokens stays None).
class _T:
    def __init__(self, session_id, step, start, end, wall, prev, cur,
                 gap=None, hit_tokens=None):
        self.session_id = session_id
        self.step = step
        self.llm_start_ts = start
        self.llm_end_ts = end
        self.llm_wall_s = wall
        self.prev_tools = tuple(prev)
        self.tool_names = list(cur)
        self.away_s = gap
        self.input_tokens = None if hit_tokens is None else 100
        self.cache_read = hit_tokens or 0
        self.output_tokens = 0

    @property
    def effective_input(self):
        if self.input_tokens is None:
            return None
        return self.input_tokens + self.cache_read

    @property
    def prev_key(self):
        return "+".join(sorted(set(self.prev_tools))) if self.prev_tools else "(none)"


# ---------- ordering / boundaries ----------


def test_order_turns_by_start_ts(mod):
    turns = [_T("B", 1, 20.0, 23.0, 3.0, [], ["grep"]),
             _T("A", 1, 0.0, 5.0, 5.0, [], ["read"]),
             _T("A", 2, 7.0, 7.5, 0.5, ["read"], ["bash"])]
    ordered = mod.order_turns(turns)
    assert [t.session_id for t in ordered] == ["A", "A", "B"]


def test_session_boundaries(mod):
    ordered = [_T("A", 1, 0, 1, 1, [], []), _T("A", 2, 2, 3, 1, [], []),
               _T("B", 1, 4, 5, 1, [], []), _T("B", 2, 6, 7, 1, [], [])]
    assert mod.session_boundaries(ordered) == [2]     # A→B at ordinal 2


def test_sample_start_times_excludes_single_turn_sessions(mod):
    # title1/title2 are single-turn helper sessions between two real samples;
    # only the >=2-turn sessions become sample boundaries, at times rel to t0.
    ordered = mod.order_turns([
        _T("A", 1, 10.0, 15.0, 5.0, [], ["read"]),
        _T("A", 2, 17.0, 17.5, 0.5, ["read"], []),
        _T("title1", 1, 16.0, 16.2, 0.2, [], []),
        _T("title2", 1, 16.3, 16.5, 0.2, [], []),
        _T("B", 1, 30.0, 33.0, 3.0, [], ["grep"]),
        _T("B", 2, 35.0, 35.4, 0.4, ["grep"], []),
    ])
    # t0 = 10.0 (A's first start); samples A@0.0 and B@20.0; titles excluded
    assert mod.sample_start_times(ordered, min_turns=2) == [0.0, 20.0]
    # even at min_turns=1, the titles start INSIDE A's window (top-level
    # filter) so still only A and B remain
    assert len(mod.sample_start_times(ordered, min_turns=1)) == 2
    # ordinal positions for the turn-indexed fig1
    assert mod.sample_start_ordinals(ordered, min_turns=2) == [0, 4]


def test_t0_uses_first_start(mod):
    ordered = [_T("A", 1, 5.0, 6.0, 1.0, [], []),
               _T("B", 1, 8.0, 9.0, 1.0, [], [])]
    assert mod._t0(ordered) == 5.0


# ---------- bottom-percentile tool distribution ----------


def test_percentile_value(mod):
    vals = [0.3, 0.5, 3.0, 5.0]   # ceil(q*n)-1 index
    assert mod._percentile_value(vals, 1.0) == 5.0
    assert mod._percentile_value(vals, 0.5) == 0.5    # ceil(2)-1=1 -> 0.5
    assert mod._percentile_value(vals, 0.25) == 0.3


def test_bottom_pct_subsetting_and_pct(mod):
    turns = [
        _T("A", 1, 0, 5, 5.0, [], ["read"]),        # slow
        _T("A", 2, 7, 7.5, 0.5, ["read"], ["bash"]),
        _T("A", 3, 9, 9.3, 0.3, ["bash"], []),
        _T("B", 1, 20, 23, 3.0, [], ["grep"]),
    ]
    rows = mod.bottom_pct_tool_dist(turns, [1.0, 0.5])
    # bottom 100%: all 4 turns
    top = [r for r in rows if r["cutoff_pct"] == 100]
    assert top[0]["subset_n"] == 4
    # bottom 50%: threshold = ceil(0.5*4)-1 = idx1 of sorted walls
    # sorted walls [0.3,0.5,3.0,5.0] -> idx1 = 0.5 -> turns with wall<=0.5 = 2
    half = [r for r in rows if r["cutoff_pct"] == 50]
    assert half[0]["subset_n"] == 2
    prev = {r["tool"]: r for r in half if r["side"] == "prev"}
    # the two fast turns had prev read and prev bash -> 50% each
    assert prev["read"]["pct"] == pytest.approx(50.0)
    assert prev["bash"]["pct"] == pytest.approx(50.0)


def test_cur_key_and_none(mod):
    t_multi = _T("A", 1, 0, 1, 1.0, [], ["bash", "read"])
    t_first = _T("A", 1, 0, 1, 1.0, [], [])
    assert mod._cur_key(t_multi) == "bash+read"
    assert mod._cur_key(t_first) == "(none)"
    assert t_first.prev_key == "(none)"


# ---------- series ----------


def test_hit_series_sorted_filtered_with_session(mod):
    # y is now HIT TOKENS (cache.read), not a ratio; no-usage turns dropped
    turns = [_T("A", 2, 7, 7.5, 0.5, ["read"], [], hit_tokens=1500),
             _T("A", 1, 0, 5, 5.0, [], [], hit_tokens=None),  # no usage -> dropped
             _T("B", 1, 20, 23, 3.0, [], [], hit_tokens=0)]
    s = mod.hit_series(turns)
    assert s == [(7.5, 1500, "A"), (23.0, 0, "B")]


def test_gap_reuse_pairs_uses_prev_turn_cache_denominator(mod):
    # prev turn (step1): effective_input=100+0=100, output=0 -> cached=100.
    # this turn (step2): cache_read=80 -> reuse ratio = 80/100 = 0.8, and it
    # is NOT deflated by this turn reading big new content.
    prev = _T("A", 1, 0, 5, 5.0, [], ["read"], gap=None, hit_tokens=0)
    prev.output_tokens = 0        # cached = eff_input(100) + 0
    cur = _T("A", 2, 7, 7.5, 0.5, ["read"], ["bash"], gap=2.0, hit_tokens=80)
    cur.input_tokens = 5000       # huge NEW read -> raw hit ratio would be tiny
    pairs = mod.gap_reuse_pairs([prev, cur])
    assert len(pairs) == 1
    g, r, sid, step = pairs[0]
    assert (g, sid, step) == (2.0, "A", 2)
    assert r == pytest.approx(80 / 100)      # prev-cache denominator, not 80/5080


def test_gap_reuse_pairs_drops_first_and_gapless(mod):
    # first turn has no prev; a gapless turn is dropped
    turns = [_T("A", 1, 0, 5, 5.0, [], [], gap=None, hit_tokens=0),
             _T("A", 2, 7, 7.5, 0.5, [], [], gap=None, hit_tokens=50)]
    assert mod.gap_reuse_pairs(turns) == []


def test_sample_start_times_drops_nested_subagent_sessions(mod):
    # sub is a 2-turn task-subagent session running INSIDE A's window —
    # min_turns alone would keep it; the top-level filter must drop it.
    ordered = mod.order_turns([
        _T("A", 1, 0.0, 2.0, 2.0, [], []),
        _T("sub", 1, 3.0, 4.0, 1.0, [], []),
        _T("sub", 2, 4.5, 5.0, 0.5, [], []),
        _T("A", 2, 6.0, 8.0, 2.0, [], []),
        _T("B", 1, 10.0, 11.0, 1.0, [], []),
        _T("B", 2, 12.0, 13.0, 1.0, [], []),
    ])
    assert mod.sample_start_times(ordered, min_turns=2) == [0.0, 10.0]


def test_trim_to_window(mod):
    series = [(0.0, 1.0), (10.0, 2.0), (100.0, 3.0)]
    assert mod.trim_to_window(series, 8.0, 20.0, margin_s=5.0) == [(10.0, 2.0)]


def test_worker_log_series_parses_and_unwraps_midnight(mod, tmp_path):
    p = tmp_path / "vllm-a0.log"
    p.write_text(
        "INFO 07-15 10:00:00 [loggers.py:1] Engine 000: GPU KV cache usage: "
        "12.5%, Prefix cache hit rate: 88.0%\n"
        "unrelated line with no stats\n"
        "INFO 07-15 10:00:10 [loggers.py:1] GPU KV cache usage: 20.0%, "
        "Prefix cache hit rate: 90.5%\n"
        # crosses midnight: seconds-of-day drops, must add a day
        "INFO 07-16 00:00:10 [loggers.py:1] GPU KV cache usage: 6.0%, "
        "Prefix cache hit rate: 11.0%\n")
    hits, kv = mod.worker_log_series(p)
    # % -> fraction, x relative to first stat, midnight unwrapped
    assert hits == [(0.0, 0.88), (10.0, 0.905),
                    (pytest.approx(50410.0), 0.11)]
    assert [round(v, 3) for _t, v in kv] == [0.125, 0.2, 0.06]


def test_worker_log_series_empty_when_no_stats(mod, tmp_path):
    p = tmp_path / "vllm-a0.log"
    p.write_text("INFO nothing to see here\nanother line\n")
    assert mod.worker_log_series(p) == ([], [])


def test_near_zero_turns_flags_buffered_steps(mod, tmp_path):
    prof = tmp_path / "profiles"; prof.mkdir()
    _write_profile(prof, "ses_main",
                   [{"ev": "llm.end", "step": 1, "duration_s": 5.0,
                     "step_duration_s": 5.1, "finish": "tool_calls",
                     "tokens": {"output": 40}},
                    {"ev": "llm.end", "step": 2, "duration_s": 0.0001,
                     "step_duration_s": 0.3, "finish": "tool_calls",
                     "tokens": {"output": 25}, "request_id": "abc"}])
    # a helper session excluded by keep_ids
    _write_profile(prof, "ses_title",
                   [{"ev": "llm.end", "step": 1, "duration_s": 0.0,
                     "tokens": {"output": 3}}])
    rows = mod.near_zero_turns(prof, {"ses_main"}, 0.01)
    assert len(rows) == 1
    assert rows[0]["step"] == 2
    assert rows[0]["output_tokens"] == 25          # generated despite ~0 dur
    assert rows[0]["step_duration_s"] == 0.3


def test_kv_usage_series(mod, tmp_path):
    p = tmp_path / "m.ndjson"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"ts": 2.0, "ok": True, "metrics":
            {"vllm:kv_cache_usage_perc": [{"labels": {}, "value": 0.4},
                                          {"labels": {}, "value": 0.6}]}},
        {"ts": 1.0, "ok": True, "metrics":
            {"vllm:kv_cache_usage_perc": [{"labels": {}, "value": 0.2}]}},
        {"ts": 3.0, "ok": False, "error": "x"},          # dropped
        {"ts": 4.0, "ok": True, "metrics": {}},           # no metric -> skipped
    ]) + "\n")
    s = mod.kv_usage_series(p)
    assert s == [(1.0, 0.2), (2.0, 0.5)]                  # sorted, mean of 0.4/0.6


def test_trace_session_ids(mod, tmp_path):
    p = tmp_path / "trace.jsonl"
    p.write_text(json.dumps({"session_id": "ses_a", "success": True}) + "\n"
                 + json.dumps({"session_id": "ses_b", "success": False}) + "\n"
                 + "not json\n"
                 + json.dumps({"no_sid": 1}) + "\n")
    assert mod.trace_session_ids(p) == {"ses_a", "ses_b"}


# ---------- main() ----------


def _write_profile(dirpath: Path, sid: str, evs: list[dict]) -> None:
    (dirpath / f"{sid}.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in evs), encoding="utf-8")


def _turn_events(step, start, end, inp, out, cache, tool):
    evs = [{"ev": "llm.start", "ts": start, "step": step},
           {"ev": "llm.end", "ts": end, "step": step, "duration_s": end - start,
            "tokens": {"input": inp, "output": out, "cache": {"read": cache}},
            "dynamo": {}}]
    if tool:
        evs.append({"ev": "tool.end", "ts": end + 0.1, "step": step,
                    "name": tool, "callID": f"c{step}", "duration_s": 1.0, "ok": True})
    return evs


def test_main_writes_csvs_no_figures(mod, tmp_path, capsys):
    prof = tmp_path / "profiles"; prof.mkdir()
    _write_profile(prof, "A",
                   _turn_events(1, 0, 5, 2000, 300, 0, "read")
                   + _turn_events(2, 7, 7.5, 500, 20, 1500, None))
    out = tmp_path / "o"
    rc = mod.main(["--profiles", str(prof), "--out", str(out), "--no-figures"])
    assert rc == 0
    assert not list(out.glob("*.pdf"))
    ordered = list(csv.DictReader((out / "turns_ordered.csv").open()))
    assert [r["step"] for r in ordered] == ["1", "2"]
    assert ordered[1]["prev_tools"] == "read"
    assert ordered[1]["turn_gap_s"] == "2.0"
    bottom = list(csv.DictReader((out / "bottom_pct_tools.csv").open()))
    assert {r["side"] for r in bottom} == {"prev", "cur"}
    assert "2 sample boundaries" not in capsys.readouterr().out  # only 1 session


def test_main_trace_filter_drops_helper_sessions(mod, tmp_path, capsys):
    prof = tmp_path / "profiles"; prof.mkdir()
    _write_profile(prof, "ses_main",
                   _turn_events(1, 0, 5, 2000, 300, 0, "read")
                   + _turn_events(2, 7, 7.5, 500, 20, 1500, None))
    _write_profile(prof, "ses_title", _turn_events(1, 6, 6.2, 100, 10, 0, None))
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps({"session_id": "ses_main"}) + "\n")
    out = tmp_path / "o"
    rc = mod.main(["--profiles", str(prof), "--trace", str(trace),
                   "--out", str(out), "--no-figures"])
    assert rc == 0
    assert "kept 1/2 sessions" in capsys.readouterr().out
    ordered = list(csv.DictReader((out / "turns_ordered.csv").open()))
    assert {r["session_id"] for r in ordered} == {"ses_main"}


def test_main_trace_missing_returns_2(mod, tmp_path, capsys):
    prof = tmp_path / "profiles"; prof.mkdir()
    _write_profile(prof, "A", _turn_events(1, 0, 5, 2000, 300, 0, None))
    rc = mod.main(["--profiles", str(prof),
                   "--trace", str(tmp_path / "nope.jsonl"), "--no-figures"])
    assert rc == 2
    assert "trace not found" in capsys.readouterr().err


def test_main_missing_profiles_returns_2(mod, tmp_path, capsys):
    rc = mod.main(["--profiles", str(tmp_path / "nope")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err
