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


# minimal stand-in matching the TurnRec attributes the script reads
class _T:
    def __init__(self, session_id, step, start, end, wall, prev, cur,
                 gap=None, hit=None):
        self.session_id = session_id
        self.step = step
        self.llm_start_ts = start
        self.llm_end_ts = end
        self.llm_wall_s = wall
        self.prev_tools = tuple(prev)
        self.tool_names = list(cur)
        self.away_s = gap
        self._hit = hit

    @property
    def prev_key(self):
        return "+".join(sorted(set(self.prev_tools))) if self.prev_tools else "(none)"

    @property
    def cache_hit_ratio(self):
        return self._hit


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


def test_hit_series_sorted_and_filtered(mod):
    turns = [_T("A", 2, 7, 7.5, 0.5, ["read"], [], hit=0.75),
             _T("A", 1, 0, 5, 5.0, [], [], hit=None),   # no hit -> dropped
             _T("B", 1, 20, 23, 3.0, [], [], hit=0.0)]
    s = mod.hit_series(turns)
    assert s == [(7.5, 0.75), (23.0, 0.0)]


def test_gap_hit_pairs(mod):
    turns = [_T("A", 2, 7, 7.5, 0.5, ["read"], [], gap=2.0, hit=0.75),
             _T("A", 1, 0, 5, 5.0, [], [], gap=None, hit=0.0)]  # no gap -> dropped
    assert mod.gap_hit_pairs(turns) == [(2.0, 0.75)]


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


def test_main_missing_profiles_returns_2(mod, tmp_path, capsys):
    rc = mod.main(["--profiles", str(tmp_path / "nope")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err
