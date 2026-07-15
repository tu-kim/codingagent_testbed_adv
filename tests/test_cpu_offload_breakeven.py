"""Tests for scripts/cpu_offload_breakeven.py.

Cost model over turn_sched.csv rows: GPU path (queue + re-prefill +
decode) vs CPU path (host-KV read + new-token attention + CPU decode),
plus the break-even cpu_decode_tps solve. Key risks: the
effective-input→new-tokens recovery (input_tokens column is
input+cache), unit conversions (tps→ms, GB/s→ms), inf/0 break-even
edges, and per-tool aggregation.
"""

from __future__ import annotations

import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cpu_offload_breakeven.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("cpu_offload_breakeven", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["cpu_offload_breakeven"] = module
    spec.loader.exec_module(module)
    return module


def _write_turn_sched(path: Path, rows: list[dict]) -> None:
    cols = ["session_id", "step", "request_id", "match", "prev_tools",
            "output_tokens", "input_tokens", "llm_wall_s", "elapsed_s",
            "prefill_queue_ms", "decode_queue_ms", "total_queue_ms",
            "queue_share", "queue_share_basis", "away_s",
            "away_displaced_tokens", "cache_read", "cache_hit_ratio",
            "cur_tools"]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def _row(sid="s", step=1, prev="bash", out=10, eff_input=1000, cache=800,
         queue=500.0):
    return {"session_id": sid, "step": step, "prev_tools": prev,
            "output_tokens": out, "input_tokens": eff_input,
            "cache_read": cache, "total_queue_ms": queue}


_KNOBS = dict(gpu_prefill_tps=1000.0, gpu_decode_tps=100.0,
              cpu_prefill_tps=100.0, cpu_decode_tps=10.0,
              host_kv_read_gbps=10.0, kv_bytes_per_token=1e5)


def test_cost_model_arithmetic(mod, tmp_path):
    p = tmp_path / "turn_sched.csv"
    # eff=1000, cache=800 -> new=200; out=10; queue=500
    _write_turn_sched(p, [_row()])
    (t,) = mod.load_turn_costs(p, **_KNOBS)
    assert t.new_tokens == 200
    # gpu = 500 + 200/1000*1e3 + 10/100*1e3 = 500 + 200 + 100 = 800
    assert t.gpu_ms == pytest.approx(800.0)
    # cpu = (1000*1e5)/(10e9)*1e3 + 200/100*1e3 + 10/10*1e3
    #     = 10ms + 2000ms + 1000ms = 3010
    assert t.cpu_ms == pytest.approx(3010.0)
    assert t.winner == "gpu"


def test_cpu_wins_on_big_queue_small_output(mod, tmp_path):
    p = tmp_path / "turn_sched.csv"
    # huge queue, tiny work, high cache hit -> cpu path wins
    _write_turn_sched(p, [_row(out=1, eff_input=1000, cache=990, queue=5000.0)])
    (t,) = mod.load_turn_costs(p, **_KNOBS)
    # gpu = 5000 + 10/1000*1e3 + 1/100*1e3 = 5020
    assert t.gpu_ms == pytest.approx(5020.0)
    # cpu = 10 + 10/100*1e3 + 1/10*1e3 = 10 + 100 + 100 = 210
    assert t.cpu_ms == pytest.approx(210.0)
    assert t.winner == "cpu"


def test_breakeven_solve(mod, tmp_path):
    p = tmp_path / "turn_sched.csv"
    _write_turn_sched(p, [_row(out=10, eff_input=1000, cache=800, queue=500.0)])
    (t,) = mod.load_turn_costs(p, **_KNOBS)
    be = mod.breakeven_cpu_decode_tps(
        t, cpu_prefill_tps=100.0, host_kv_read_gbps=10.0, kv_bytes_per_token=1e5)
    # cpu_fixed = 10 + 2000 = 2010; gpu = 800 -> slack < 0 -> inf
    assert math.isinf(be)
    # with fast cpu prefill: fixed = 10 + 200/10000*1e3 = 30; slack = 770
    be2 = mod.breakeven_cpu_decode_tps(
        t, cpu_prefill_tps=10000.0, host_kv_read_gbps=10.0, kv_bytes_per_token=1e5)
    assert be2 == pytest.approx(10 * 1e3 / 770.0)


def test_breakeven_zero_output(mod, tmp_path):
    p = tmp_path / "turn_sched.csv"
    _write_turn_sched(p, [_row(out=0, eff_input=100, cache=90, queue=1000.0)])
    (t,) = mod.load_turn_costs(p, **_KNOBS)
    be = mod.breakeven_cpu_decode_tps(
        t, cpu_prefill_tps=10000.0, host_kv_read_gbps=10.0, kv_bytes_per_token=1e5)
    assert be == 0.0


def test_by_tool_summary_win_rates(mod, tmp_path):
    p = tmp_path / "turn_sched.csv"
    _write_turn_sched(p, [
        _row(step=1, out=1, eff_input=1000, cache=990, queue=5000.0),  # cpu wins, small
        _row(step=2, out=500, eff_input=1000, cache=0, queue=0.0),     # gpu wins, large
    ])
    costs = mod.load_turn_costs(p, **_KNOBS)
    rows = mod.by_tool_summary(costs, small_tokens=64,
                               cpu_prefill_tps=_KNOBS["cpu_prefill_tps"],
                               host_kv_read_gbps=_KNOBS["host_kv_read_gbps"],
                               kv_bytes_per_token=_KNOBS["kv_bytes_per_token"])
    (r,) = rows
    assert r["count"] == 2
    assert r["small_count"] == 1
    assert r["cpu_win_rate"] == pytest.approx(0.5)
    assert r["cpu_win_rate_small"] == pytest.approx(1.0)


def test_main_writes_outputs(mod, tmp_path, capsys):
    p = tmp_path / "turn_sched.csv"
    _write_turn_sched(p, [_row(), _row(step=2, out=1, cache=990, queue=5000.0)])
    out = tmp_path / "be"
    rc = mod.main(["--turn-sched", str(p), "--out", str(out),
                   "--gpu-prefill-tps", "1000", "--gpu-decode-tps", "100",
                   "--cpu-prefill-tps", "100", "--cpu-decode-tps", "10",
                   "--host-kv-read-gbps", "10", "--kv-bytes-per-token", "100000"])
    assert rc == 0
    turns = list(csv.DictReader((out / "breakeven_turns.csv").open()))
    assert len(turns) == 2
    assert {t["winner"] for t in turns} == {"gpu", "cpu"}
    assert (out / "breakeven_by_tool.csv").is_file()
    assert "cpu wins" in capsys.readouterr().out


def test_main_missing_csv_returns_2(mod, tmp_path, capsys):
    rc = mod.main(["--turn-sched", str(tmp_path / "nope.csv")])
    assert rc == 2
    assert "not found" in capsys.readouterr().err
