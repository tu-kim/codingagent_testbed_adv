"""Tests for scripts/arm/e9_plot_ttft_tpot_sla.py.

No network, no GPU: this is the pure-CPU reporting half of the split-off
`e9_offline_ttft_tpot_sweep.py` (see tests/test_e9_offline_sweep.py for
the measurement half). It only ever reads CSV files written by that
script's `write_csv`/columns and never touches vLLM, so no fake `vllm`
module is needed here -- loaded via importlib (script, not a package
module) the same way as the sibling test file.
"""
import csv
import importlib.util
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "arm" / "e9_plot_ttft_tpot_sla.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e9p():
    return _load_module("e9_plot_ttft_tpot_sla", _SCRIPT_PATH)


# ---------------------------------------------------------------------------
# shared row helpers
# ---------------------------------------------------------------------------

_SWEEP_COLS = ["batch", "prompt_tokens", "gen_tokens", "ttft_ms", "tpot_ms",
              "decode_ms", "total_ms", "error"]


def _rw(batch, prompt_tokens, ttft_ms="", error=""):
    """Minimal row for sla_limits/print_sweep/plot_metric -- those only
    ever read batch/prompt_tokens/<metric>/error."""
    return {"batch": batch, "prompt_tokens": prompt_tokens,
            "ttft_ms": ttft_ms, "error": error}


def _sweep_row(batch, prompt_tokens, ttft_ms, tpot_ms, gen_tokens=8):
    """A full sweep.csv row, for round-tripping through write_csv/read_csv
    (merge_sweeps and main() read real files, not in-memory dicts)."""
    decode_ms = tpot_ms * (gen_tokens - 1)
    return {"batch": batch, "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens, "ttft_ms": ttft_ms, "tpot_ms": tpot_ms,
            "decode_ms": decode_ms, "total_ms": ttft_ms + decode_ms,
            "error": ""}


def _write_sweep_csv(path: Path, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_SWEEP_COLS)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# read_csv / merge_sweeps
# ---------------------------------------------------------------------------

class TestMergeSweeps:
    def test_later_file_wins_on_repeated_cell(self, e9p, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        _write_sweep_csv(f1, [_sweep_row(1, 1000, 50.0, 5.0)])
        _write_sweep_csv(f2, [_sweep_row(1, 1000, 999.0, 5.0)])

        out = e9p.merge_sweeps([f1, f2])

        assert len(out) == 1
        assert float(out[0]["ttft_ms"]) == 999.0

    def test_cells_are_sorted_by_batch_then_prompt_tokens(self, e9p, tmp_path):
        f1 = tmp_path / "a.csv"
        _write_sweep_csv(f1, [
            _sweep_row(4, 2000, 1.0, 1.0),
            _sweep_row(1, 4000, 1.0, 1.0),
            _sweep_row(1, 1000, 1.0, 1.0),
        ])

        out = e9p.merge_sweeps([f1])

        assert [(int(r["batch"]), int(r["prompt_tokens"])) for r in out] == [
            (1, 1000), (1, 4000), (4, 2000),
        ]

    def test_multiple_files_are_unioned(self, e9p, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        _write_sweep_csv(f1, [_sweep_row(1, 1000, 1.0, 1.0)])
        _write_sweep_csv(f2, [_sweep_row(1, 2000, 1.0, 1.0)])

        out = e9p.merge_sweeps([f1, f2])

        assert len(out) == 2
        assert {(int(r["batch"]), int(r["prompt_tokens"])) for r in out} == {
            (1, 1000), (1, 2000),
        }

    def test_read_csv_round_trips_header_and_rows(self, e9p, tmp_path):
        f1 = tmp_path / "a.csv"
        _write_sweep_csv(f1, [_sweep_row(1, 1000, 50.0, 5.0)])

        rows = e9p.read_csv(f1)

        assert len(rows) == 1
        assert rows[0]["batch"] == "1"
        assert rows[0]["prompt_tokens"] == "1000"


# ---------------------------------------------------------------------------
# sla_limits
# ---------------------------------------------------------------------------

class TestSlaLimits:
    def test_interpolated_crossing(self, e9p):
        rows = [_rw(1, 1000, 50.0), _rw(1, 2000, 150.0)]
        out = e9p.sla_limits(rows, "ttft_ms", 100.0)
        # y_ok=50 at x=1000, y_bad=150 at x=2000; frac=(100-50)/(150-50)=0.5
        # -> interp = 1000 + 0.5*(2000-1000) = 1500
        assert out == [{"metric": "ttft_ms", "sla": 100.0, "batch": 1,
                        "max_ok_measured": 1000, "max_ok_interp": 1500,
                        "crossed": True}]

    def test_never_crosses_reports_top_of_grid(self, e9p):
        rows = [_rw(1, 1000, 10.0), _rw(1, 2000, 20.0), _rw(1, 4000, 30.0)]
        out = e9p.sla_limits(rows, "ttft_ms", 100.0)
        assert out == [{"metric": "ttft_ms", "sla": 100.0, "batch": 1,
                        "max_ok_measured": 4000, "max_ok_interp": 4000,
                        "crossed": False}]

    def test_all_fail_reports_zero_and_crossed(self, e9p):
        rows = [_rw(1, 1000, 500.0), _rw(1, 2000, 600.0)]
        out = e9p.sla_limits(rows, "ttft_ms", 100.0)
        assert out == [{"metric": "ttft_ms", "sla": 100.0, "batch": 1,
                        "max_ok_measured": 0, "max_ok_interp": 0,
                        "crossed": True}]

    def test_error_and_blank_rows_excluded(self, e9p):
        # An error row (tiny value, which would otherwise become the new
        # max_ok) and a blank-metric row both sit strictly between two real
        # points. Correct exclusion leaves the single real point as a
        # "never crosses" result rather than corrupting the curve with
        # either excluded row.
        rows = [
            _rw(1, 1000, 50.0),
            _rw(1, 1500, ""),
            _rw(1, 2000, 5.0, error="OOM: boom"),
        ]
        out = e9p.sla_limits(rows, "ttft_ms", 100.0)
        assert out == [{"metric": "ttft_ms", "sla": 100.0, "batch": 1,
                        "max_ok_measured": 1000, "max_ok_interp": 1000,
                        "crossed": False}]

    def test_multiple_batches_sorted_ascending(self, e9p):
        rows = [_rw(4, 1000, 10.0), _rw(1, 1000, 10.0)]
        out = e9p.sla_limits(rows, "ttft_ms", 100.0)
        assert [r["batch"] for r in out] == [1, 4]


# ---------------------------------------------------------------------------
# print_sweep / print_sla
# ---------------------------------------------------------------------------

class TestPrintSweep:
    def test_error_cell_renders_as_x_missing_cell_as_dash(self, e9p, capsys):
        rows = [
            _rw(1, 1000, 10.0),
            _rw(1, 2000, 20.0),
            _rw(4, 1000, "", error="OOM: boom"),
            # (4, 2000) is simply absent from `rows` -> a missing cell.
        ]
        e9p.print_sweep(rows, "ttft_ms", "TTFT (ms)")
        out = capsys.readouterr().out
        assert "TTFT (ms) (rows = batch size, cols = prompt tokens):" in out
        lines = [l for l in out.splitlines() if l.strip()]
        row1 = next(l for l in lines if l.lstrip().startswith("1"))
        row4 = next(l for l in lines if l.lstrip().startswith("4"))
        assert row1.split()[1:] == ["10.0", "20.0"]
        assert row4.split()[1:] == ["x", "-"]


class TestPrintSla:
    def test_crossed_and_never_crossed_notes(self, e9p, capsys):
        limits = [
            {"metric": "ttft_ms", "sla": 100.0, "batch": 1,
             "max_ok_measured": 1000, "max_ok_interp": 1500, "crossed": True},
            {"metric": "ttft_ms", "sla": 100.0, "batch": 2,
             "max_ok_measured": 4000, "max_ok_interp": 4000,
             "crossed": False},
        ]
        e9p.print_sla(limits, "ms")
        out = capsys.readouterr().out
        assert "Longest prompt meeting ttft_ms <= 100 ms:" in out
        lines = [l for l in out.splitlines() if l.strip()]
        row1 = next(l for l in lines if l.split()[:1] == ["1"])
        row2 = next(l for l in lines if l.split()[:1] == ["2"])
        assert row1.split()[:3] == ["1", "1000", "1500"]
        assert "(never crossed within the grid)" not in row1
        assert "(never crossed within the grid)" in row2

    def test_empty_limits_prints_nothing(self, e9p, capsys):
        e9p.print_sla([], "ms")
        assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# plot_metric (matplotlib-dependent)
# ---------------------------------------------------------------------------

class TestPlotMetric:
    def test_writes_a_file_with_data(self, e9p, tmp_path):
        pytest.importorskip("matplotlib")
        rows = [_rw(1, 1000, 50.0), _rw(1, 2000, 150.0),
                _rw(4, 1000, 80.0), _rw(4, 2000, 220.0)]
        limits = e9p.sla_limits(rows, "ttft_ms", 100.0)
        out = tmp_path / "fig.pdf"
        e9p.plot_metric(rows, "ttft_ms", "TTFT (ms)",
                        "TTFT vs prompt tokens", 100.0, limits, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_all_error_rows_render_no_data_without_raising(self, e9p, tmp_path):
        pytest.importorskip("matplotlib")
        rows = [_rw(1, 1000, "", error="boom"),
                _rw(1, 2000, "", error="boom2")]
        out = tmp_path / "fig_empty.pdf"
        e9p.plot_metric(rows, "ttft_ms", "TTFT (ms)", "title", None, [], out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_end_to_end_returns_zero_and_prints_both_tables(
            self, e9p, tmp_path, capsys):
        csv_path = tmp_path / "sweep.csv"
        _write_sweep_csv(csv_path, [
            _sweep_row(1, 1000, 50.0, 5.0),
            _sweep_row(1, 2000, 150.0, 6.0),
        ])
        out_dir = tmp_path / "out"

        rc = e9p.main(["--sweep", str(csv_path), "--out", str(out_dir),
                      "--no-figures"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "TTFT (ms)" in out
        assert "TPOT (ms/token)" in out

    def test_sla_limits_csv_written_with_header_when_sla_flag_given(
            self, e9p, tmp_path):
        csv_path = tmp_path / "sweep.csv"
        _write_sweep_csv(csv_path, [
            _sweep_row(1, 1000, 50.0, 5.0),
            _sweep_row(1, 2000, 150.0, 6.0),
        ])
        out_dir = tmp_path / "out"

        rc = e9p.main(["--sweep", str(csv_path), "--out", str(out_dir),
                      "--no-figures", "--ttft-sla-ms", "100"])

        assert rc == 0
        sla_path = out_dir / "sla_limits.csv"
        assert sla_path.exists()
        with sla_path.open() as fh:
            reader = csv.DictReader(fh)
            assert reader.fieldnames == [
                "metric", "sla", "batch", "max_ok_measured",
                "max_ok_interp", "crossed",
            ]
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["metric"] == "ttft_ms"

    def test_sla_limits_csv_not_written_without_any_sla_flag(
            self, e9p, tmp_path):
        csv_path = tmp_path / "sweep.csv"
        _write_sweep_csv(csv_path, [_sweep_row(1, 1000, 50.0, 5.0)])
        out_dir = tmp_path / "out"

        rc = e9p.main(["--sweep", str(csv_path), "--out", str(out_dir),
                      "--no-figures"])

        assert rc == 0
        assert not (out_dir / "sla_limits.csv").exists()

    def test_merges_multiple_sweep_files(self, e9p, tmp_path, capsys):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        _write_sweep_csv(f1, [_sweep_row(1, 1000, 50.0, 5.0)])
        _write_sweep_csv(f2, [_sweep_row(1, 2000, 60.0, 5.0)])
        out_dir = tmp_path / "out"

        rc = e9p.main(["--sweep", str(f1), str(f2), "--out", str(out_dir),
                      "--no-figures"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "cells: 2 from 2 file(s)" in out

    def test_missing_file_returns_two(self, e9p, tmp_path, capsys):
        out_dir = tmp_path / "out"
        missing = tmp_path / "nope.csv"

        rc = e9p.main(["--sweep", str(missing), "--out", str(out_dir),
                      "--no-figures"])

        assert rc == 2
        err = capsys.readouterr().err
        assert str(missing) in err

    def test_empty_csv_returns_two(self, e9p, tmp_path):
        csv_path = tmp_path / "empty.csv"
        _write_sweep_csv(csv_path, [])  # header only, zero data rows
        out_dir = tmp_path / "out"

        rc = e9p.main(["--sweep", str(csv_path), "--out", str(out_dir),
                      "--no-figures"])

        assert rc == 2
