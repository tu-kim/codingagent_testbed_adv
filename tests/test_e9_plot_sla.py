"""Tests for scripts/arm/e9_plot_ttft_tpot_sla.py.

No network, no GPU: this is the pure-CPU reporting half of the split-off
`e9_offline_ttft_tpot_sweep.py` (see tests/test_e9_offline_sweep.py for
the measurement half). It only ever reads CSV files written by that
script's `write_csv`/columns and never touches vLLM, so no fake `vllm`
module is needed here -- loaded via importlib (script, not a package
module) the same way as the sibling test file.

Two sweep.csv shapes are exercised: the pre-`--repeat` shape (`_SWEEP_COLS`
/ `_sweep_row` / `_write_sweep_csv`, no ttft_ms_min/max or tpot_ms_min/max
columns) and the current `--repeat`-aware shape (`_SWEEP_COLS_WITH_REPEAT`
/ `_sweep_row_with_repeat` / `_write_sweep_csv_with_repeat`). merge_sweeps /
sla_limits / print_sweep / print_sla only ever read batch/prompt_tokens/
<metric>/error and are indifferent to which shape they're given -- the
`_min`/`_max` columns exist solely for `plot_metric`'s shaded band, tested
under TestPlotMetric with matplotlib-guarded (`pytest.importorskip`) cases
for: band columns present, absent, blank-but-present, a mix of banded and
unbanded rows in one call (what merge_sweeps hands plot_metric when a
--repeat run and an older run are merged), and repeat=1 (columns present
but every hi==lo, so the "any hi > lo" guard must suppress the band without
raising).

Per-batch-folder input (`batch_from_dir`, `expand_sweep_paths`, and
`merge_sweeps`'s folder-overrides-column behaviour) is covered under
TestBatchFromDir / TestExpandSweepPaths and the override/mismatch cases in
TestMergeSweeps. NOTE: `merge_sweeps` now returns `(rows, report)`, not
just `rows` -- every call site below unpacks the tuple.
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
    ever read batch/prompt_tokens/<metric>/error. No <metric>_min/_max keys
    at all -- this is the shape of an older sweep.csv written before
    --repeat existed, or any row plot_metric must render sans band."""
    return {"batch": batch, "prompt_tokens": prompt_tokens,
            "ttft_ms": ttft_ms, "error": error}


def _rw_band(batch, prompt_tokens, ttft_ms, ttft_ms_min, ttft_ms_max, error=""):
    """Row carrying the repeat>1 band columns plot_metric shades between."""
    return {"batch": batch, "prompt_tokens": prompt_tokens,
            "ttft_ms": ttft_ms, "ttft_ms_min": ttft_ms_min,
            "ttft_ms_max": ttft_ms_max, "error": error}


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


# Current sweep.csv shape (--repeat > 1 recorded): adds repeat plus the
# ttft_ms/tpot_ms _min/_max band columns. Kept SEPARATE from _SWEEP_COLS /
# _sweep_row / _write_sweep_csv above, which stay as the pre-repeat shape on
# purpose -- this script must keep reading both (merge_sweeps/sla_limits/
# print_sweep never touch the new columns; only plot_metric's band logic
# cares whether they're present).
_SWEEP_COLS_WITH_REPEAT = ["batch", "prompt_tokens", "gen_tokens", "repeat",
                          "ttft_ms", "ttft_ms_min", "ttft_ms_max",
                          "tpot_ms", "tpot_ms_min", "tpot_ms_max",
                          "decode_ms", "total_ms", "error"]


def _sweep_row_with_repeat(batch, prompt_tokens, ttft_ms, ttft_ms_min,
                           ttft_ms_max, tpot_ms, tpot_ms_min, tpot_ms_max,
                           gen_tokens=8, repeat=3):
    decode_ms = tpot_ms * (gen_tokens - 1)
    return {"batch": batch, "prompt_tokens": prompt_tokens,
            "gen_tokens": gen_tokens, "repeat": repeat,
            "ttft_ms": ttft_ms, "ttft_ms_min": ttft_ms_min,
            "ttft_ms_max": ttft_ms_max, "tpot_ms": tpot_ms,
            "tpot_ms_min": tpot_ms_min, "tpot_ms_max": tpot_ms_max,
            "decode_ms": decode_ms, "total_ms": ttft_ms + decode_ms,
            "error": ""}


def _write_sweep_csv_with_repeat(path: Path, rows: list) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_SWEEP_COLS_WITH_REPEAT)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# batch_from_dir
# ---------------------------------------------------------------------------

class TestBatchFromDir:
    @pytest.mark.parametrize("name, expected", [
        ("b8", 8), ("batch8", 8), ("B-8", 8), ("b_8", 8),
        ("BATCH-8", 8), ("Batch_8", 8), ("b-8", 8),
        ("b100", 100), ("batch_100", 100),
    ])
    def test_accepted_spellings(self, e9p, name, expected):
        assert e9p.batch_from_dir(Path(name)) == expected

    @pytest.mark.parametrize("name", [
        "b",        # no digits at all
        "x8",       # doesn't start with b/batch
        "8b",       # digits before the b
        "batch",    # "atch" with no digits
        "bx8",      # stray char between b and the digits
        "b-",       # separator with no digits
        "",         # empty name
    ])
    def test_rejected_spellings(self, e9p, name):
        assert e9p.batch_from_dir(Path(name)) is None


# ---------------------------------------------------------------------------
# expand_sweep_paths
# ---------------------------------------------------------------------------

class TestExpandSweepPaths:
    def test_file_passes_through_with_batch_from_parent(self, e9p, tmp_path):
        f = tmp_path / "b3" / "data.csv"
        _write_sweep_csv(f, [_sweep_row(3, 1000, 10.0, 1.0)])

        out = e9p.expand_sweep_paths([f])

        assert out == [(f, 3)]

    def test_file_passes_through_with_none_when_parent_does_not_parse(
            self, e9p, tmp_path):
        f = tmp_path / "randomparent" / "data.csv"
        _write_sweep_csv(f, [_sweep_row(1, 1000, 10.0, 1.0)])

        out = e9p.expand_sweep_paths([f])

        assert out == [(f, None)]

    def test_dir_yields_own_sweep_csv_plus_batch_subfolders(
            self, e9p, tmp_path):
        root = tmp_path / "root"
        own = root / "sweep.csv"
        b1 = root / "b1" / "sweep.csv"
        b2 = root / "b2" / "sweep.csv"
        _write_sweep_csv(own, [_sweep_row(9, 500, 1.0, 1.0)])
        _write_sweep_csv(b1, [_sweep_row(1, 1000, 2.0, 1.0)])
        _write_sweep_csv(b2, [_sweep_row(2, 2000, 3.0, 1.0)])

        out = e9p.expand_sweep_paths([root])

        # own file first (batch_from_dir("root") -> None, column stays in
        # charge), then each batch subfolder in name-sorted order.
        assert out == [(own, None), (b1, 1), (b2, 2)]

    def test_batch_subfolder_without_sweep_csv_falls_back_to_its_own_glob(
            self, e9p, tmp_path):
        root = tmp_path / "root2"
        a = root / "b5" / "result_a.csv"
        b = root / "b5" / "result_b.csv"
        _write_sweep_csv(a, [_sweep_row(5, 100, 1.0, 1.0)])
        _write_sweep_csv(b, [_sweep_row(5, 200, 2.0, 1.0)])

        out = e9p.expand_sweep_paths([root])

        assert out == [(a, 5), (b, 5)]

    def test_dir_without_sweep_csv_or_batch_subdirs_falls_back_to_own_glob(
            self, e9p, tmp_path):
        root = tmp_path / "flat"
        a = root / "a.csv"
        b = root / "b.csv"
        _write_sweep_csv(a, [_sweep_row(1, 100, 1.0, 1.0)])
        _write_sweep_csv(b, [_sweep_row(1, 200, 2.0, 1.0)])
        # A non-batch-named subfolder must not count as a batch subfolder
        # and must not suppress the own-glob fallback.
        (root / "notes").mkdir(parents=True, exist_ok=True)

        out = e9p.expand_sweep_paths([root])

        assert out == [(a, None), (b, None)]


# ---------------------------------------------------------------------------
# read_csv / merge_sweeps
# ---------------------------------------------------------------------------

class TestMergeSweeps:
    def test_later_file_wins_on_repeated_cell(self, e9p, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        _write_sweep_csv(f1, [_sweep_row(1, 1000, 50.0, 5.0)])
        _write_sweep_csv(f2, [_sweep_row(1, 1000, 999.0, 5.0)])

        out, rep = e9p.merge_sweeps([f1, f2])

        assert len(out) == 1
        assert float(out[0]["ttft_ms"]) == 999.0
        assert rep == {"files": [str(f1), str(f2)], "overridden": 0,
                       "mismatched": 0}

    def test_cells_are_sorted_by_batch_then_prompt_tokens(self, e9p, tmp_path):
        f1 = tmp_path / "a.csv"
        _write_sweep_csv(f1, [
            _sweep_row(4, 2000, 1.0, 1.0),
            _sweep_row(1, 4000, 1.0, 1.0),
            _sweep_row(1, 1000, 1.0, 1.0),
        ])

        out, _rep = e9p.merge_sweeps([f1])

        assert [(int(r["batch"]), int(r["prompt_tokens"])) for r in out] == [
            (1, 1000), (1, 4000), (4, 2000),
        ]

    def test_multiple_files_are_unioned(self, e9p, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        _write_sweep_csv(f1, [_sweep_row(1, 1000, 1.0, 1.0)])
        _write_sweep_csv(f2, [_sweep_row(1, 2000, 1.0, 1.0)])

        out, rep = e9p.merge_sweeps([f1, f2])

        assert len(out) == 2
        assert {(int(r["batch"]), int(r["prompt_tokens"])) for r in out} == {
            (1, 1000), (1, 2000),
        }
        assert rep["files"] == [str(f1), str(f2)]

    def test_folder_batch_overrides_column_and_counts_override(
            self, e9p, tmp_path):
        # Folder name parses and AGREES with the CSV's own batch column:
        # still counted as "overridden" (the folder decided, not the
        # column), but not as a mismatch.
        f = tmp_path / "b3" / "sweep.csv"
        _write_sweep_csv(f, [_sweep_row(3, 2000, 60.0, 5.0)])

        out, rep = e9p.merge_sweeps([tmp_path])

        assert out[0]["batch"] == 3
        assert rep["overridden"] == 1
        assert rep["mismatched"] == 0

    def test_folder_batch_disagreeing_with_column_counts_mismatch(
            self, e9p, tmp_path):
        # CSV column says batch=1, folder name says batch=2 -- the folder
        # wins on the output row, and the disagreement is counted.
        f = tmp_path / "b2" / "sweep.csv"
        _write_sweep_csv(f, [_sweep_row(1, 1000, 50.0, 5.0)])

        out, rep = e9p.merge_sweeps([tmp_path])

        assert len(out) == 1
        assert out[0]["batch"] == 2
        assert rep["overridden"] == 1
        assert rep["mismatched"] == 1

    def test_override_and_mismatch_counts_accumulate_across_files(
            self, e9p, tmp_path):
        _write_sweep_csv(tmp_path / "b2" / "sweep.csv",
                         [_sweep_row(1, 1000, 50.0, 5.0)])  # mismatched
        _write_sweep_csv(tmp_path / "b3" / "sweep.csv",
                         [_sweep_row(3, 2000, 60.0, 5.0)])  # matched

        out, rep = e9p.merge_sweeps([tmp_path])

        assert {(r["batch"], int(r["prompt_tokens"])) for r in out} == {
            (2, 1000), (3, 2000),
        }
        assert rep["overridden"] == 2
        assert rep["mismatched"] == 1

    def test_no_folder_batch_leaves_column_untouched(self, e9p, tmp_path):
        # A folder name that doesn't parse (e.g. the bare tmp_path itself)
        # must not increment overridden/mismatched at all -- the CSV
        # column is the sole source of the batch, exactly as before this
        # feature existed.
        f = tmp_path / "sweep.csv"
        _write_sweep_csv(f, [_sweep_row(1, 1000, 50.0, 5.0)])

        out, rep = e9p.merge_sweeps([f])

        assert out[0]["batch"] == "1"
        assert rep["overridden"] == 0
        assert rep["mismatched"] == 0

    def test_later_file_wins_across_folders(self, e9p, tmp_path):
        # The later-file-wins rule from test_later_file_wins_on_repeated_cell
        # holds even when one contributor is a batch-folder expansion and
        # the other is a plain file passed after it -- both resolve to the
        # same (batch, prompt_tokens) cell and the later one in --sweep
        # argument order wins.
        _write_sweep_csv(tmp_path / "b1" / "sweep.csv",
                         [_sweep_row(1, 500, 10.0, 1.0)])
        override = tmp_path / "override.csv"
        _write_sweep_csv(override, [_sweep_row(1, 500, 999.0, 1.0)])

        out, rep = e9p.merge_sweeps([tmp_path / "b1", override])

        assert len(out) == 1
        assert float(out[0]["ttft_ms"]) == 999.0
        # only the folder-expanded file counted as an override; the plain
        # file's own "1" batch column stood unmodified.
        assert rep["overridden"] == 1
        assert rep["mismatched"] == 0

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

    def test_band_columns_present_written_without_raising(self, e9p, tmp_path):
        # repeat>1 sweep: every row carries ttft_ms_min/_max with hi > lo,
        # so the shaded min-max band path is exercised.
        pytest.importorskip("matplotlib")
        rows = [_rw_band(1, 1000, 50.0, 40.0, 60.0),
                _rw_band(1, 2000, 150.0, 130.0, 170.0)]
        out = tmp_path / "fig_band.pdf"
        e9p.plot_metric(rows, "ttft_ms", "TTFT (ms)", "title", None, [], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_band_columns_absent_still_written(self, e9p, tmp_path):
        # Older sweep.csv (pre --repeat): no ttft_ms_min/_max keys at all.
        # plot_metric must render the curve with no band and not raise.
        pytest.importorskip("matplotlib")
        rows = [_rw(1, 1000, 50.0), _rw(1, 2000, 150.0)]
        out = tmp_path / "fig_noband.pdf"
        e9p.plot_metric(rows, "ttft_ms", "TTFT (ms)", "title", None, [], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_mix_of_banded_and_unbanded_rows_written_without_raising(
            self, e9p, tmp_path):
        # batch 1 came from a repeat>1 run (has band columns), batch 4 from
        # an older repeat=1-shaped file merged alongside it (no columns at
        # all) -- exactly what merge_sweeps() can hand plot_metric.
        pytest.importorskip("matplotlib")
        rows = [
            _rw_band(1, 1000, 50.0, 40.0, 60.0),
            _rw_band(1, 2000, 150.0, 130.0, 170.0),
            _rw(4, 1000, 80.0),
            _rw(4, 2000, 220.0),
        ]
        out = tmp_path / "fig_mixed.pdf"
        e9p.plot_metric(rows, "ttft_ms", "TTFT (ms)", "title", None, [], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_blank_band_values_are_skipped_without_raising(self, e9p, tmp_path):
        # Column present (e.g. merge_sweeps unioned it in from a sibling
        # file's header) but blank on THIS row -- must be treated the same
        # as "absent", not crash trying to float("").
        pytest.importorskip("matplotlib")
        rows = [_rw_band(1, 1000, 50.0, "", ""),
                _rw_band(1, 2000, 150.0, 130.0, 170.0)]
        out = tmp_path / "fig_blankband.pdf"
        e9p.plot_metric(rows, "ttft_ms", "TTFT (ms)", "title", None, [], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_repeat_one_min_equals_max_produces_no_band_but_no_raise(
            self, e9p, tmp_path):
        # repeat=1: columns are present but every hi == lo, so the "any hi
        # > lo" guard must suppress the (zero-width, useless) band without
        # raising -- this is the "no variance information" case the
        # measurement script's docstring calls out explicitly.
        pytest.importorskip("matplotlib")
        rows = [_rw_band(1, 1000, 50.0, 50.0, 50.0),
                _rw_band(1, 2000, 150.0, 150.0, 150.0)]
        out = tmp_path / "fig_flatband.pdf"
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

    def test_end_to_end_with_repeat_columns_produces_banded_figures(
            self, e9p, tmp_path):
        # Forward-compat check: a --repeat sweep.csv (extra repeat/_min/_max
        # columns) flows through merge_sweeps/sla_limits/print_sweep (which
        # ignore the extra columns) and plot_metric (which uses them) with
        # no special main()-side handling.
        pytest.importorskip("matplotlib")
        csv_path = tmp_path / "sweep.csv"
        _write_sweep_csv_with_repeat(csv_path, [
            _sweep_row_with_repeat(1, 1000, 50.0, 40.0, 60.0, 5.0, 4.0, 6.0),
            _sweep_row_with_repeat(1, 2000, 150.0, 130.0, 170.0, 6.0, 5.0, 7.0),
        ])
        out_dir = tmp_path / "out"

        rc = e9p.main(["--sweep", str(csv_path), "--out", str(out_dir)])

        assert rc == 0
        assert (out_dir / "fig_ttft.pdf").exists()
        assert (out_dir / "fig_tpot.pdf").exists()

    def test_batch_folder_tree_end_to_end_prints_three_curves_and_mismatch(
            self, e9p, tmp_path, capsys):
        # Per-batch-folder layout: b1/, b2/, b4/, each sweep.csv's OWN
        # `batch` column left at 1 (as if copy-pasted from a template) --
        # the folder name must win for b2 and b4, producing two mismatches,
        # and the merged table must still show three distinct batch-size
        # curves (1, 2, 4), one row of data per folder.
        root = tmp_path / "root"
        _write_sweep_csv(root / "b1" / "sweep.csv",
                         [_sweep_row(1, 1000, 50.0, 5.0)])
        _write_sweep_csv(root / "b2" / "sweep.csv",
                         [_sweep_row(1, 1000, 60.0, 5.0)])
        _write_sweep_csv(root / "b4" / "sweep.csv",
                         [_sweep_row(1, 1000, 70.0, 5.0)])
        out_dir = tmp_path / "out"

        rc = e9p.main(["--sweep", str(root), "--out", str(out_dir),
                      "--no-figures"])

        assert rc == 0
        out = capsys.readouterr().out
        assert "cells: 3 from 3 file(s)" in out
        assert ("note: 2 rows had a batch column disagreeing with their "
                "folder name; the FOLDER won") in out
        # three separate curves' worth of rows: one per folder, each
        # carrying its own (folder-assigned) batch size and ttft value.
        lines = [l for l in out.splitlines() if l.strip()]
        row1 = next(l for l in lines if l.split()[:1] == ["1"])
        row2 = next(l for l in lines if l.split()[:1] == ["2"])
        row4 = next(l for l in lines if l.split()[:1] == ["4"])
        assert row1.split()[1:] == ["50.0"]
        assert row2.split()[1:] == ["60.0"]
        assert row4.split()[1:] == ["70.0"]
