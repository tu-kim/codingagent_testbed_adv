"""Tests for scripts/analyze_latency_breakdown.py.

Covers: all outputs created and non-empty, pooled share exact values,
per-request share table structure, bucket n_requests and share-sum
invariant, zero-variance component (violin fallback path), total column
absent (reconstructed from component sum), custom --components / --total.
No network, no GPU — pandas/matplotlib are import-skipped gracefully.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_latency_breakdown.py"
)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location(
        "analyze_latency_breakdown", _SCRIPT_PATH
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_latency_breakdown"] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list-of-dicts as a CSV (header derived from first row keys)."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Synthetic 10-row dataset with uniform component proportions:
#   llm_wall_s  = 60% of duration_s
#   tool_wall_s = 30% of duration_s
#   others_s    = 10% of duration_s
#
# Hand-computed sums:
#   Σ duration_s  = 550,  Σ llm_wall_s = 330
#   Σ tool_wall_s = 165,  Σ others_s   =  55
#
# pd.Series.quantile (linear, default) on [10, 20, ..., 100]:
#   p50 = 55.0
#   p90 = 91.0   (index 8.1 → 90 + 0.1 * 10)
#   p99 = 99.1   (index 8.91 → 90 + 0.91 * 10)
#
# Bucket assignments:
#   <=p50   : [10, 20, 30, 40, 50]  → 5 requests  (50 ≤ 55.0)
#   p50-p90 : [60, 70, 80, 90]     → 4 requests  (90 ≤ 91.0)
#   p90-p99 : (91.0, 99.1]         → 0 requests
#   >p99    : [100]                → 1 request
# ---------------------------------------------------------------------------
_ROWS_10 = [
    {
        "duration_s": float(d),
        "llm_wall_s": d * 0.6,
        "tool_wall_s": d * 0.3,
        "others_s": d * 0.1,
    }
    for d in range(10, 110, 10)  # [10, 20, ..., 100]
]


# ---------------------------------------------------------------------------
# Test 1: all 6 outputs created and non-empty
# ---------------------------------------------------------------------------


def test_all_outputs_created(mod, tmp_path):
    """Happy path: all 3 CSVs and 3 PNGs are written and non-empty."""
    inp = tmp_path / "input.csv"
    _write_csv(inp, _ROWS_10)
    out = tmp_path / "out"

    rc = mod.main(["--input", str(inp), "--output", str(out)])
    assert rc == 0

    for fname in (
        "latency_pooled_share.csv",
        "latency_per_request_share.csv",
        "latency_conditional_by_bucket.csv",
        "latency_share_violin.png",
        "latency_sorted_stacked_bar.png",
        "latency_bucket_stacked_bar.png",
    ):
        p = out / fname
        assert p.exists(), f"expected output missing: {fname}"
        assert p.stat().st_size > 0, f"output file is empty: {fname}"


# ---------------------------------------------------------------------------
# Test 2: pooled share exact values
# ---------------------------------------------------------------------------


def test_pooled_share_exact(mod, tmp_path):
    """Pooled share = Σcomponent / Σtotal, hand-verified against _ROWS_10.

    Expected:
      llm_wall_s  = 330 / 550 = 0.6
      tool_wall_s = 165 / 550 = 0.3
      others_s    =  55 / 550 = 0.1
      TOTAL row   = 1.0
    """
    inp = tmp_path / "input.csv"
    _write_csv(inp, _ROWS_10)
    out = tmp_path / "out"
    mod.main(["--input", str(inp), "--output", str(out)])

    df = pd.read_csv(out / "latency_pooled_share.csv").set_index("component")

    assert df.loc["llm_wall_s", "pooled_share"] == pytest.approx(0.6)
    assert df.loc["tool_wall_s", "pooled_share"] == pytest.approx(0.3)
    assert df.loc["others_s", "pooled_share"] == pytest.approx(0.1)
    assert df.loc["TOTAL", "pooled_share"] == pytest.approx(1.0)

    assert df.loc["llm_wall_s", "total_seconds"] == pytest.approx(330.0)
    assert df.loc["tool_wall_s", "total_seconds"] == pytest.approx(165.0)
    assert df.loc["others_s", "total_seconds"] == pytest.approx(55.0)
    assert df.loc["TOTAL", "total_seconds"] == pytest.approx(550.0)


# ---------------------------------------------------------------------------
# Test 3: per-request share table structure
# ---------------------------------------------------------------------------


def test_per_request_share_rows_and_range(mod, tmp_path):
    """Per-request share table: one row per component (in component order),
    n_requests = number of input rows, mean in [0, 1].

    With _ROWS_10 every row carries identical proportions so
    mean = p50 = p90 = p99 = the component's exact share.
    """
    inp = tmp_path / "input.csv"
    _write_csv(inp, _ROWS_10)
    out = tmp_path / "out"
    mod.main(["--input", str(inp), "--output", str(out)])

    df = pd.read_csv(out / "latency_per_request_share.csv")
    assert list(df["component"]) == ["llm_wall_s", "tool_wall_s", "others_s"]

    for _, row in df.iterrows():
        assert 0.0 <= row["mean"] <= 1.0, (
            f"mean for {row['component']} out of [0, 1]: {row['mean']}"
        )
        assert row["n_requests"] == len(_ROWS_10)

    by_c = df.set_index("component")
    assert by_c.loc["llm_wall_s", "mean"] == pytest.approx(0.6)
    assert by_c.loc["tool_wall_s", "mean"] == pytest.approx(0.3)
    assert by_c.loc["others_s", "mean"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Test 4: bucket n_requests and mean shares sum to 1.0
# ---------------------------------------------------------------------------


def test_bucket_n_requests_and_share_sum(mod, tmp_path):
    """Conditional-by-bucket: verify n_requests per bucket and that mean
    component shares sum to 1.0 in every non-empty bucket.

    With _ROWS_10 (see module-level comment for derivation):
      <=p50   : 5 requests
      p50-p90 : 4 requests
      p90-p99 : 0 requests
      >p99    : 1 request
    """
    inp = tmp_path / "input.csv"
    _write_csv(inp, _ROWS_10)
    out = tmp_path / "out"
    mod.main(["--input", str(inp), "--output", str(out)])

    df = pd.read_csv(out / "latency_conditional_by_bucket.csv").set_index("bucket")

    assert int(df.loc["<=p50", "n_requests"]) == 5
    assert int(df.loc["p50-p90", "n_requests"]) == 4
    assert int(df.loc["p90-p99", "n_requests"]) == 0
    assert int(df.loc[">p99", "n_requests"]) == 1

    share_cols = [
        "llm_wall_s_mean_share",
        "tool_wall_s_mean_share",
        "others_s_mean_share",
    ]
    for bucket in ("<=p50", "p50-p90", ">p99"):
        row = df.loc[bucket]
        total_share = sum(float(row[c]) for c in share_cols)
        assert total_share == pytest.approx(1.0, abs=1e-9), (
            f"bucket '{bucket}': shares sum to {total_share}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# Test 5: zero-variance component — violin fallback must not crash
# ---------------------------------------------------------------------------


def test_zero_component_violin_no_crash(mod, tmp_path):
    """A component fixed at 0 for every request (np.var == 0) must not
    crash violinplot (which calls gaussian_kde and fails on zero-variance
    input).  The script's fallback branch draws a scatter + hline instead.
    Assert main returns 0 and the violin PNG is produced.
    """
    rows = [
        {
            "duration_s": float(d),
            "llm_wall_s": d * 0.8,
            "tool_wall_s": 0.0,   # always zero — degenerate variance
            "others_s": d * 0.2,
        }
        for d in [10, 20, 30, 40, 50, 60]
    ]
    inp = tmp_path / "input.csv"
    _write_csv(inp, rows)
    out = tmp_path / "out"

    rc = mod.main(["--input", str(inp), "--output", str(out)])
    assert rc == 0
    violin = out / "latency_share_violin.png"
    assert violin.exists()
    assert violin.stat().st_size > 0


# ---------------------------------------------------------------------------
# Test 6: total column absent — reconstructed from sum of components
# ---------------------------------------------------------------------------


def test_total_column_absent_reconstructed(mod, tmp_path):
    """When 'duration_s' is absent from the CSV, _load reconstructs it as
    the row-sum of components.  Pooled shares must still match the expected
    proportion (0.6 / 0.3 / 0.1), and main must return 0.
    """
    rows = [
        {"llm_wall_s": d * 0.6, "tool_wall_s": d * 0.3, "others_s": d * 0.1}
        for d in range(10, 70, 10)  # 6 rows: [10, 20, ..., 60]
    ]
    inp = tmp_path / "input.csv"
    _write_csv(inp, rows)
    out = tmp_path / "out"

    # Default --total is 'duration_s'; it's absent so the script falls
    # back to reconstructing from sum of components.
    rc = mod.main(["--input", str(inp), "--output", str(out)])
    assert rc == 0

    df = pd.read_csv(out / "latency_pooled_share.csv").set_index("component")
    assert df.loc["llm_wall_s", "pooled_share"] == pytest.approx(0.6)
    assert df.loc["tool_wall_s", "pooled_share"] == pytest.approx(0.3)
    assert df.loc["others_s", "pooled_share"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Test 7: custom --components and --total
# ---------------------------------------------------------------------------


def test_custom_components_and_total(mod, tmp_path):
    """--components a b and --total tot should override the defaults.
    Pooled-share CSV must reference only 'a', 'b', and 'TOTAL' (not the
    default column names).
    """
    rows = [
        {"tot": float(d), "a": d * 0.7, "b": d * 0.3}
        for d in [5, 10, 15, 20, 25, 30]
    ]
    inp = tmp_path / "input.csv"
    _write_csv(inp, rows)
    out = tmp_path / "out"

    rc = mod.main([
        "--input", str(inp),
        "--output", str(out),
        "--components", "a", "b",
        "--total", "tot",
    ])
    assert rc == 0

    df = pd.read_csv(out / "latency_pooled_share.csv").set_index("component")
    assert list(df.index) == ["a", "b", "TOTAL"]
    assert df.loc["a", "pooled_share"] == pytest.approx(0.7)
    assert df.loc["b", "pooled_share"] == pytest.approx(0.3)
    assert df.loc["TOTAL", "pooled_share"] == pytest.approx(1.0)
