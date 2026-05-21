"""Tests for scripts/analyze_frontend_log.py.

Parsing logic + ANSI stripping + CSV serialization. `main` is
exercised end-to-end, which DOES render matplotlib figures to PDF
under tmp_path -- warnings about missing Times fonts are acceptable
on systems without a serif font installed.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "analyze_frontend_log.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("analyze_frontend_log", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["analyze_frontend_log"] = module
    spec.loader.exec_module(module)
    return module


# Real `request completed` line shape from the user's actual log,
# stripped of ANSI codes for clarity. Includes the typical second
# block of (model=, input_tokens=, output_tokens=, ttft_ms=, avg_itl_ms=).
SAMPLE_LINE = (
    "2026-05-21T03:56:01.134595Z  INFO http-request: "
    "dynamo_llm::http::service::metrics: request completed "
    "request_id=c2f4f71e-39b1-46f6-ac9a-5ae6b054200c "
    "model=qwen3-coder-30b-a3b-instruct-fp8 endpoint=chat_completions "
    "request_type=stream status=success elapsed_ms=297 method=POST "
    "uri=/v1/chat/completions version=HTTP/1.1 "
    "request_id=8b9d1cac-3505-4f00-ac96-fd84b7ac184a "
    'model="qwen3-coder-30b-a3b-instruct-fp8" '
    'input_tokens=1187 output_tokens=13 ttft_ms="187.48" avg_itl_ms="9.07"'
)


def test_strip_ansi_handles_pretty_formatter_codes(mod):
    """tracing-subscriber's pretty formatter wraps fields in SGR codes
    even when stdout is redirected to a file. Stripping must keep the
    semantic text intact."""
    raw = (
        "\x1b[2m2026-05-21T03:50:31.845841Z\x1b[0m \x1b[32m INFO\x1b[0m "
        "\x1b[2mmain.async_main\x1b[0m\x1b[2m:\x1b[0m Request migration "
        "disabled (limit: 0)"
    )
    out = mod.strip_ansi(raw)
    assert "\x1b" not in out
    assert "Request migration disabled (limit: 0)" in out
    assert "2026-05-21T03:50:31.845841Z" in out


def test_parse_extracts_all_request_fields(mod, tmp_path):
    log = tmp_path / "frontend.log"
    log.write_text(SAMPLE_LINE + "\n")
    rows = mod.parse_frontend_log(log)
    assert len(rows) == 1
    r = rows[0]
    assert r["model"] == "qwen3-coder-30b-a3b-instruct-fp8"
    assert r["status"] == "success"
    assert r["elapsed_ms"] == 297.0
    assert r["ttft_ms"] == pytest.approx(187.48)
    assert r["input_tokens"] == 1187
    assert r["output_tokens"] == 13
    # derived: decode_ms = elapsed - ttft = 297 - 187.48 = 109.52
    assert r["decode_ms"] == pytest.approx(109.52)
    # itl_ms_per_token = decode_ms / output = 109.52 / 13
    assert r["itl_ms_per_token"] == pytest.approx(109.52 / 13)
    # isl/osl = 1187 / 13
    assert r["isl_osl_ratio"] == pytest.approx(1187 / 13)


def test_parse_ignores_non_request_lines(mod, tmp_path):
    log = tmp_path / "frontend.log"
    log.write_text("\n".join([
        "2026-05-21T03:50:31.845841Z  INFO main: Request migration disabled",
        SAMPLE_LINE,
        "2026-05-21T03:50:32.000000Z  WARN something: unrelated",
    ]) + "\n")
    rows = mod.parse_frontend_log(log)
    assert len(rows) == 1


def test_parse_strips_ansi_from_actual_raw_line(mod, tmp_path):
    """Round-trip the production format: with embedded SGR codes the
    parser still recovers all fields, INCLUDING derived ones."""
    ansi_line = (
        "\x1b[2m2026-05-21T03:56:01Z\x1b[0m \x1b[32m INFO\x1b[0m "
        "\x1b[1mhttp-request\x1b[0m: dynamo_llm::http::service::metrics: "
        "request completed "
        "\x1b[3mmodel\x1b[0m\x1b[2m=\x1b[0mqwen3-coder-30b-a3b-instruct-fp8 "
        "\x1b[3mstatus\x1b[0m\x1b[2m=\x1b[0msuccess "
        "\x1b[3melapsed_ms\x1b[0m\x1b[2m=\x1b[0m297 "
        "\x1b[3mmodel\x1b[0m\x1b[2m=\x1b[0m\"qwen3-coder-30b-a3b-instruct-fp8\" "
        "\x1b[3minput_tokens\x1b[0m\x1b[2m=\x1b[0m1187 "
        "\x1b[3moutput_tokens\x1b[0m\x1b[2m=\x1b[0m13 "
        '\x1b[3mttft_ms\x1b[0m\x1b[2m=\x1b[0m"187.48"'
    )
    log = tmp_path / "frontend.log"
    log.write_text(ansi_line + "\n")
    rows = mod.parse_frontend_log(log)
    assert len(rows) == 1
    r = rows[0]
    assert r["elapsed_ms"] == 297.0
    assert r["input_tokens"] == 1187
    assert r["output_tokens"] == 13
    assert r["ttft_ms"] == pytest.approx(187.48)
    # Derived fields must compute correctly through the ANSI strip path.
    assert r["decode_ms"] == pytest.approx(109.52)
    assert r["itl_ms_per_token"] == pytest.approx(109.52 / 13)
    assert r["isl_osl_ratio"] == pytest.approx(1187 / 13)


def test_parse_filters_by_model(mod, tmp_path):
    other = SAMPLE_LINE.replace(
        'model="qwen3-coder-30b-a3b-instruct-fp8"', 'model="some-other-model"'
    ).replace(
        "model=qwen3-coder-30b-a3b-instruct-fp8 ", "model=some-other-model "
    )
    log = tmp_path / "frontend.log"
    log.write_text(SAMPLE_LINE + "\n" + other + "\n")
    rows = mod.parse_frontend_log(log, model_filter="qwen3-coder-30b-a3b-instruct-fp8")
    assert len(rows) == 1
    assert rows[0]["model"] == "qwen3-coder-30b-a3b-instruct-fp8"


def test_parse_returns_all_models_when_filter_is_none(mod, tmp_path):
    """No --model flag → both models survive. Guards against a
    regression where the filter accidentally rejects None case."""
    other = SAMPLE_LINE.replace(
        'model="qwen3-coder-30b-a3b-instruct-fp8"', 'model="some-other-model"'
    ).replace(
        "model=qwen3-coder-30b-a3b-instruct-fp8 ", "model=some-other-model "
    )
    log = tmp_path / "frontend.log"
    log.write_text(SAMPLE_LINE + "\n" + other + "\n")
    rows = mod.parse_frontend_log(log, model_filter=None)
    assert len(rows) == 2
    models = {r["model"] for r in rows}
    assert models == {"qwen3-coder-30b-a3b-instruct-fp8", "some-other-model"}


def test_parse_handles_missing_ttft_gracefully(mod, tmp_path):
    """Error / cancel paths can omit ttft_ms; row is still returned
    with derived fields = None."""
    no_ttft = (
        "2026-05-21T03:56:01Z  INFO http-request: "
        "dynamo_llm::http::service::metrics: request completed "
        "model=m endpoint=chat_completions status=error "
        "elapsed_ms=42 method=POST uri=/v1/chat/completions "
        "input_tokens=100 output_tokens=5"
    )
    log = tmp_path / "frontend.log"
    log.write_text(no_ttft + "\n")
    rows = mod.parse_frontend_log(log)
    assert len(rows) == 1
    r = rows[0]
    assert r["elapsed_ms"] == 42.0
    assert r["ttft_ms"] is None
    assert r["decode_ms"] is None
    assert r["itl_ms_per_token"] is None
    assert r["input_tokens"] == 100
    assert r["output_tokens"] == 5
    assert r["isl_osl_ratio"] == pytest.approx(100 / 5)
    assert r["status"] == "error"


def test_parse_skips_rows_without_tokens(mod, tmp_path):
    """A `request completed` without input_tokens / output_tokens is
    unusable for our metrics -- drop it silently."""
    no_tokens = (
        "2026-05-21T03:56:01Z  INFO http-request: "
        "dynamo_llm::http::service::metrics: request completed "
        "model=m endpoint=chat_completions status=error "
        "elapsed_ms=10 method=POST"
    )
    log = tmp_path / "frontend.log"
    log.write_text(no_tokens + "\n")
    rows = mod.parse_frontend_log(log)
    assert rows == []


def test_parse_zero_output_tokens_avoids_division_by_zero(mod, tmp_path):
    zero_out = (
        "2026-05-21T03:56:01Z  INFO http-request: "
        "dynamo_llm::http::service::metrics: request completed "
        "model=m endpoint=chat_completions status=success "
        "elapsed_ms=200 method=POST input_tokens=100 output_tokens=0 "
        'ttft_ms="150.0"'
    )
    log = tmp_path / "frontend.log"
    log.write_text(zero_out + "\n")
    rows = mod.parse_frontend_log(log)
    assert len(rows) == 1
    r = rows[0]
    assert r["output_tokens"] == 0
    # decode is computable (elapsed - ttft) even when output is 0
    assert r["decode_ms"] == pytest.approx(50.0)
    # itl per token is None (would be div by zero)
    assert r["itl_ms_per_token"] is None
    assert r["isl_osl_ratio"] is None


def test_stats_handles_empty_and_some_none(mod):
    s_empty = mod._stats([])
    assert s_empty["n"] == 0
    s_some = mod._stats([1, 2, None, 3, 4, 5])
    assert s_some["n"] == 5
    assert s_some["median"] == 3
    assert s_some["mean"] == pytest.approx(3.0)


def test_write_requests_csv_has_expected_columns(mod, tmp_path):
    rows = [{
        "model": "m", "status": "success",
        "elapsed_ms": 297.0, "ttft_ms": 187.48,
        "decode_ms": 109.52, "itl_ms_per_token": 8.42,
        "input_tokens": 1187, "output_tokens": 13,
        "isl_osl_ratio": 91.31,
    }]
    csv_path = tmp_path / "requests.csv"
    mod.write_requests_csv(rows, csv_path)
    with csv_path.open() as f:
        r = csv.DictReader(f)
        row = next(r)
    assert row["elapsed_ms"] == "297.0"
    assert row["input_tokens"] == "1187"
    assert row["output_tokens"] == "13"
    assert row["isl_osl_ratio"].startswith("91.3")


def test_stats_returns_none_for_empty_input(mod):
    """Empty / all-None input must NOT produce float('nan') — that
    silently spells "nan" into the CSV. Use None as 'no data' sentinel
    so _fmt_stat can map to an empty cell."""
    s = mod._stats([])
    assert s["n"] == 0
    for k in ("mean", "median", "p90", "p99"):
        assert s[k] is None, f"_stats('empty')[{k!r}] should be None, got {s[k]}"


def test_write_stats_csv_empty_rows_writes_empty_cells_not_nan(mod, tmp_path):
    """Regression guard: when no rows are present, summary_stats.csv
    must NOT contain the literal string 'nan' (which is what
    f"{float('nan'):.4f}" produces and which silently breaks
    downstream tools that read the CSV as floats)."""
    csv_path = tmp_path / "summary_stats.csv"
    mod.write_stats_csv([], csv_path)
    text = csv_path.read_text()
    assert "nan" not in text.lower(), (
        f"summary_stats.csv contains literal 'nan' on empty input:\n{text}"
    )
    # Every metric row should have n=0 with empty value cells.
    lines = text.splitlines()
    assert lines[0] == "metric,n,mean,median,p90,p99"
    for line in lines[1:]:
        parts = line.split(",")
        assert parts[1] == "0", f"expected n=0, got {line!r}"


def test_write_stats_csv_lists_all_metrics(mod, tmp_path):
    rows = [{
        "model": "m", "status": "success",
        "elapsed_ms": 297.0, "ttft_ms": 187.48,
        "decode_ms": 109.52, "itl_ms_per_token": 8.42,
        "input_tokens": 1187, "output_tokens": 13,
        "isl_osl_ratio": 91.31,
    }] * 5
    csv_path = tmp_path / "stats.csv"
    mod.write_stats_csv(rows, csv_path)
    text = csv_path.read_text()
    for metric in ("elapsed_ms", "ttft_ms", "decode_ms", "itl_ms_per_token",
                   "input_tokens", "output_tokens", "isl_osl_ratio"):
        assert metric in text


def test_main_returns_nonzero_on_empty_log(mod, tmp_path, capsys):
    """Documented contract: empty / all-non-request input → rc=1 +
    nothing-to-plot message to stderr."""
    log = tmp_path / "frontend.log"
    log.write_text("2026-05-21T03:50:31Z  INFO main: startup\n")  # no request_completed
    out = tmp_path / "figs"
    rc = mod.main(["--input", str(log), "--output", str(out)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "nothing to plot" in err


def test_main_end_to_end(mod, tmp_path, capsys):
    log = tmp_path / "frontend.log"
    log.write_text("\n".join([SAMPLE_LINE, SAMPLE_LINE, SAMPLE_LINE]) + "\n")
    out = tmp_path / "figs"
    rc = mod.main(["--input", str(log), "--output", str(out)])
    assert rc == 0
    files = {p.name for p in out.iterdir()}
    assert "requests.csv" in files
    assert "summary_stats.csv" in files
    assert "fig_e2e_latency.pdf" in files
    assert "fig_ttft.pdf" in files
    assert "fig_itl.pdf" in files
    assert "fig_tokens.pdf" in files
    assert "fig_isl_osl_ratio.pdf" in files
    captured = capsys.readouterr()
    assert "parsed 3 request_completed rows" in captured.out
