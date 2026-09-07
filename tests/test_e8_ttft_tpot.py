"""Tests for scripts/arm/e8_ttft_tpot_by_tokens.py.

No network, no GPU. Loaded via importlib (script, not a package module),
matching this repo's convention for scripts/ tests (see
tests/test_e4_osl_correction.py, tests/test_e6_kv_capacity.py).
"""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "arm" / "e8_ttft_tpot_by_tokens.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e8():
    return _load_module("e8_ttft_tpot_by_tokens", _SCRIPT_PATH)


def _frontend_line(rid, elapsed, ttft, out):
    body = (
        "request completed {"
        f'"request_id":"{rid}","elapsed_ms":{elapsed},'
        f'"ttft_ms":{ttft},"output_tokens":{out}}}'
    )
    return body + "\n"


def _write_frontend(path: Path, rows: list[tuple]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(_frontend_line(*row))


def _sched_line(rid, role, queue_ms):
    return f"SCHED_DELAY request_id={rid} role={role} queue_ms={queue_ms}\n"


def _write_worker_log(path: Path, lines: list[str]) -> Path:
    p = path / "vllm-worker.log"
    p.write_text("".join(lines), encoding="utf-8")
    return p


def _llm_end(rid, input_tok, cache_read=None):
    tokens = {"output": 10, "input": input_tok}
    if cache_read is not None:
        tokens["cache"] = {"read": cache_read}
    return {"ev": "llm.end", "request_id": rid, "tokens": tokens}


def _write_jsonl(path: Path, lines: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, str):
                f.write(line + "\n")
            else:
                f.write(json.dumps(line) + "\n")


# ---------------------------------------------------------------------------
# _bucket / _bucket_order
# ---------------------------------------------------------------------------

class TestBucket:
    BINS = [512, 1024, 2048]

    def test_value_below_first_edge(self, e8):
        assert e8._bucket(100, self.BINS) == "0-512"

    def test_value_exactly_on_edge_goes_to_upper_bucket(self, e8):
        # v < b is strict, so v == 512 does not match the first bin and
        # falls into the next one.
        assert e8._bucket(512, self.BINS) == "512-1024"
        assert e8._bucket(1024, self.BINS) == "1024-2048"

    def test_value_above_last_edge_is_open_ended(self, e8):
        assert e8._bucket(5000, self.BINS) == "2048+"

    def test_zero_value(self, e8):
        assert e8._bucket(0, self.BINS) == "0-512"

    def test_bucket_order_matches_labels(self, e8):
        assert e8._bucket_order(self.BINS) == ["0-512", "512-1024",
                                                "1024-2048", "2048+"]


# ---------------------------------------------------------------------------
# profile_isl
# ---------------------------------------------------------------------------

class TestProfileIsl:
    def test_basic_fields(self, e8, tmp_path):
        f = tmp_path / "p.jsonl"
        _write_jsonl(f, [_llm_end("r1", input_tok=100, cache_read=50)])
        result = e8.profile_isl(f)
        assert result == {
            "r1": {"reprefill_tokens": 100, "cached_tokens": 50,
                   "prompt_tokens": 150},
        }

    def test_missing_cache_dict_defaults_to_zero(self, e8, tmp_path):
        f = tmp_path / "p.jsonl"
        _write_jsonl(f, [_llm_end("r1", input_tok=100)])
        result = e8.profile_isl(f)
        assert result == {
            "r1": {"reprefill_tokens": 100, "cached_tokens": 0,
                   "prompt_tokens": 100},
        }

    def test_llm_end_without_tokens_skipped(self, e8, tmp_path):
        f = tmp_path / "p.jsonl"
        _write_jsonl(f, [
            {"ev": "llm.end", "request_id": "r1"},
            _llm_end("r-ok", input_tok=10),
        ])
        result = e8.profile_isl(f)
        assert result == {
            "r-ok": {"reprefill_tokens": 10, "cached_tokens": 0,
                     "prompt_tokens": 10},
        }

    def test_llm_end_without_request_id_skipped(self, e8, tmp_path):
        f = tmp_path / "p.jsonl"
        _write_jsonl(f, [
            {"ev": "llm.end", "tokens": {"input": 10}},
            _llm_end("r-ok", input_tok=10),
        ])
        result = e8.profile_isl(f)
        assert list(result.keys()) == ["r-ok"]

    def test_malformed_json_line_skipped(self, e8, tmp_path):
        f = tmp_path / "p.jsonl"
        _write_jsonl(f, [
            '{"ev": "llm.end", "request_id": "r1", "tokens": {broken',
            _llm_end("r-ok", input_tok=10),
        ])
        result = e8.profile_isl(f)
        assert list(result.keys()) == ["r-ok"]

    def test_non_llm_end_ignored(self, e8, tmp_path):
        f = tmp_path / "p.jsonl"
        _write_jsonl(f, [
            {"ev": "turn.start", "request_id": "r1", "tokens": {"input": 10}},
            _llm_end("r-ok", input_tok=10),
        ])
        result = e8.profile_isl(f)
        assert list(result.keys()) == ["r-ok"]

    def test_directory_of_files_merged(self, e8, tmp_path):
        d = tmp_path / "profiles"
        d.mkdir()
        _write_jsonl(d / "a.jsonl", [_llm_end("r1", input_tok=10)])
        _write_jsonl(d / "b.jsonl", [_llm_end("r2", input_tok=20)])
        result = e8.profile_isl(d)
        assert set(result.keys()) == {"r1", "r2"}

    def test_non_numeric_input_skipped(self, e8, tmp_path):
        f = tmp_path / "p.jsonl"
        _write_jsonl(f, [
            {"ev": "llm.end", "request_id": "r1",
             "tokens": {"input": "many"}},
            _llm_end("r-ok", input_tok=10),
        ])
        result = e8.profile_isl(f)
        assert list(result.keys()) == ["r-ok"]


# ---------------------------------------------------------------------------
# build_rows
# ---------------------------------------------------------------------------

def _front_row(rid, elapsed_ms, prefill_ms, output_tokens):
    return {
        "request_id": rid,
        "elapsed_ms": elapsed_ms,
        "prefill_ms": prefill_ms,
        "decode_ms": elapsed_ms - prefill_ms,
        "output_tokens": output_tokens,
    }


class TestBuildRows:
    def test_prefill_queue_used_when_present(self, e8, tmp_path):
        ats = _load_module("analyze_turn_scheduling",
                            _REPO_ROOT / "scripts" / "analyze_turn_scheduling.py")
        log = _write_worker_log(tmp_path, [
            _sched_line("r1", "prefill", 5.0),
            _sched_line("r1", "decode", 3.0),
        ])
        sched = ats.load_sched(log)
        front = [_front_row("r1", 100.0, 20.0, 10)]
        rows, rep = e8.build_rows(front, sched, {})
        assert rows[0]["queue_role"] == "prefill"
        assert rows[0]["queue_ms"] == 5.0
        assert rows[0]["ttft_net_ms"] == 15.0
        assert rep["queue_role"] == {"prefill": 1}

    def test_falls_back_to_decode_queue_when_only_decode_present(self, e8, tmp_path):
        ats = _load_module("analyze_turn_scheduling",
                            _REPO_ROOT / "scripts" / "analyze_turn_scheduling.py")
        log = _write_worker_log(tmp_path, [
            _sched_line("r1", "decode", 3.0),
        ])
        sched = ats.load_sched(log)
        front = [_front_row("r1", 100.0, 20.0, 10)]
        rows, rep = e8.build_rows(front, sched, {})
        assert rows[0]["queue_role"] == "decode"
        assert rows[0]["queue_ms"] == 3.0
        assert rows[0]["ttft_net_ms"] == 17.0
        assert rep["queue_role"] == {"decode": 1}

    def test_no_sched_record_leaves_ttft_net_blank(self, e8):
        front = [_front_row("r1", 100.0, 20.0, 10)]
        rows, rep = e8.build_rows(front, {}, {})
        assert rows[0]["ttft_net_ms"] == ""
        assert rows[0]["queue_ms"] == ""
        assert rows[0]["queue_role"] == ""
        assert rep["queued"] == 0

    def test_negative_net_dropped_and_counted(self, e8, tmp_path):
        ats = _load_module("analyze_turn_scheduling",
                            _REPO_ROOT / "scripts" / "analyze_turn_scheduling.py")
        log = _write_worker_log(tmp_path, [
            _sched_line("r1", "prefill", 50.0),  # queue > ttft(20)
        ])
        sched = ats.load_sched(log)
        front = [_front_row("r1", 100.0, 20.0, 10)]
        rows, rep = e8.build_rows(front, sched, {})
        assert rows[0]["ttft_net_ms"] == ""
        assert rep["negative_net"] == 1
        # still counted as queued (a record was found)
        assert rep["queued"] == 1

    def test_tpot_computed_from_decode_ms_and_output_tokens(self, e8):
        front = [_front_row("r1", 100.0, 20.0, output_tokens=5)]
        rows, rep = e8.build_rows(front, {}, {})
        # decode_ms = 100 - 20 = 80; tpot = 80 / (5-1) = 20
        assert rows[0]["tpot_ms"] == 20.0

    def test_tpot_blank_when_output_tokens_le_1(self, e8):
        front = [_front_row("r1", 100.0, 20.0, output_tokens=1)]
        rows, rep = e8.build_rows(front, {}, {})
        assert rows[0]["tpot_ms"] == ""

        front2 = [_front_row("r2", 100.0, 20.0, output_tokens=0)]
        rows2, _ = e8.build_rows(front2, {}, {})
        assert rows2[0]["tpot_ms"] == ""

    def test_isl_fields_populated_and_reported(self, e8):
        front = [_front_row("r1", 100.0, 20.0, 10)]
        isl = {"r1": {"prompt_tokens": 150, "reprefill_tokens": 100,
                      "cached_tokens": 50}}
        rows, rep = e8.build_rows(front, {}, isl)
        assert rows[0]["prompt_tokens"] == 150
        assert rows[0]["reprefill_tokens"] == 100
        assert rows[0]["cached_tokens"] == 50
        assert rep["with_isl"] == 1

    def test_missing_isl_fields_blank(self, e8):
        front = [_front_row("r1", 100.0, 20.0, 10)]
        rows, rep = e8.build_rows(front, {}, {})
        assert rows[0]["prompt_tokens"] == ""
        assert rows[0]["reprefill_tokens"] == ""
        assert rows[0]["cached_tokens"] == ""
        assert rep["with_isl"] == 0


# ---------------------------------------------------------------------------
# bucket_rows
# ---------------------------------------------------------------------------

class TestBucketRows:
    def test_skips_rows_with_blank_token_or_value(self, e8):
        rows = [
            {"tok": "", "val": 5.0},
            {"tok": 100.0, "val": ""},
            {"tok": 100.0, "val": 5.0},
        ]
        out = e8.bucket_rows(rows, "tok", "val", [512], False)
        assert len(out) == 1
        assert out[0]["n"] == 1

    def test_per_token_us_column_present_only_when_requested(self, e8):
        rows = [{"tok": 100.0, "val": 5.0}]
        with_col = e8.bucket_rows(rows, "tok", "val", [512], True)
        without_col = e8.bucket_rows(rows, "tok", "val", [512], False)
        assert "us_per_token_p50" in with_col[0]
        assert "us_per_token_p50" not in without_col[0]

    def test_bucket_ordering_follows_bucket_order(self, e8):
        rows = [
            {"tok": 3000.0, "val": 1.0},  # 2048+
            {"tok": 100.0, "val": 1.0},   # 0-512
            {"tok": 1500.0, "val": 1.0},  # 1024-2048
        ]
        out = e8.bucket_rows(rows, "tok", "val", [512, 1024, 2048], False)
        labels = [r["bucket"] for r in out]
        assert labels == ["0-512", "1024-2048", "2048+"]

    def test_empty_buckets_omitted(self, e8):
        rows = [{"tok": 100.0, "val": 1.0}]
        out = e8.bucket_rows(rows, "tok", "val", [512, 1024], False)
        assert len(out) == 1
        assert out[0]["bucket"] == "0-512"


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------

class TestMain:
    def _setup_fixtures(self, tmp_path):
        frontend = tmp_path / "frontend.log"
        _write_frontend(frontend, [
            ("r1", 100.0, 20.0, 10),
            ("r2", 200.0, 50.0, 5),
        ])

        logdir = tmp_path / "logs"
        logdir.mkdir()
        _write_worker_log(logdir, [
            _sched_line("r1", "prefill", 5.0),
            _sched_line("r2", "decode", 10.0),
        ])

        profdir = tmp_path / "profiles"
        profdir.mkdir()
        _write_jsonl(profdir / "s1.jsonl", [
            _llm_end("r1", input_tok=100, cache_read=50),
            _llm_end("r2", input_tok=200, cache_read=0),
        ])
        return frontend, logdir, profdir

    def test_end_to_end_writes_expected_csvs(self, e8, tmp_path):
        frontend, logdir, profdir = self._setup_fixtures(tmp_path)
        out = tmp_path / "out"
        rc = e8.main([
            "--frontend", str(frontend),
            "--logs", str(logdir),
            "--profiles", str(profdir),
            "--out", str(out),
            "--no-figures",
        ])
        assert rc == 0

        detail = out / "ttft_tpot.csv"
        assert detail.exists()
        with detail.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "request_id", "prompt_tokens", "reprefill_tokens",
                "cached_tokens", "output_tokens", "elapsed_ms", "ttft_ms",
                "queue_ms", "queue_role", "ttft_net_ms", "decode_ms",
                "tpot_ms",
            ]
            rows = list(reader)
            assert len(rows) == 2

        prompt_csv = out / "ttft_by_prompt_tokens.csv"
        reprefill_csv = out / "ttft_by_reprefill.csv"
        tpot_csv = out / "tpot_by_output_tokens.csv"
        assert prompt_csv.exists()
        assert reprefill_csv.exists()
        assert tpot_csv.exists()

        with prompt_csv.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "bucket", "n", "prompt_tokens_p50", "mean_ms", "p50_ms",
                "p90_ms", "us_per_token_p50",
            ]
        with reprefill_csv.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "bucket", "n", "reprefill_tokens_p50", "mean_ms", "p50_ms",
                "p90_ms", "us_per_token_p50",
            ]
        with tpot_csv.open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                "bucket", "n", "output_tokens_p50", "mean_ms", "p50_ms",
                "p90_ms",
            ]

    def test_no_logs_path_returns_zero_with_header_only_ttft_csvs(self, e8, tmp_path):
        frontend, _logdir, profdir = self._setup_fixtures(tmp_path)
        out = tmp_path / "out_nologs"
        rc = e8.main([
            "--frontend", str(frontend),
            "--profiles", str(profdir),
            "--out", str(out),
            "--no-figures",
        ])
        assert rc == 0

        detail = out / "ttft_tpot.csv"
        assert detail.exists()
        with detail.open() as f:
            rows = list(csv.DictReader(f))
        # no SCHED_DELAY records at all -> every row's ttft_net_ms is blank
        assert len(rows) == 2
        assert all(r["ttft_net_ms"] == "" for r in rows)

        prompt_csv = out / "ttft_by_prompt_tokens.csv"
        with prompt_csv.open() as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            data_rows = list(reader)
        assert header == [
            "bucket", "n", "prompt_tokens_p50", "mean_ms", "p50_ms",
            "p90_ms", "us_per_token_p50",
        ]
        # header-only: no bucket data because ttft_net_ms is blank for all rows
        assert data_rows == []

        reprefill_csv = out / "ttft_by_reprefill.csv"
        with reprefill_csv.open() as f:
            data_rows = list(csv.DictReader(f))
        assert data_rows == []
