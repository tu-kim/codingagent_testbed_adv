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

    def test_osl_source_profile_when_output_tokens_frontend_stamped(self, e8):
        # apply_profile_tokens() stamps output_tokens_frontend only on rows
        # it actually corrected; build_rows must key osl_source off that
        # stamp, not off the ISL dict or anything else.
        front = [_front_row("r1", 100.0, 20.0, 10)]
        front[0]["output_tokens_frontend"] = 2  # pretend e4 corrected it
        rows, rep = e8.build_rows(front, {}, {})
        assert rows[0]["osl_source"] == "profile"
        assert rep["osl_frontend"] == 0

    def test_osl_source_frontend_when_not_stamped(self, e8):
        front = [_front_row("r1", 100.0, 20.0, 10)]
        rows, rep = e8.build_rows(front, {}, {})
        assert "output_tokens_frontend" not in front[0]
        assert rows[0]["osl_source"] == "frontend"
        assert rep["osl_frontend"] == 1

    def test_osl_frontend_count_mixed(self, e8):
        front = [_front_row("r1", 100.0, 20.0, 10),
                 _front_row("r2", 100.0, 20.0, 10)]
        front[0]["output_tokens_frontend"] = 3
        rows, rep = e8.build_rows(front, {}, {})
        sources = {r["request_id"]: r["osl_source"] for r in rows}
        assert sources == {"r1": "profile", "r2": "frontend"}
        assert rep["osl_frontend"] == 1


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
                "cached_tokens", "output_tokens", "osl_source", "elapsed_ms",
                "ttft_ms", "queue_ms", "queue_role", "ttft_net_ms",
                "decode_ms", "tpot_ms",
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

    def test_partial_profile_coverage_excludes_uncovered_from_tpot_bucket(
            self, e8, tmp_path):
        # r1 has a profile llm.end record -> osl_source=profile, corrected
        # output_tokens=50 (bucket 32-64). r2 has NO profile record -> it
        # keeps the frontend's (truncated) output_tokens=100, which would
        # land in bucket 64-128 if it were included. Nothing else lands in
        # 64-128, so its absence from tpot_by_output_tokens.csv is the
        # observable signal that the exclusion happened.
        frontend = tmp_path / "frontend.log"
        _write_frontend(frontend, [
            ("r1", 100.0, 20.0, 1),    # frontend OSL is a throwaway stub
            ("r2", 300.0, 20.0, 100),  # frontend OSL is the only one -- used
        ])

        logdir = tmp_path / "logs"
        logdir.mkdir()
        _write_worker_log(logdir, [
            _sched_line("r1", "prefill", 5.0),
            _sched_line("r2", "prefill", 5.0),
        ])

        profdir = tmp_path / "profiles"
        profdir.mkdir()
        _write_jsonl(profdir / "s1.jsonl", [
            _llm_end("r1", input_tok=100, cache_read=0),  # r2 absent
        ])

        out = tmp_path / "out_partial"
        rc = e8.main([
            "--frontend", str(frontend),
            "--logs", str(logdir),
            "--profiles", str(profdir),
            "--out", str(out),
            "--no-figures",
        ])
        assert rc == 0

        with (out / "ttft_tpot.csv").open() as f:
            rows = {r["request_id"]: r for r in csv.DictReader(f)}
        assert len(rows) == 2
        assert rows["r1"]["osl_source"] == "profile"
        assert rows["r1"]["output_tokens"] == "10"  # from _llm_end's tokens.output
        assert rows["r2"]["osl_source"] == "frontend"
        assert rows["r2"]["output_tokens"] == "100"  # untouched truncated value

        with (out / "tpot_by_output_tokens.csv").open() as f:
            bucket_rows = list(csv.DictReader(f))
        buckets = {r["bucket"] for r in bucket_rows}
        # r1's corrected OSL (10) buckets into 8-16; r2's excluded 100 would
        # bucket into 64-128 -- confirm that bucket never appears.
        assert "64-128" not in buckets
        assert buckets == {"8-16"}

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

    def test_no_profiles_leaves_tpot_bucket_csv_header_only(self, e8, tmp_path):
        # Without --profiles, no row's output_tokens_frontend gets stamped,
        # so every row is osl_source=frontend and tpot_rows is empty --
        # tpot_by_output_tokens.csv must still be written, header-only.
        frontend, logdir, _profdir = self._setup_fixtures(tmp_path)
        out = tmp_path / "out_noprofiles"
        rc = e8.main([
            "--frontend", str(frontend),
            "--logs", str(logdir),
            "--out", str(out),
            "--no-figures",
        ])
        assert rc == 0

        with (out / "ttft_tpot.csv").open() as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2
        assert all(r["osl_source"] == "frontend" for r in rows)

        tpot_csv = out / "tpot_by_output_tokens.csv"
        with tpot_csv.open() as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            data_rows = list(reader)
        assert header == [
            "bucket", "n", "output_tokens_p50", "mean_ms", "p50_ms", "p90_ms",
        ]
        assert data_rows == []


# ---------------------------------------------------------------------------
# _ktok
# ---------------------------------------------------------------------------

class TestKtok:
    def test_exact_thousand(self, e8):
        assert e8._ktok(1000) == "1k"

    def test_exact_multiple_of_thousand(self, e8):
        assert e8._ktok(2000) == "2k"

    def test_below_thousand_is_plain(self, e8):
        assert e8._ktok(500) == "500"

    def test_non_multiple_of_thousand_is_plain(self, e8):
        assert e8._ktok(1500) == "1500"

    def test_zero_is_plain(self, e8):
        assert e8._ktok(0) == "0"


# ---------------------------------------------------------------------------
# _range_label
# ---------------------------------------------------------------------------

class TestRangeLabel:
    def test_bucket_one_starts_at_zero_not_one(self, e8):
        # Bucket 1 covers (0, step] but also absorbs 0-token rows, so its
        # label starts at 0 rather than 1.
        assert e8._range_label(1, 1000) == "0-1000"

    def test_later_bucket_is_exclusive_lower_inclusive_upper(self, e8):
        assert e8._range_label(2, 1000) == "1001-2000"
        assert e8._range_label(3, 1000) == "2001-3000"

    def test_non_1000_step(self, e8):
        assert e8._range_label(1, 500) == "0-500"
        assert e8._range_label(2, 500) == "501-1000"
        assert e8._range_label(5, 500) == "2001-2500"


# ---------------------------------------------------------------------------
# grid_rows
# ---------------------------------------------------------------------------

class TestGridRows:
    def test_value_on_step_edge_stays_in_that_bucket(self, e8):
        rows = [{"tok": 2000.0, "val": 5.0}]
        out = e8.grid_rows(rows, "tok", "val", 1000)
        assert len(out) == 1
        assert out[0]["tokens"] == 2000
        assert out[0]["token_range"] == "1001-2000"

    def test_value_just_above_edge_moves_up_a_bucket(self, e8):
        rows = [{"tok": 2001.0, "val": 5.0}]
        out = e8.grid_rows(rows, "tok", "val", 1000)
        assert len(out) == 1
        assert out[0]["tokens"] == 3000
        assert out[0]["token_range"] == "2001-3000"

    def test_zero_token_lands_in_bucket_one(self, e8):
        rows = [{"tok": 0.0, "val": 5.0}]
        out = e8.grid_rows(rows, "tok", "val", 1000)
        assert len(out) == 1
        assert out[0]["tokens"] == 1000
        assert out[0]["token_range"] == "0-1000"

    def test_blank_token_or_value_skipped(self, e8):
        rows = [
            {"tok": "", "val": 5.0},
            {"tok": 100.0, "val": ""},
            {"tok": 100.0, "val": 5.0},
        ]
        out = e8.grid_rows(rows, "tok", "val", 1000)
        assert len(out) == 1
        assert out[0]["n"] == 1

    def test_sorted_ascending_by_tokens(self, e8):
        rows = [
            {"tok": 3000.0, "val": 1.0},
            {"tok": 1000.0, "val": 1.0},
            {"tok": 2000.0, "val": 1.0},
        ]
        out = e8.grid_rows(rows, "tok", "val", 1000)
        assert [r["tokens"] for r in out] == [1000, 2000, 3000]

    def test_row_shape_and_token_col_p50_key(self, e8):
        rows = [
            {"tok": 1000.0, "val": 10.0},
            {"tok": 900.0, "val": 20.0},
        ]
        out = e8.grid_rows(rows, "tok", "val", 1000)
        row = out[0]
        assert row["n"] == 2
        assert "tok_p50" in row
        assert row["mean_ms"] == 15.0
        assert row["p50_ms"] == 15.0
        assert row["p90_ms"] == pytest.approx(19.0)

    def test_us_per_token_blank_when_no_positive_token(self, e8):
        rows = [{"tok": 0.0, "val": 5.0}]
        out = e8.grid_rows(rows, "tok", "val", 1000)
        assert out[0]["us_per_token_p50"] == ""

    def test_us_per_token_populated_for_positive_token(self, e8):
        rows = [{"tok": 1000.0, "val": 2.0}]
        out = e8.grid_rows(rows, "tok", "val", 1000)
        assert out[0]["us_per_token_p50"] == round(1000.0 * 2.0 / 1000.0, 2)


# ---------------------------------------------------------------------------
# grid2d_rows
# ---------------------------------------------------------------------------

class TestGrid2dRows:
    def test_cached_zero_maps_to_bin_zero_not_one(self, e8):
        rows = [{"reprefill_tokens": 500.0, "cached_tokens": 0.0,
                  "ttft_net_ms": 5.0}]
        out = e8.grid2d_rows(rows, 1000)
        assert out[0]["cached_tokens"] == 0

    def test_reprefill_zero_maps_to_bin_zero(self, e8):
        rows = [{"reprefill_tokens": 0.0, "cached_tokens": 500.0,
                  "ttft_net_ms": 5.0}]
        out = e8.grid2d_rows(rows, 1000)
        assert out[0]["reprefill_tokens"] == 0

    def test_positive_values_bucket_up(self, e8):
        rows = [{"reprefill_tokens": 1001.0, "cached_tokens": 2000.0,
                  "ttft_net_ms": 5.0}]
        out = e8.grid2d_rows(rows, 1000)
        assert out[0]["reprefill_tokens"] == 2000
        assert out[0]["cached_tokens"] == 2000

    def test_skips_rows_with_any_blank_field(self, e8):
        rows = [
            {"reprefill_tokens": "", "cached_tokens": 0.0, "ttft_net_ms": 5.0},
            {"reprefill_tokens": 0.0, "cached_tokens": "", "ttft_net_ms": 5.0},
            {"reprefill_tokens": 0.0, "cached_tokens": 0.0, "ttft_net_ms": ""},
            {"reprefill_tokens": 0.0, "cached_tokens": 0.0, "ttft_net_ms": 5.0},
        ]
        out = e8.grid2d_rows(rows, 1000)
        assert len(out) == 1

    def test_sorted_by_key_tuple(self, e8):
        rows = [
            {"reprefill_tokens": 2000.0, "cached_tokens": 0.0, "ttft_net_ms": 1.0},
            {"reprefill_tokens": 0.0, "cached_tokens": 0.0, "ttft_net_ms": 1.0},
            {"reprefill_tokens": 0.0, "cached_tokens": 1000.0, "ttft_net_ms": 1.0},
        ]
        out = e8.grid2d_rows(rows, 1000)
        assert [(r["reprefill_tokens"], r["cached_tokens"]) for r in out] == [
            (0, 0), (0, 1000), (2000, 0),
        ]

    def test_row_carries_n_and_ms_stats(self, e8):
        rows = [
            {"reprefill_tokens": 0.0, "cached_tokens": 0.0, "ttft_net_ms": 10.0},
            {"reprefill_tokens": 0.0, "cached_tokens": 0.0, "ttft_net_ms": 20.0},
        ]
        out = e8.grid2d_rows(rows, 1000)
        assert out[0]["n"] == 2
        assert out[0]["mean_ms"] == 15.0
        assert out[0]["p50_ms"] == 15.0
        assert out[0]["p90_ms"] == pytest.approx(19.0)


# ---------------------------------------------------------------------------
# print_grid / print_grid2d
# ---------------------------------------------------------------------------

class TestPrintGrid:
    def test_empty_rows_prints_no_data(self, e8, capsys):
        e8.print_grid("My Title", [], "reprefill_tokens")
        out = capsys.readouterr().out
        assert "My Title" in out
        assert "(no data)" in out

    def test_header_and_one_row_per_entry(self, e8, capsys):
        rows = e8.grid_rows(
            [{"tok": 1000.0, "val": 10.0}, {"tok": 2500.0, "val": 20.0}],
            "tok", "val", 1000,
        )
        e8.print_grid("Grid Title", rows, "tok")
        out = capsys.readouterr().out
        assert "Grid Title" in out
        assert "tokens" in out
        assert "0-1000" in out
        assert "2001-3000" in out


class TestPrintGrid2d:
    def test_empty_rows_prints_no_data(self, e8, capsys):
        e8.print_grid2d([], 1000)
        out = capsys.readouterr().out
        assert "(no data)" in out

    def test_header_and_rows_present(self, e8, capsys):
        rows = e8.grid2d_rows(
            [{"reprefill_tokens": 500.0, "cached_tokens": 0.0,
              "ttft_net_ms": 15.0},
             {"reprefill_tokens": 1500.0, "cached_tokens": 1000.0,
              "ttft_net_ms": 25.0}],
            1000,
        )
        e8.print_grid2d(rows, 1000)
        out = capsys.readouterr().out
        assert "reused ->" in out
        assert "1k" in out


# ---------------------------------------------------------------------------
# main() -- representative-token grid CSVs
# ---------------------------------------------------------------------------

class TestMainGrid:
    def _setup_fixtures_vllm(self, tmp_path):
        # Same shape as TestMain._setup_fixtures, but with a vllm-*.log
        # filename since analyze_turn_scheduling.load_sched globs
        # "vllm-*.log" and silently returns {} for any other name.
        frontend = tmp_path / "frontend.log"
        _write_frontend(frontend, [
            ("r1", 100.0, 20.0, 10),
            ("r2", 200.0, 50.0, 5),
        ])

        logdir = tmp_path / "logs"
        logdir.mkdir()
        (logdir / "vllm-worker.log").write_text(
            _sched_line("r1", "prefill", 5.0) + _sched_line("r2", "prefill", 5.0),
            encoding="utf-8",
        )

        profdir = tmp_path / "profiles"
        profdir.mkdir()
        _write_jsonl(profdir / "s1.jsonl", [
            _llm_end("r1", input_tok=1000, cache_read=0),
            _llm_end("r2", input_tok=2001, cache_read=0),
        ])
        return frontend, logdir, profdir

    def test_grid_csvs_written_with_expected_headers(self, e8, tmp_path):
        frontend, logdir, profdir = self._setup_fixtures_vllm(tmp_path)
        out = tmp_path / "out_grid"
        rc = e8.main([
            "--frontend", str(frontend),
            "--logs", str(logdir),
            "--profiles", str(profdir),
            "--out", str(out),
            "--no-figures",
        ])
        assert rc == 0

        expected = {
            "prefill_by_reprefill_grid.csv": [
                "token_range", "tokens", "n", "reprefill_tokens_p50",
                "mean_ms", "p50_ms", "p90_ms", "us_per_token_p50",
            ],
            "prefill_by_cached_grid.csv": [
                "token_range", "tokens", "n", "cached_tokens_p50", "mean_ms",
                "p50_ms", "p90_ms", "us_per_token_p50",
            ],
            "prefill_grid_2d.csv": [
                "reprefill_tokens", "cached_tokens", "n", "mean_ms",
                "p50_ms", "p90_ms",
            ],
            "ttft_by_prompt_grid.csv": [
                "token_range", "tokens", "n", "mean_reuse_tokens",
                "mean_reprefill_tokens", "mean_ttft_ms",
            ],
            "tpot_by_prompt_grid.csv": [
                "token_range", "tokens", "n", "mean_output_tokens",
                "mean_tpot_ms",
            ],
        }
        for name, cols in expected.items():
            p = out / name
            assert p.exists()
            with p.open() as f:
                reader = csv.DictReader(f)
                assert reader.fieldnames == cols
                assert len(list(reader)) >= 1

        # the old pre-rename files must not reappear
        assert not (out / "prefill_by_prompt_grid.csv").exists()
        assert not (out / "tpot_by_output_grid.csv").exists()

    def test_grid_step_changes_row_labels(self, e8, tmp_path):
        frontend, logdir, profdir = self._setup_fixtures_vllm(tmp_path)

        out_default = tmp_path / "out_default"
        e8.main([
            "--frontend", str(frontend), "--logs", str(logdir),
            "--profiles", str(profdir), "--out", str(out_default),
            "--no-figures",
        ])
        out_500 = tmp_path / "out_500"
        e8.main([
            "--frontend", str(frontend), "--logs", str(logdir),
            "--profiles", str(profdir), "--out", str(out_500),
            "--no-figures", "--grid-step", "500",
        ])

        with (out_default / "prefill_by_reprefill_grid.csv").open() as f:
            default_labels = [r["token_range"] for r in csv.DictReader(f)]
        with (out_500 / "prefill_by_reprefill_grid.csv").open() as f:
            step500_labels = [r["token_range"] for r in csv.DictReader(f)]

        # r1 (1000 tokens) sits exactly on the 1000 edge -> bucket 1 -> "0-1000"
        # on the 1000-step grid, but bucket 2 -> "501-1000" on the 500-step
        # grid; r2 (2001 tokens) is "2001-3000" on the 1000-step grid but
        # "2001-2500" on the 500-step grid (ceil(2001/500)=5 -> 5*500=2500).
        assert default_labels == ["0-1000", "2001-3000"]
        assert step500_labels == ["501-1000", "2001-2500"]
        assert default_labels != step500_labels

    def test_grid_step_alone_drives_both_ttft_and_tpot_csvs(self, e8, tmp_path):
        # With only --grid-step set (no --ttft-grid-step / --tpot-grid-step),
        # both the prompt-token TTFT grid and the output-token TPOT grid must
        # use it as their band width.
        frontend, logdir, profdir = self._setup_fixtures_vllm(tmp_path)
        out = tmp_path / "out_gridstep_only"
        e8.main([
            "--frontend", str(frontend), "--logs", str(logdir),
            "--profiles", str(profdir), "--out", str(out),
            "--no-figures", "--grid-step", "500",
        ])
        with (out / "ttft_by_prompt_grid.csv").open() as f:
            ttft_ranges = [r["token_range"] for r in csv.DictReader(f)]
        with (out / "tpot_by_prompt_grid.csv").open() as f:
            tpot_ranges = [r["token_range"] for r in csv.DictReader(f)]
        # every band width in both tables must be a multiple of 500
        for label in ttft_ranges + tpot_ranges:
            hi = int(label.split("-")[-1])
            assert hi % 500 == 0

    def test_ttft_and_tpot_grid_step_independently_override_grid_step(
            self, e8, tmp_path):
        # Both ttft_by_prompt_grid.csv and tpot_by_prompt_grid.csv now
        # bucket on the SAME axis (total prompt tokens: r1=1000, r2=2001
        # from _setup_fixtures_vllm's cache_read=0 profiles), so this test
        # exercises --ttft-grid-step and --tpot-grid-step producing
        # DIFFERENT band widths over that one axis.
        frontend, logdir, profdir = self._setup_fixtures_vllm(tmp_path)
        out = tmp_path / "out_split_steps"
        e8.main([
            "--frontend", str(frontend), "--logs", str(logdir),
            "--profiles", str(profdir), "--out", str(out),
            "--no-figures",
            "--grid-step", "1000",
            "--ttft-grid-step", "2000",
            "--tpot-grid-step", "3",
        ])
        with (out / "ttft_by_prompt_grid.csv").open() as f:
            ttft_rows = list(csv.DictReader(f))
        with (out / "tpot_by_prompt_grid.csv").open() as f:
            tpot_rows = list(csv.DictReader(f))

        # TTFT bands are on the 2000-wide grid, not the 1000-wide --grid-step
        # default: every upper edge is a multiple of 2000.
        for r in ttft_rows:
            assert int(r["tokens"]) % 2000 == 0
        # TPOT bands are on the 3-wide grid over prompt tokens: r1's prompt
        # (1000) -> ceil(1000/3)=334 -> upper edge 1002; r2's prompt (2001)
        # -> ceil(2001/3)=667 -> upper edge 2001 exactly. Both distinct from
        # --grid-step(1000) and --ttft-grid-step(2000) multiples.
        for r in tpot_rows:
            assert int(r["tokens"]) % 3 == 0
        assert {int(r["tokens"]) for r in tpot_rows} == {1002, 2001}


# ---------------------------------------------------------------------------
# prompt_grid_rows
# ---------------------------------------------------------------------------

def _prompt_row(prompt_tokens, cached_tokens, reprefill_tokens, ttft_net_ms):
    return {
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "reprefill_tokens": reprefill_tokens,
        "ttft_net_ms": ttft_net_ms,
    }


class TestPromptGridRows:
    def test_value_on_step_edge_stays_in_that_bucket(self, e8):
        rows = [_prompt_row(2000.0, 500.0, 1500.0, 30.0)]
        out = e8.prompt_grid_rows(rows, 1000)
        assert len(out) == 1
        assert out[0]["tokens"] == 2000
        assert out[0]["token_range"] == "1001-2000"

    def test_value_just_above_edge_moves_up_a_bucket(self, e8):
        rows = [_prompt_row(2001.0, 500.0, 1501.0, 30.0)]
        out = e8.prompt_grid_rows(rows, 1000)
        assert out[0]["tokens"] == 3000
        assert out[0]["token_range"] == "2001-3000"

    def test_mean_arithmetic(self, e8):
        rows = [
            _prompt_row(1000.0, 400.0, 600.0, 10.0),
            _prompt_row(1000.0, 600.0, 400.0, 30.0),
        ]
        out = e8.prompt_grid_rows(rows, 1000)
        assert len(out) == 1
        row = out[0]
        assert row["n"] == 2
        assert row["mean_reuse_tokens"] == 500.0
        assert row["mean_reprefill_tokens"] == 500.0
        assert row["mean_ttft_ms"] == 20.0

    def test_skips_rows_with_any_blank_field(self, e8):
        rows = [
            _prompt_row("", 400.0, 600.0, 10.0),
            _prompt_row(1000.0, "", 600.0, 10.0),
            _prompt_row(1000.0, 400.0, "", 10.0),
            _prompt_row(1000.0, 400.0, 600.0, ""),
            _prompt_row(1000.0, 400.0, 600.0, 10.0),
        ]
        out = e8.prompt_grid_rows(rows, 1000)
        assert len(out) == 1
        assert out[0]["n"] == 1

    def test_ascending_order(self, e8):
        rows = [
            _prompt_row(3000.0, 0.0, 3000.0, 1.0),
            _prompt_row(1000.0, 0.0, 1000.0, 1.0),
            _prompt_row(2000.0, 0.0, 2000.0, 1.0),
        ]
        out = e8.prompt_grid_rows(rows, 1000)
        assert [r["tokens"] for r in out] == [1000, 2000, 3000]

    def test_zero_prompt_tokens_lands_in_bucket_one(self, e8):
        rows = [_prompt_row(0.0, 0.0, 0.0, 5.0)]
        out = e8.prompt_grid_rows(rows, 1000)
        assert out[0]["tokens"] == 1000

    def test_empty_input_returns_empty(self, e8):
        assert e8.prompt_grid_rows([], 1000) == []


# ---------------------------------------------------------------------------
# tpot_grid_rows
#
# Buckets on `prompt_tokens` (total ISL), NOT `output_tokens` -- per-token
# decode cost tracks the context the attention step reads (dominated by the
# prompt), not how many decode steps ran. `output_tokens` is read only to
# compute the along-for-the-ride `mean_output_tokens` field; it does not
# gate which bucket a row lands in.
# ---------------------------------------------------------------------------

def _tpot_row(prompt_tokens, output_tokens, tpot_ms):
    return {"prompt_tokens": prompt_tokens, "output_tokens": output_tokens,
            "tpot_ms": tpot_ms}


class TestTpotGridRows:
    def test_value_on_step_edge_stays_in_that_bucket(self, e8):
        rows = [_tpot_row(1000.0, 10.0, 5.0)]
        out = e8.tpot_grid_rows(rows, 1000)
        assert len(out) == 1
        assert out[0]["tokens"] == 1000
        assert out[0]["token_range"] == "0-1000"

    def test_value_just_above_edge_moves_up_a_bucket(self, e8):
        rows = [_tpot_row(1001.0, 10.0, 5.0)]
        out = e8.tpot_grid_rows(rows, 1000)
        assert out[0]["tokens"] == 2000
        assert out[0]["token_range"] == "1001-2000"

    def test_mean_arithmetic(self, e8):
        rows = [_tpot_row(500.0, 20.0, 10.0), _tpot_row(600.0, 40.0, 30.0)]
        out = e8.tpot_grid_rows(rows, 1000)
        assert len(out) == 1
        assert out[0]["n"] == 2
        assert out[0]["mean_output_tokens"] == 30.0
        assert out[0]["mean_tpot_ms"] == 20.0

    def test_blank_prompt_tokens_or_tpot_skipped(self, e8):
        # Blankness of prompt_tokens/tpot_ms gates inclusion. output_tokens
        # blankness does NOT gate inclusion (it only feeds the mean) -- see
        # the module docstring; every included row here has a real
        # output_tokens so the mean stays well-defined.
        rows = [
            {"prompt_tokens": "", "output_tokens": 10.0, "tpot_ms": 5.0},
            {"prompt_tokens": 100.0, "output_tokens": 10.0, "tpot_ms": ""},
            _tpot_row(100.0, 10.0, 5.0),
        ]
        out = e8.tpot_grid_rows(rows, 1000)
        assert len(out) == 1
        assert out[0]["n"] == 1

    def test_ascending_order(self, e8):
        rows = [_tpot_row(3000.0, 10.0, 1.0), _tpot_row(1000.0, 10.0, 1.0),
                _tpot_row(2000.0, 10.0, 1.0)]
        out = e8.tpot_grid_rows(rows, 1000)
        assert [r["tokens"] for r in out] == [1000, 2000, 3000]

    def test_zero_prompt_tokens_lands_in_bucket_one(self, e8):
        rows = [_tpot_row(0.0, 10.0, 5.0)]
        out = e8.tpot_grid_rows(rows, 1000)
        assert out[0]["tokens"] == 1000

    def test_empty_input_returns_empty(self, e8):
        assert e8.tpot_grid_rows([], 1000) == []

    def test_same_prompt_band_different_output_tokens_merge_into_one_row(
            self, e8):
        # Two requests with the same prompt-token band but different
        # output_tokens must land in ONE grid row (bucketing is on
        # prompt_tokens only), with mean_output_tokens averaging the two.
        rows = [
            _tpot_row(100.0, 10.0, 8.0),
            _tpot_row(150.0, 50.0, 12.0),
        ]
        out = e8.tpot_grid_rows(rows, 1000)
        assert len(out) == 1
        row = out[0]
        assert row["n"] == 2
        assert row["mean_output_tokens"] == 30.0
        assert row["mean_tpot_ms"] == 10.0


# ---------------------------------------------------------------------------
# print_prompt_grid / print_tpot_grid
# ---------------------------------------------------------------------------

class TestPrintPromptGrid:
    def test_empty_rows_prints_no_data(self, e8, capsys):
        e8.print_prompt_grid([], 1000)
        out = capsys.readouterr().out
        assert "TTFT by total prompt tokens" in out
        assert "(no data)" in out

    def test_header_and_row_present(self, e8, capsys):
        rows = e8.prompt_grid_rows(
            [_prompt_row(1000.0, 400.0, 600.0, 12.5)], 1000)
        e8.print_prompt_grid(rows, 1000)
        out = capsys.readouterr().out
        assert "mean_ttft_ms" in out
        assert "0-1000" in out


class TestPrintTpotGrid:
    def test_empty_rows_prints_no_data(self, e8, capsys):
        e8.print_tpot_grid([], 1000)
        out = capsys.readouterr().out
        assert "TPOT by total prompt tokens" in out
        assert "(no data)" in out

    def test_header_and_row_present(self, e8, capsys):
        rows = e8.tpot_grid_rows([_tpot_row(1000.0, 20.0, 12.5)], 1000)
        e8.print_tpot_grid(rows, 1000)
        out = capsys.readouterr().out
        assert "mean_output" in out
        assert "mean_tpot_ms" in out
        assert "0-1000" in out


# ---------------------------------------------------------------------------
# main() -- --min-tpot-ms TPOT floor filter
#
# Fixtures below control tpot_ms precisely via elapsed/ttft (-> decode_ms)
# and a profile llm.end `tokens.output` value (-> corrected output_tokens,
# via e4.apply_profile_tokens): tpot_ms = decode_ms / max(output_tokens-1, 1).
# Each request also gets a distinct `input_tok` so its prompt-token grid
# bucket is unique and its presence/absence in tpot_by_prompt_grid.csv is
# individually observable.
# ---------------------------------------------------------------------------

def _llm_end_out(rid, input_tok, output_tok, cache_read=0):
    return {"ev": "llm.end", "request_id": rid,
            "tokens": {"output": output_tok, "input": input_tok,
                       "cache": {"read": cache_read}}}


class TestMainMinTpotFilter:
    def _setup(self, tmp_path, *, include_blank=False):
        # r1: decode_ms=1.0,  output=11 -> tpot_ms=0.1  (below default 1.0)
        # r2: decode_ms=80.0, output=5  -> tpot_ms=20.0 (kept at default;
        #                                                  dropped at 25)
        # r3 (optional): decode_ms=50.0, output=1 -> tpot_ms="" (blank --
        #                                             output_tokens<=1)
        frontend = tmp_path / "frontend.log"
        front_rows = [
            ("r1", 101.0, 100.0, 999),
            ("r2", 180.0, 100.0, 999),
        ]
        prof_recs = [_llm_end_out("r1", input_tok=1000, output_tok=11),
                     _llm_end_out("r2", input_tok=2000, output_tok=5)]
        if include_blank:
            front_rows.append(("r3", 150.0, 100.0, 999))
            prof_recs.append(_llm_end_out("r3", input_tok=3000, output_tok=1))
        _write_frontend(frontend, front_rows)
        profdir = tmp_path / "profiles"
        profdir.mkdir()
        _write_jsonl(profdir / "p.jsonl", prof_recs)
        return frontend, profdir

    def test_default_threshold_drops_from_grid_kept_in_detail_csv(
            self, e8, tmp_path, capsys):
        frontend, profdir = self._setup(tmp_path)
        out = tmp_path / "out"
        rc = e8.main([
            "--frontend", str(frontend), "--profiles", str(profdir),
            "--out", str(out), "--no-figures",
        ])
        assert rc == 0
        stdout = capsys.readouterr().out
        assert ("excluded 1 requests from TPOT: tpot_ms < 1 ms/token"
                in stdout)

        # dropped rows stay in ttft_tpot.csv with their real tpot_ms
        with (out / "ttft_tpot.csv").open() as f:
            rows = {r["request_id"]: r for r in csv.DictReader(f)}
        assert rows["r1"]["tpot_ms"] == "0.1"
        assert rows["r2"]["tpot_ms"] == "20.0"

        # ... but r1's prompt-token band (1000) is absent from the TPOT
        # grid, while r2's (2000) survives.
        with (out / "tpot_by_prompt_grid.csv").open() as f:
            tokens = {r["tokens"] for r in csv.DictReader(f)}
        assert tokens == {"2000"}

    def test_min_tpot_ms_zero_disables_filter(self, e8, tmp_path, capsys):
        frontend, profdir = self._setup(tmp_path)
        out = tmp_path / "out_zero"
        rc = e8.main([
            "--frontend", str(frontend), "--profiles", str(profdir),
            "--out", str(out), "--no-figures", "--min-tpot-ms", "0",
        ])
        assert rc == 0
        stdout = capsys.readouterr().out
        assert "tpot_ms <" not in stdout  # filter message never printed

        with (out / "tpot_by_prompt_grid.csv").open() as f:
            tokens = {r["tokens"] for r in csv.DictReader(f)}
        assert tokens == {"1000", "2000"}

    def test_custom_threshold_drops_more(self, e8, tmp_path, capsys):
        frontend, profdir = self._setup(tmp_path)
        out = tmp_path / "out_25"
        rc = e8.main([
            "--frontend", str(frontend), "--profiles", str(profdir),
            "--out", str(out), "--no-figures", "--min-tpot-ms", "25",
        ])
        assert rc == 0
        stdout = capsys.readouterr().out
        assert ("excluded 2 requests from TPOT: tpot_ms < 25 ms/token"
                in stdout)

        # both r1 (0.1) and r2 (20.0) are now below 25 -> grid is empty
        with (out / "tpot_by_prompt_grid.csv").open() as f:
            assert list(csv.DictReader(f)) == []

    def test_blank_tpot_row_not_counted_and_not_treated_as_filtered(
            self, e8, tmp_path, capsys):
        # r3's tpot_ms is blank (output_tokens=1 -> no inter-token interval
        # to measure). The filter must skip it entirely: not counted in
        # n_fast (else the message would say "excluded 2", not "excluded
        # 1"), and not dropped from tpot_rows by the list-comprehension
        # guard (`r["tpot_ms"] == "" or float(...) >= threshold`) -- a
        # naive `float(r["tpot_ms"])` on the blank string would also raise
        # ValueError, so this doubles as a crash regression check.
        frontend, profdir = self._setup(tmp_path, include_blank=True)
        out = tmp_path / "out_blank"
        rc = e8.main([
            "--frontend", str(frontend), "--profiles", str(profdir),
            "--out", str(out), "--no-figures",
        ])
        assert rc == 0
        stdout = capsys.readouterr().out
        assert ("excluded 1 requests from TPOT: tpot_ms < 1 ms/token"
                in stdout)

        with (out / "ttft_tpot.csv").open() as f:
            rows = {r["request_id"]: r for r in csv.DictReader(f)}
        assert len(rows) == 3
        assert rows["r3"]["tpot_ms"] == ""
        assert rows["r3"]["osl_source"] == "profile"


# ---------------------------------------------------------------------------
# fig_tpot -- plots against prompt_tokens (not output_tokens)
# ---------------------------------------------------------------------------

class TestFigTpot:
    def test_reads_prompt_tokens_not_output_tokens(self, e8, tmp_path):
        pytest.importorskip("matplotlib")
        rows = [
            {"prompt_tokens": 100.0, "tpot_ms": 5.0},
            {"prompt_tokens": 200.0, "tpot_ms": 10.0},
        ]
        out = tmp_path / "fig2.pdf"
        # Rows deliberately carry no "output_tokens" key -- if fig_tpot
        # still keyed its x-axis off output_tokens internally this would
        # raise KeyError instead of writing the file.
        e8.fig_tpot(rows, [], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_output_tokens_only_rows_raise(self, e8, tmp_path):
        # Companion negative check: rows with ONLY output_tokens (no
        # prompt_tokens) must fail, proving the function is not tolerant
        # of the old axis as a fallback.
        pytest.importorskip("matplotlib")
        rows = [{"output_tokens": 100.0, "tpot_ms": 5.0}]
        with pytest.raises(KeyError):
            e8.fig_tpot(rows, [], tmp_path / "fig2_bad.pdf")


class TestMainFigureFilenames:
    def test_fig2_filename_is_prompt_tokens(self, e8, tmp_path):
        pytest.importorskip("matplotlib")
        frontend = tmp_path / "frontend.log"
        _write_frontend(frontend, [("r1", 100.0, 20.0, 10)])
        profdir = tmp_path / "profiles"
        profdir.mkdir()
        _write_jsonl(profdir / "p.jsonl",
                     [_llm_end("r1", input_tok=100, cache_read=0)])
        out = tmp_path / "out_figs"
        rc = e8.main([
            "--frontend", str(frontend),
            "--profiles", str(profdir),
            "--out", str(out),
        ])
        assert rc == 0
        assert (out / "fig2_tpot_vs_prompt_tokens.pdf").exists()
        assert not (out / "fig2_tpot_vs_output_tokens.pdf").exists()


# ---------------------------------------------------------------------------
# fig_ttft_plane (matplotlib-dependent)
# ---------------------------------------------------------------------------

class TestFigTtftPlane:
    def test_writes_file_with_data(self, e8, tmp_path):
        pytest.importorskip("matplotlib")
        rows = [
            {"cached_tokens": 100.0, "reprefill_tokens": 200.0,
             "ttft_net_ms": 15.0},
            {"cached_tokens": 300.0, "reprefill_tokens": 400.0,
             "ttft_net_ms": 45.0},
        ]
        out = tmp_path / "fig3.pdf"
        e8.fig_ttft_plane(rows, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_all_blank_or_non_positive_still_writes_file(self, e8, tmp_path):
        pytest.importorskip("matplotlib")
        rows = [
            {"cached_tokens": "", "reprefill_tokens": 200.0,
             "ttft_net_ms": 15.0},
            {"cached_tokens": 100.0, "reprefill_tokens": "",
             "ttft_net_ms": 15.0},
            {"cached_tokens": 100.0, "reprefill_tokens": 200.0,
             "ttft_net_ms": ""},
            {"cached_tokens": 100.0, "reprefill_tokens": 200.0,
             "ttft_net_ms": 0.0},
            {"cached_tokens": 100.0, "reprefill_tokens": 200.0,
             "ttft_net_ms": -5.0},
        ]
        out = tmp_path / "fig3_empty.pdf"
        e8.fig_ttft_plane(rows, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# reuse_reprefill_points
# ---------------------------------------------------------------------------

class TestReuseReprefillPoints:
    def test_both_present_rows_kept_in_order(self, e8):
        rows = [
            {"cached_tokens": 5.0, "reprefill_tokens": 10.0},
            {"cached_tokens": 20.0, "reprefill_tokens": 1.0},
        ]
        assert e8.reuse_reprefill_points(rows) == [(5.0, 10.0), (20.0, 1.0)]

    def test_blank_cached_tokens_skipped(self, e8):
        rows = [
            {"cached_tokens": "", "reprefill_tokens": 10.0},
            {"cached_tokens": 5.0, "reprefill_tokens": 10.0},
        ]
        assert e8.reuse_reprefill_points(rows) == [(5.0, 10.0)]

    def test_blank_reprefill_tokens_skipped(self, e8):
        rows = [
            {"cached_tokens": 5.0, "reprefill_tokens": ""},
            {"cached_tokens": 5.0, "reprefill_tokens": 10.0},
        ]
        assert e8.reuse_reprefill_points(rows) == [(5.0, 10.0)]

    def test_missing_key_skipped(self, e8):
        # .get() rather than [] -- a row missing either field entirely (not
        # just blank) must be skipped, not raise KeyError.
        rows = [
            {"reprefill_tokens": 10.0},
            {"cached_tokens": 5.0},
            {"cached_tokens": 5.0, "reprefill_tokens": 10.0},
        ]
        assert e8.reuse_reprefill_points(rows) == [(5.0, 10.0)]

    def test_empty_input_returns_empty(self, e8):
        assert e8.reuse_reprefill_points([]) == []

    def test_zero_valued_fields_are_not_blank(self, e8):
        # 0.0 is a real value, not a missing one -- must not be treated the
        # same as "" (an `if not c` guard would wrongly drop it).
        rows = [{"cached_tokens": 0.0, "reprefill_tokens": 0.0}]
        assert e8.reuse_reprefill_points(rows) == [(0.0, 0.0)]


# ---------------------------------------------------------------------------
# fig_reuse_reprefill_density (matplotlib-dependent)
# ---------------------------------------------------------------------------

class TestFigReuseReprefillDensity:
    def test_writes_nonempty_file_for_normal_input(self, e8, tmp_path):
        pytest.importorskip("matplotlib")
        rows = [
            {"cached_tokens": float(i * 10), "reprefill_tokens": float(200 - i * 5)}
            for i in range(20)
        ]
        out = tmp_path / "fig4.pdf"
        e8.fig_reuse_reprefill_density(rows, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_input_writes_no_data_file_without_raising(self, e8, tmp_path):
        pytest.importorskip("matplotlib")
        out = tmp_path / "fig4_empty.pdf"
        e8.fig_reuse_reprefill_density([], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_single_point_does_not_raise(self, e8, tmp_path):
        pytest.importorskip("matplotlib")
        rows = [{"cached_tokens": 100.0, "reprefill_tokens": 200.0}]
        out = tmp_path / "fig4_single.pdf"
        e8.fig_reuse_reprefill_density(rows, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_all_zero_point_does_not_divide_by_zero(self, e8, tmp_path):
        # hi = max(max(cs), max(ns)) or 1.0 -- with a single (0, 0) point,
        # max(cs)=max(ns)=0 (falsy), so `or 1.0` substitutes a safe nonzero
        # axis limit instead of dividing by zero when normalizing xlim/ylim
        # and the diagonal-line loop bounds. Traced directly (not just
        # asserted) before writing this test: raises nothing.
        rows = [{"cached_tokens": 0.0, "reprefill_tokens": 0.0}]
        out = tmp_path / "fig4_allzero.pdf"
        e8.fig_reuse_reprefill_density(rows, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# main() -- fig4 filename
# ---------------------------------------------------------------------------

class TestMainFig4Filename:
    def test_fig4_reuse_reprefill_density_written(self, e8, tmp_path):
        pytest.importorskip("matplotlib")
        frontend = tmp_path / "frontend.log"
        _write_frontend(frontend, [("r1", 100.0, 20.0, 10)])
        profdir = tmp_path / "profiles"
        profdir.mkdir()
        _write_jsonl(profdir / "p.jsonl",
                     [_llm_end("r1", input_tok=100, cache_read=0)])
        out = tmp_path / "out_fig4"
        rc = e8.main([
            "--frontend", str(frontend),
            "--profiles", str(profdir),
            "--out", str(out),
        ])
        assert rc == 0
        assert (out / "fig4_reuse_reprefill_density.pdf").exists()
