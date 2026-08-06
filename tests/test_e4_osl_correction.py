"""Tests for the OSL (output-tokens) correction helpers in
scripts/arm/e4_prefill_decode.py: `profile_output_tokens` and
`apply_profile_tokens`.

No network, no GPU. Loaded via importlib (script, not a package module),
matching this repo's convention for scripts/ tests (see
tests/test_e6_kv_capacity.py, tests/test_e3_compare_runs.py).

Background (why this correction exists): dynamo frontend.log's
output_tokens comes from ResponseMetricCollector::Drop writing self.osl
onto the span -- a value that races the cancel path (disconnect.rs) and
can freeze at a small partial count (observed 1..32) instead of the real
OSL. The profile NDJSON's `llm.end` event carries `tokens.output` from an
independent usage-chunk accumulator (up to 32000 observed) and is the
trustworthy source. Without this correction, itl_ms/tpot blow up into the
tens of thousands of ms because decode_ms is divided by a near-zero
output_tokens count.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "arm" / "e4_prefill_decode.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e4():
    return _load_module("e4_prefill_decode_osl", _SCRIPT_PATH)


def _write_jsonl(path: Path, lines: list) -> None:
    with path.open("w", encoding="utf-8") as f:
        for line in lines:
            if isinstance(line, str):
                f.write(line + "\n")
            else:
                f.write(json.dumps(line) + "\n")


def _llm_end(request_id, output, ev="llm.end"):
    return {"ev": ev, "request_id": request_id, "tokens": {"output": output}}


# ---------------------------------------------------------------------------
# profile_output_tokens
# ---------------------------------------------------------------------------

class TestProfileOutputTokens:
    def test_single_file(self, e4, tmp_path):
        f = tmp_path / "profile.jsonl"
        _write_jsonl(f, [_llm_end("req-1", 32000), _llm_end("req-2", 500)])
        result = e4.profile_output_tokens(f)
        assert result == {"req-1": 32000, "req-2": 500}

    def test_directory_of_files_merged(self, e4, tmp_path):
        d = tmp_path / "profiles"
        d.mkdir()
        _write_jsonl(d / "a.jsonl", [_llm_end("req-1", 100)])
        _write_jsonl(d / "b.jsonl", [_llm_end("req-2", 200)])
        result = e4.profile_output_tokens(d)
        assert result == {"req-1": 100, "req-2": 200}

    def test_non_llm_end_events_ignored(self, e4, tmp_path):
        f = tmp_path / "profile.jsonl"
        _write_jsonl(f, [
            {"ev": "turn.start", "request_id": "req-1", "tokens": {"output": 999}},
            {"ev": "tool.end", "request_id": "req-2", "tokens": {"output": 999}},
            _llm_end("req-3", 50),
        ])
        result = e4.profile_output_tokens(f)
        assert result == {"req-3": 50}

    def test_missing_request_id_skipped(self, e4, tmp_path):
        f = tmp_path / "profile.jsonl"
        _write_jsonl(f, [
            {"ev": "llm.end", "tokens": {"output": 50}},
            {"ev": "llm.end", "request_id": "", "tokens": {"output": 50}},
            _llm_end("req-ok", 50),
        ])
        result = e4.profile_output_tokens(f)
        assert result == {"req-ok": 50}

    def test_missing_or_non_dict_tokens_skipped(self, e4, tmp_path):
        f = tmp_path / "profile.jsonl"
        _write_jsonl(f, [
            {"ev": "llm.end", "request_id": "req-1"},
            {"ev": "llm.end", "request_id": "req-2", "tokens": "not-a-dict"},
            {"ev": "llm.end", "request_id": "req-3", "tokens": {"input": 10}},
            _llm_end("req-ok", 50),
        ])
        result = e4.profile_output_tokens(f)
        assert result == {"req-ok": 50}

    def test_non_numeric_output_skipped(self, e4, tmp_path):
        f = tmp_path / "profile.jsonl"
        _write_jsonl(f, [
            {"ev": "llm.end", "request_id": "req-1", "tokens": {"output": "many"}},
            {"ev": "llm.end", "request_id": "req-2", "tokens": {"output": None}},
            _llm_end("req-ok", 50),
        ])
        result = e4.profile_output_tokens(f)
        assert result == {"req-ok": 50}

    def test_malformed_json_line_ignored(self, e4, tmp_path):
        f = tmp_path / "profile.jsonl"
        _write_jsonl(f, [
            '{"ev": "llm.end", "request_id": "req-1", "tokens": {broken',
            _llm_end("req-ok", 50),
        ])
        result = e4.profile_output_tokens(f)
        assert result == {"req-ok": 50}

    def test_blank_lines_ignored(self, e4, tmp_path):
        f = tmp_path / "profile.jsonl"
        _write_jsonl(f, ["", "   ", _llm_end("req-ok", 50)])
        result = e4.profile_output_tokens(f)
        assert result == {"req-ok": 50}

    def test_empty_directory_yields_empty_map(self, e4, tmp_path):
        d = tmp_path / "profiles"
        d.mkdir()
        result = e4.profile_output_tokens(d)
        assert result == {}


# ---------------------------------------------------------------------------
# apply_profile_tokens
# ---------------------------------------------------------------------------

class TestApplyProfileTokens:
    def test_overwrite_and_preserve_original(self, e4):
        rows = [{"request_id": "req-1", "output_tokens": 2, "decode_ms": 100.0}]
        by_rid = {"req-1": 32000}
        e4.apply_profile_tokens(rows, by_rid)
        row = rows[0]
        assert row["output_tokens"] == 32000
        assert row["output_tokens_frontend"] == 2

    def test_recomputes_itl_when_decode_ms_present(self, e4):
        rows = [{"request_id": "req-1", "output_tokens": 2, "decode_ms": 100.0,
                  "itl_ms": 100.0}]
        e4.apply_profile_tokens(rows, {"req-1": 51})
        # decode_ms / max(o - 1, 1) = 100 / 50 = 2.0
        assert rows[0]["itl_ms"] == pytest.approx(2.0)

    def test_no_decode_ms_key_skips_itl_recompute(self, e4):
        # e6's frontend rows carry no decode_ms.
        rows = [{"request_id": "req-1", "output_tokens": 2}]
        report = e4.apply_profile_tokens(rows, {"req-1": 500})
        assert "itl_ms" not in rows[0]
        assert rows[0]["output_tokens"] == 500
        assert report["n_fixed"] == 1

    def test_unmatched_row_left_untouched(self, e4):
        rows = [{"request_id": "req-unmatched", "output_tokens": 2,
                  "decode_ms": 100.0, "itl_ms": 100.0}]
        report = e4.apply_profile_tokens(rows, {"req-other": 500})
        assert rows[0]["output_tokens"] == 2
        assert rows[0]["itl_ms"] == 100.0
        assert "output_tokens_frontend" not in rows[0]
        assert report["n_fixed"] == 0

    def test_report_counts_and_median_growth(self, e4):
        rows = [
            {"request_id": "req-1", "output_tokens": 2, "decode_ms": 100.0},   # grows 2->20 (10x)
            {"request_id": "req-2", "output_tokens": 10, "decode_ms": 100.0},  # grows 10->100 (10x)
            {"request_id": "req-3", "output_tokens": 100, "decode_ms": 100.0},  # shrinks/same, not "grew"
            {"request_id": "req-4", "output_tokens": 5, "decode_ms": 100.0},   # not matched
        ]
        by_rid = {"req-1": 20, "req-2": 100, "req-3": 50}
        report = e4.apply_profile_tokens(rows, by_rid)
        assert report["n_fixed"] == 3
        assert report["n_grew"] == 2
        assert report["median_growth"] == pytest.approx(10.0)

    def test_report_median_growth_none_when_nothing_grew(self, e4):
        rows = [{"request_id": "req-1", "output_tokens": 100, "decode_ms": 100.0}]
        report = e4.apply_profile_tokens(rows, {"req-1": 50})
        assert report["n_fixed"] == 1
        assert report["n_grew"] == 0
        assert report["median_growth"] is None

    def test_itl_regression_recovers_to_normal_range(self, e4):
        """Regression: the exact failure mode from the docstring. A
        cancel-path race freezes frontend output_tokens at 2, so
        decode_ms/max(o-1,1) = 62000/1 = 62000ms itl (tens-of-thousands-ms
        blowup). Wait -- with o=2, max(o-1,1)=1, itl = 62000/1 = 62000ms.
        The corrected profile value (32000 real tokens) recovers itl to a
        normal ~1.94ms/token range.
        """
        rows = [{"request_id": "req-1", "output_tokens": 2,
                  "decode_ms": 62000.0,
                  "itl_ms": 62000.0 / max(2 - 1, 1)}]
        assert rows[0]["itl_ms"] == pytest.approx(62000.0)

        by_rid = {"req-1": 32000}
        report = e4.apply_profile_tokens(rows, by_rid)

        assert rows[0]["output_tokens"] == 32000
        assert rows[0]["output_tokens_frontend"] == 2
        expected_itl = 62000.0 / max(32000 - 1, 1)
        assert rows[0]["itl_ms"] == pytest.approx(expected_itl)
        assert rows[0]["itl_ms"] < 5.0  # recovered to normal per-token range
        assert report["n_fixed"] == 1
        assert report["n_grew"] == 1
