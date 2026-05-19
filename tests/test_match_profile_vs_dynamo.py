"""Tests for scripts/match_profile_vs_dynamo.py.

Lives in tests/ even though the script is in scripts/ (not part of the
installed `testbed` package) -- imported via importlib to keep the
script self-contained and runnable directly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "match_profile_vs_dynamo.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("match_profile_vs_dynamo", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["match_profile_vs_dynamo"] = module
    spec.loader.exec_module(module)
    return module


# Real `request completed` line shape pasted from a live dynamo frontend log.
SAMPLE_DYNAMO_LINE = (
    'INFO http-request: dynamo_llm::http::service::metrics: request completed '
    "request_id=e5bdcfe9-230c-4b4e-ad86-4c22e2535eb7 "
    "model=qwen3-coder-30b-a3b-instruct-fp8 "
    "endpoint=chat_completions request_type=stream status=success "
    "elapsed_ms=1628 method=POST uri=/v1/chat/completions version=HTTP/1.1 "
    "request_id=ff9830cd-0671-47ad-b0bd-b1ea455194a3 "
    'model="qwen3-coder-30b-a3b-instruct-fp8" '
    'input_tokens=15473 output_tokens=49 ttft_ms="1203.00" avg_itl_ms="30.30"'
)


def test_parse_dynamo_log_extracts_fields(mod, tmp_path):
    log = tmp_path / "frontend.log"
    log.write_text(SAMPLE_DYNAMO_LINE + "\n")

    entries = mod.parse_dynamo_log(log, model_filter=None)
    assert len(entries) == 1
    e = entries[0]
    assert e.line_no == 1
    assert e.input_tokens == 15473
    assert e.output_tokens == 49
    assert e.elapsed_ms == 1628
    assert e.ttft_ms == pytest.approx(1203.0)
    assert e.avg_itl_ms == pytest.approx(30.3)
    assert e.model == "qwen3-coder-30b-a3b-instruct-fp8"
    assert e.status == "success"
    assert e.elapsed_s == pytest.approx(1.628)
    # ITL is the inter-token gap; 49 tokens → 48 gaps.
    assert e.expected_decode_s == pytest.approx(48 * 30.3 / 1000.0)


def test_expected_decode_s_uses_n_minus_one_gaps(mod):
    """ITL counts *gaps* between tokens. 1 or 0 tokens → 0s (no gap).
    First-token latency belongs to ttft_ms, not decode."""
    one = mod.DynamoEntry(line_no=1, input_tokens=100, output_tokens=1,
                          elapsed_ms=500, ttft_ms=500.0, avg_itl_ms=30.0,
                          model="m", status="success")
    assert one.expected_decode_s == 0.0

    zero = mod.DynamoEntry(line_no=2, input_tokens=100, output_tokens=0,
                           elapsed_ms=500, ttft_ms=500.0, avg_itl_ms=30.0,
                           model="m", status="success")
    assert zero.expected_decode_s == 0.0

    many = mod.DynamoEntry(line_no=3, input_tokens=100, output_tokens=10,
                           elapsed_ms=900, ttft_ms=600.0, avg_itl_ms=30.0,
                           model="m", status="success")
    # 10 tokens → 9 gaps × 30ms = 270ms
    assert many.expected_decode_s == pytest.approx(9 * 30.0 / 1000.0)


def test_parse_dynamo_log_filters_by_model(mod, tmp_path):
    other = SAMPLE_DYNAMO_LINE.replace(
        'model="qwen3-coder-30b-a3b-instruct-fp8"', 'model="some-other-model"'
    )
    log = tmp_path / "frontend.log"
    log.write_text(SAMPLE_DYNAMO_LINE + "\n" + other + "\n")

    entries = mod.parse_dynamo_log(log, model_filter="qwen3-coder-30b-a3b-instruct-fp8")
    assert len(entries) == 1
    assert entries[0].model == "qwen3-coder-30b-a3b-instruct-fp8"


def test_parse_dynamo_log_skips_unrelated_lines(mod, tmp_path):
    log = tmp_path / "frontend.log"
    log.write_text(
        "INFO startup: dynamo_llm starting\n"
        + SAMPLE_DYNAMO_LINE + "\n"
        + "DEBUG http-request: forwarded chunk\n"
    )
    entries = mod.parse_dynamo_log(log, model_filter=None)
    assert len(entries) == 1


def test_parse_profile_picks_llm_end_only(mod, tmp_path):
    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text("\n".join([
        json.dumps({"ev": "query.start", "ts": 1.0}),
        json.dumps({"ev": "turn.start", "step": 1, "ts": 2.0}),
        json.dumps({"ev": "llm.start", "step": 1, "ts": 3.0}),
        json.dumps({
            "ev": "llm.end", "step": 1, "ts": 5.0,
            "duration_s": 1.3, "step_duration_s": 2.0,
            "tokens": {"prompt_tokens": 100, "completion_tokens": 49},
            "finish": "tool-calls",
        }),
        json.dumps({"ev": "tool.start", "step": 1, "ts": 4.0}),
        json.dumps({
            "ev": "llm.end", "step": 2, "ts": 10.0,
            "duration_s": 0.5, "step_duration_s": 0.8,
            "tokens": {"prompt_tokens": 200, "completion_tokens": 30},
            "finish": "stop",
        }),
        "",
        "{not valid json}",
    ]))
    steps = mod.parse_profile(ndjson)
    assert [s.step for s in steps] == [1, 2]
    assert steps[0].prompt_tokens == 100
    assert steps[0].completion_tokens == 49
    assert steps[0].step_duration_s == pytest.approx(2.0)
    assert steps[1].completion_tokens == 30


def test_parse_profile_sorts_by_ts(mod, tmp_path):
    """ts ordering wins even if events are emitted out of order in the file."""
    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text("\n".join([
        json.dumps({"ev": "llm.end", "step": 2, "ts": 10.0,
                    "tokens": {"completion_tokens": 30}}),
        json.dumps({"ev": "llm.end", "step": 1, "ts": 5.0,
                    "tokens": {"completion_tokens": 49}}),
    ]))
    steps = mod.parse_profile(ndjson)
    assert [s.step for s in steps] == [1, 2]


def test_match_in_order_pairs_by_index(mod):
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=10, completion_tokens=49,
                        duration_s=1.0, step_duration_s=2.0, finish=None),
        mod.ProfileStep(step=2, ts=2.0, prompt_tokens=20, completion_tokens=30,
                        duration_s=0.5, step_duration_s=0.8, finish=None),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        elapsed_ms=1500, ttft_ms=200.0, avg_itl_ms=20.0,
                        model="m", status="success"),
        mod.DynamoEntry(line_no=2, input_tokens=20, output_tokens=30,
                        elapsed_ms=800, ttft_ms=150.0, avg_itl_ms=15.0,
                        model="m", status="success"),
    ]
    pairs = mod.match_in_order(profile, dynamo)
    assert len(pairs) == 2
    assert pairs[0][0].step == 1 and pairs[0][1].output_tokens == 49
    assert pairs[1][0].step == 2 and pairs[1][1].output_tokens == 30


def test_match_in_order_marks_excess_profile_unmatched(mod):
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=10, completion_tokens=49,
                        duration_s=None, step_duration_s=None, finish=None),
        mod.ProfileStep(step=2, ts=2.0, prompt_tokens=20, completion_tokens=30,
                        duration_s=None, step_duration_s=None, finish=None),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        elapsed_ms=1500, ttft_ms=200.0, avg_itl_ms=20.0,
                        model="m", status="success"),
    ]
    pairs = mod.match_in_order(profile, dynamo)
    assert pairs[1][1] is None


def test_match_in_order_excess_dynamo_ignored(mod):
    """Order-based pairing drops dynamo entries past the profile end -- the
    extra requests aren't in this session's NDJSON (other client, leftover
    log lines). Documented behavior; render_table doesn't surface them."""
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=10, completion_tokens=49,
                        duration_s=None, step_duration_s=None, finish=None),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        elapsed_ms=1500, ttft_ms=200.0, avg_itl_ms=20.0,
                        model="m", status="success"),
        mod.DynamoEntry(line_no=2, input_tokens=20, output_tokens=30,
                        elapsed_ms=800, ttft_ms=150.0, avg_itl_ms=15.0,
                        model="m", status="success"),
    ]
    pairs = mod.match_in_order(profile, dynamo)
    assert len(pairs) == 1
    assert pairs[0][1].line_no == 1


def test_render_table_flags_token_mismatch(mod):
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=99, completion_tokens=49,
                        duration_s=1.0, step_duration_s=2.0, finish=None),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        elapsed_ms=1500, ttft_ms=200.0, avg_itl_ms=20.0,
                        model="m", status="success"),
    ]
    table, mismatches = mod.render_table(mod.match_in_order(profile, dynamo))
    assert "PROMPT_DIFF(99!=10)" in table
    assert mismatches >= 1


def test_render_table_status_not_counted_as_mismatch(mod):
    """status != "success" surfaces as a flag string but does NOT bump the
    mismatch count (token + duration agreement is the strict check; status
    is purely informational). Pinning this so a future change is deliberate."""
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=10, completion_tokens=49,
                        duration_s=1.0, step_duration_s=1.7, finish=None),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        elapsed_ms=1500, ttft_ms=200.0, avg_itl_ms=20.0,
                        model="m", status="error"),
    ]
    table, mismatches = mod.render_table(mod.match_in_order(profile, dynamo))
    assert "STATUS=error" in table
    assert mismatches == 0


def test_parse_profile_tokens_null_and_camelcase(mod, tmp_path):
    """tokens may be null/missing (rare degenerate case) and the AI SDK
    sometimes surfaces camelCase keys (promptTokens / completionTokens)
    instead of snake_case. parse_profile must handle both."""
    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text("\n".join([
        json.dumps({"ev": "llm.end", "step": 1, "ts": 1.0,
                    "tokens": None,
                    "duration_s": 1.0, "step_duration_s": 1.5}),
        json.dumps({"ev": "llm.end", "step": 2, "ts": 2.0,
                    "duration_s": 1.0, "step_duration_s": 1.5}),
        json.dumps({"ev": "llm.end", "step": 3, "ts": 3.0,
                    "tokens": {"promptTokens": 77, "completionTokens": 42},
                    "duration_s": 1.0, "step_duration_s": 1.5}),
    ]))
    steps = mod.parse_profile(ndjson)
    assert [s.step for s in steps] == [1, 2, 3]
    assert steps[0].prompt_tokens is None
    assert steps[0].completion_tokens is None
    assert steps[1].prompt_tokens is None
    assert steps[1].completion_tokens is None
    assert steps[2].prompt_tokens == 77
    assert steps[2].completion_tokens == 42


def test_render_table_no_flags_on_clean_match(mod):
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=10, completion_tokens=49,
                        duration_s=1.0, step_duration_s=1.7, finish=None),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        elapsed_ms=1500, ttft_ms=200.0, avg_itl_ms=20.0,
                        model="m", status="success"),
    ]
    table, mismatches = mod.render_table(mod.match_in_order(profile, dynamo))
    assert mismatches == 0
    # framework_s = 1.7 - 1.5 = 0.2 (positive, normal)
    assert "NEG_FRAMEWORK" not in table


def test_main_end_to_end_clean(mod, tmp_path, capsys):
    log = tmp_path / "frontend.log"
    log.write_text(SAMPLE_DYNAMO_LINE + "\n")

    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text(json.dumps({
        "ev": "llm.end", "step": 1, "ts": 1.0,
        "duration_s": 1.2, "step_duration_s": 1.7,
        "tokens": {"prompt_tokens": 15473, "completion_tokens": 49},
        "finish": "tool-calls",
    }) + "\n")

    rc = mod.main(["--profile", str(ndjson), "--dynamo-log", str(log),
                   "--model", "qwen3-coder-30b-a3b-instruct-fp8", "--strict"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "profile llm.end events:   1" in captured.out
    assert "dynamo request completed: 1" in captured.out
    assert "flagged rows: 0" in captured.out


def test_main_strict_exits_nonzero_on_mismatch(mod, tmp_path, capsys):
    log = tmp_path / "frontend.log"
    log.write_text(SAMPLE_DYNAMO_LINE + "\n")

    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text(json.dumps({
        "ev": "llm.end", "step": 1, "ts": 1.0,
        "duration_s": 1.2, "step_duration_s": 1.7,
        "tokens": {"prompt_tokens": 99999, "completion_tokens": 49},  # prompt mismatch
        "finish": "tool-calls",
    }) + "\n")

    rc = mod.main(["--profile", str(ndjson), "--dynamo-log", str(log), "--strict"])
    assert rc == 1
