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
    # Wall-clock decode portion of the request, directly measured.
    assert e.post_ttft_s == pytest.approx((1628 - 1203.0) / 1000.0)


def test_post_ttft_s_directly_measured(mod):
    """post_ttft_s = max(0, elapsed - ttft). Doesn't rely on avg_itl_ms,
    which is unreliable when the first SSE chunk carries many tokens
    (denominator excludes them, inflating per-token average)."""
    e = mod.DynamoEntry(line_no=1, input_tokens=100, output_tokens=10,
                        elapsed_ms=1500, ttft_ms=500.0, avg_itl_ms=999.0,
                        model="m", status="success")
    assert e.post_ttft_s == pytest.approx(1.0)

    # Defensive: ttft > elapsed (shouldn't happen but don't go negative).
    weird = mod.DynamoEntry(line_no=2, input_tokens=100, output_tokens=1,
                            elapsed_ms=500, ttft_ms=700.0, avg_itl_ms=None,
                            model="m", status="success")
    assert weird.post_ttft_s == 0.0


def test_post_ttft_s_none_when_ttft_missing(mod):
    e = mod.DynamoEntry(line_no=1, input_tokens=100, output_tokens=1,
                        elapsed_ms=500, ttft_ms=None, avg_itl_ms=None,
                        model="m", status="success")
    assert e.post_ttft_s is None


def test_parse_profile_ai_sdk_v6_token_shape(mod, tmp_path):
    """opencode session.ts:getUsage emits {input, output, reasoning,
    cache:{read,write}, total}, where `input` already has cache subtracted.
    To compare with dynamo ISL we must add cache.read back."""
    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text("\n".join([
        json.dumps({
            "ev": "llm.end", "step": 1, "ts": 1.0,
            "duration_s": 1.0, "step_duration_s": 1.5,
            "tokens": {
                "total": 1100, "input": 200, "output": 49,
                "reasoning": 0, "cache": {"read": 800, "write": 100},
            },
        }),
        json.dumps({
            "ev": "llm.end", "step": 2, "ts": 2.0,
            "duration_s": 1.0, "step_duration_s": 1.5,
            "tokens": {
                "total": 50, "input": 30, "output": 20,
                "cache": {"read": 0, "write": 0},
            },
        }),
    ]))
    steps = mod.parse_profile(ndjson)
    assert len(steps) == 2
    # input(200) + cache.read(800) = 1000  -- matches dynamo ISL
    assert steps[0].prompt_tokens == 1000
    assert steps[0].completion_tokens == 49
    # no cache hits
    assert steps[1].prompt_tokens == 30
    assert steps[1].completion_tokens == 20


def test_parse_profile_classic_openai_shape_still_works(mod, tmp_path):
    """Older NDJSON files (prior to the v5 normalization) carry the
    classic OpenAI usage shape directly. Backward compatibility."""
    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text(json.dumps({
        "ev": "llm.end", "step": 1, "ts": 1.0,
        "tokens": {"prompt_tokens": 500, "completion_tokens": 30},
    }) + "\n")
    steps = mod.parse_profile(ndjson)
    assert steps[0].prompt_tokens == 500
    assert steps[0].completion_tokens == 30


def test_parse_dynamo_log_strips_ansi_color_codes(mod, tmp_path):
    """tracing-subscriber's pretty formatter wraps both keys and `=`
    separators in ANSI SGR codes even when stdout is redirected to a
    file. Parser must strip them per line before applying field regexes.
    Pasted from a live dynamo log opened in VSCode."""
    ansi_line = (
        "\x1b[2m2026-05-19T11:48:13.237919Z\x1b[0m \x1b[32m INFO\x1b[0m "
        "\x1b[1mhttp-request\x1b[0m: \x1b[2mdynamo_llm::http::service::metrics\x1b[0m\x1b[2m:\x1b[0m "
        "request completed \x1b[3mrequest_id\x1b[0m\x1b[2m=\x1b[0m47b34438 "
        "\x1b[3mmodel\x1b[0m\x1b[2m=\x1b[0mqwen3-coder-30b-a3b-instruct-fp8 "
        "\x1b[3mendpoint\x1b[0m\x1b[2m=\x1b[0mchat_completions "
        "\x1b[3mstatus\x1b[0m\x1b[2m=\x1b[0msuccess "
        "\x1b[3melapsed_ms\x1b[0m\x1b[2m=\x1b[0m1205 "
        "\x1b[3minput_tokens\x1b[0m\x1b[2m=\x1b[0m721 "
        "\x1b[3moutput_tokens\x1b[0m\x1b[2m=\x1b[0m7 "
        '\x1b[3mttft_ms\x1b[0m\x1b[2m=\x1b[0m"1162.22" '
        '\x1b[3mavg_itl_ms\x1b[0m\x1b[2m=\x1b[0m"6.94"\x1b[0m'
    )
    log = tmp_path / "frontend.log"
    log.write_text(ansi_line + "\n")
    entries = mod.parse_dynamo_log(log, model_filter=None)
    assert len(entries) == 1
    e = entries[0]
    assert e.elapsed_ms == 1205
    assert e.input_tokens == 721
    assert e.output_tokens == 7
    assert e.ttft_ms == pytest.approx(1162.22)
    assert e.avg_itl_ms == pytest.approx(6.94)
    assert e.model == "qwen3-coder-30b-a3b-instruct-fp8"
    assert e.status == "success"


def test_parse_dynamo_log_accepts_json_format(mod, tmp_path):
    """tracing-subscriber can also emit JSON when configured for log
    aggregators. Field regex uses `[=:]` so both pretty `k=v` and JSON
    `"k":v` match."""
    json_line = (
        '{"timestamp":"2026-05-19T11:48:13Z","level":"INFO","target":'
        '"dynamo_llm::http::service::metrics","fields":{"message":'
        '"request completed","elapsed_ms":1205,"model":'
        '"qwen3-coder-30b-a3b-instruct-fp8","status":"success",'
        '"input_tokens":721,"output_tokens":7,"ttft_ms":"1162.22",'
        '"avg_itl_ms":"6.94"}}'
    )
    log = tmp_path / "frontend.log"
    log.write_text(json_line + "\n")
    entries = mod.parse_dynamo_log(log, model_filter=None)
    assert len(entries) == 1
    e = entries[0]
    assert e.elapsed_ms == 1205
    assert e.input_tokens == 721
    assert e.output_tokens == 7
    assert e.model == "qwen3-coder-30b-a3b-instruct-fp8"


def test_parse_dynamo_log_handles_missing_optional_fields(mod, tmp_path):
    """Real-world: short responses (output_tokens<2) omit avg_itl_ms;
    error/cancel paths may omit ttft_ms or even token counts entirely.
    Previously these lines were silently dropped by a strict 4-field
    regex; now each field is optional and the line is returned with
    None for the missing ones."""
    # output_tokens=1 → no avg_itl_ms recorded
    short = (
        'INFO http-request: dynamo_llm::http::service::metrics: request completed '
        "request_id=a model=m endpoint=chat_completions request_type=stream "
        "status=success elapsed_ms=200 method=POST uri=/ version=HTTP/1.1 "
        'request_id=b model="m" input_tokens=10 output_tokens=1 ttft_ms="200.00"'
    )
    # error path → no token / timing fields at all (only InflightGuard core)
    err = (
        'ERROR http-request: dynamo_llm::http::service::metrics: request completed '
        "request_id=c model=m endpoint=chat_completions request_type=stream "
        "status=error error_type=cancelled "
        'error_detail="cancelled before completion" elapsed_ms=42 method=POST'
    )
    log = tmp_path / "frontend.log"
    log.write_text(short + "\n" + err + "\n")

    stats = mod.DynamoParseStats()
    entries = mod.parse_dynamo_log(log, model_filter=None, stats=stats)
    assert len(entries) == 2
    assert stats.total_completed == 2
    assert stats.matched == 2

    short_e = entries[0]
    assert short_e.input_tokens == 10
    assert short_e.output_tokens == 1
    assert short_e.ttft_ms == pytest.approx(200.0)
    assert short_e.avg_itl_ms is None
    assert short_e.expected_decode_s is None
    assert stats.missing_avg_itl_ms == 1

    err_e = entries[1]
    assert err_e.input_tokens is None
    assert err_e.output_tokens is None
    assert err_e.ttft_ms is None
    assert err_e.avg_itl_ms is None
    assert err_e.elapsed_ms == 42
    assert err_e.status == "error"
    assert stats.missing_input_tokens == 1
    assert stats.missing_output_tokens == 1
    assert stats.missing_ttft_ms == 2  # both lines


def test_parse_dynamo_log_skips_no_elapsed(mod, tmp_path):
    """Defensive: a `request completed` without elapsed_ms shouldn't
    happen per InflightGuard::Drop, but if it does we don't trust the
    row -- drop it and bump the stat."""
    log = tmp_path / "frontend.log"
    log.write_text(
        "INFO http-request: dynamo_llm::http::service::metrics: request completed "
        "request_id=x model=m\n"
    )
    stats = mod.DynamoParseStats()
    entries = mod.parse_dynamo_log(log, model_filter=None, stats=stats)
    assert entries == []
    assert stats.total_completed == 1
    assert stats.missing_elapsed_ms == 1
    assert stats.matched == 0


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


def test_main_skip_dynamo_leading(mod, tmp_path, capsys):
    """When opencode probes the model at startup, the dynamo log has a
    leading `request completed` line that has no profile counterpart.
    --skip-dynamo-leading=1 should drop the head and align the rest."""
    probe = (
        'INFO http-request: dynamo_llm::http::service::metrics: request completed '
        "request_id=a model=m endpoint=chat_completions request_type=stream "
        "status=success elapsed_ms=100 method=POST uri=/ version=HTTP/1.1 "
        'request_id=b model="m" input_tokens=1 output_tokens=1 ttft_ms="100.00"'
    )
    real = (
        'INFO http-request: dynamo_llm::http::service::metrics: request completed '
        "request_id=c model=m endpoint=chat_completions request_type=stream "
        "status=success elapsed_ms=1500 method=POST uri=/ version=HTTP/1.1 "
        'request_id=d model="m" input_tokens=200 output_tokens=49 '
        'ttft_ms="500.00" avg_itl_ms="20.00"'
    )
    log = tmp_path / "frontend.log"
    log.write_text(probe + "\n" + real + "\n")

    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text(json.dumps({
        "ev": "llm.end", "step": 1, "ts": 1.0,
        "duration_s": 1.2, "step_duration_s": 1.5,
        "tokens": {"prompt_tokens": 200, "completion_tokens": 49},
        "finish": "tool-calls",
    }) + "\n")

    rc = mod.main([
        "--profile", str(ndjson), "--dynamo-log", str(log),
        "--skip-dynamo-leading", "1", "--strict",
    ])
    captured = capsys.readouterr()
    assert rc == 0, captured.out + captured.err
    assert "dropped 1 leading entries" in captured.out
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
