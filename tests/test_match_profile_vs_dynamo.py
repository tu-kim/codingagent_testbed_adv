"""Tests for scripts/match_profile_vs_dynamo.py.

Lives in tests/ even though the script is in scripts/ (not part of the
installed `testbed` package) -- imported via importlib to keep the
script self-contained and runnable directly.

Token-count cross-check only -- all timing/duration comparisons were
removed because dynamo's elapsed_ms and the AI SDK's step_duration_s
measure intervals with different start origins (HTTP request received
vs first-chunk receipt at start-step), so direct delta is unphysical.
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


def test_parse_dynamo_log_extracts_token_fields(mod, tmp_path):
    log = tmp_path / "frontend.log"
    log.write_text(SAMPLE_DYNAMO_LINE + "\n")

    entries = mod.parse_dynamo_log(log, model_filter=None)
    assert len(entries) == 1
    e = entries[0]
    assert e.line_no == 1
    assert e.input_tokens == 15473
    assert e.output_tokens == 49
    assert e.model == "qwen3-coder-30b-a3b-instruct-fp8"
    assert e.status == "success"


def test_parse_dynamo_log_filters_by_model(mod, tmp_path):
    other = SAMPLE_DYNAMO_LINE.replace(
        'model="qwen3-coder-30b-a3b-instruct-fp8"', 'model="some-other-model"'
    ).replace(
        "model=qwen3-coder-30b-a3b-instruct-fp8 ", "model=some-other-model "
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
        "\x1b[3moutput_tokens\x1b[0m\x1b[2m=\x1b[0m7\x1b[0m"
    )
    log = tmp_path / "frontend.log"
    log.write_text(ansi_line + "\n")
    entries = mod.parse_dynamo_log(log, model_filter=None)
    assert len(entries) == 1
    e = entries[0]
    assert e.input_tokens == 721
    assert e.output_tokens == 7
    assert e.model == "qwen3-coder-30b-a3b-instruct-fp8"
    assert e.status == "success"


def test_parse_dynamo_log_accepts_json_format(mod, tmp_path):
    """tracing-subscriber can also emit JSON when configured for log
    aggregators. Field regex uses `[=:]` so both pretty `k=v` and JSON
    `"k":v` match."""
    json_line = (
        '{"timestamp":"2026-05-19T11:48:13Z","level":"INFO","target":'
        '"dynamo_llm::http::service::metrics","fields":{"message":'
        '"request completed","model":'
        '"qwen3-coder-30b-a3b-instruct-fp8","status":"success",'
        '"input_tokens":721,"output_tokens":7}}'
    )
    log = tmp_path / "frontend.log"
    log.write_text(json_line + "\n")
    entries = mod.parse_dynamo_log(log, model_filter=None)
    assert len(entries) == 1
    e = entries[0]
    assert e.input_tokens == 721
    assert e.output_tokens == 7
    assert e.model == "qwen3-coder-30b-a3b-instruct-fp8"


def test_parse_dynamo_log_handles_missing_tokens(mod, tmp_path):
    """Error/cancel paths may omit token fields entirely. Entry is still
    returned but with None tokens and a bumped stat counter."""
    err = (
        'ERROR http-request: dynamo_llm::http::service::metrics: request completed '
        "request_id=c model=m endpoint=chat_completions request_type=stream "
        "status=error error_type=cancelled "
        'error_detail="cancelled before completion" elapsed_ms=42 method=POST'
    )
    log = tmp_path / "frontend.log"
    log.write_text(err + "\n")

    stats = mod.DynamoParseStats()
    entries = mod.parse_dynamo_log(log, model_filter=None, stats=stats)
    assert len(entries) == 1
    assert entries[0].input_tokens is None
    assert entries[0].output_tokens is None
    assert entries[0].status == "error"
    assert stats.missing_input_tokens == 1
    assert stats.missing_output_tokens == 1


def test_parse_profile_picks_llm_end_only(mod, tmp_path):
    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text("\n".join([
        json.dumps({"ev": "query.start", "ts": 1.0}),
        json.dumps({"ev": "turn.start", "step": 1, "ts": 2.0}),
        json.dumps({"ev": "llm.start", "step": 1, "ts": 3.0}),
        json.dumps({
            "ev": "llm.end", "step": 1, "ts": 5.0,
            "tokens": {"prompt_tokens": 100, "completion_tokens": 49},
        }),
        json.dumps({"ev": "tool.start", "step": 1, "ts": 4.0}),
        json.dumps({
            "ev": "llm.end", "step": 2, "ts": 10.0,
            "tokens": {"prompt_tokens": 200, "completion_tokens": 30},
        }),
        "",
        "{not valid json}",
    ]))
    steps = mod.parse_profile(ndjson)
    assert [s.step for s in steps] == [1, 2]
    assert steps[0].prompt_tokens == 100
    assert steps[0].completion_tokens == 49
    assert steps[1].prompt_tokens == 200
    assert steps[1].completion_tokens == 30


def test_parse_profile_sorts_by_ts(mod, tmp_path):
    """ts ordering wins even if events appear out of order in the file."""
    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text("\n".join([
        json.dumps({"ev": "llm.end", "step": 2, "ts": 10.0,
                    "tokens": {"completion_tokens": 30}}),
        json.dumps({"ev": "llm.end", "step": 1, "ts": 5.0,
                    "tokens": {"completion_tokens": 49}}),
    ]))
    steps = mod.parse_profile(ndjson)
    assert [s.step for s in steps] == [1, 2]


def test_parse_profile_ai_sdk_v6_token_shape(mod, tmp_path):
    """opencode session.ts:getUsage emits {input, output, reasoning,
    cache:{read,write}, total}, where `input` already has cache subtracted.
    To compare with dynamo ISL we must add cache.read back."""
    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text("\n".join([
        json.dumps({"ev": "llm.end", "step": 1, "ts": 1.0,
                    "tokens": {"total": 1100, "input": 200, "output": 49,
                               "reasoning": 0,
                               "cache": {"read": 800, "write": 100}}}),
        json.dumps({"ev": "llm.end", "step": 2, "ts": 2.0,
                    "tokens": {"total": 50, "input": 30, "output": 20,
                               "cache": {"read": 0, "write": 0}}}),
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


def test_parse_profile_tokens_null_or_missing(mod, tmp_path):
    """tokens may be null/missing (rare degenerate case)."""
    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text("\n".join([
        json.dumps({"ev": "llm.end", "step": 1, "ts": 1.0, "tokens": None}),
        json.dumps({"ev": "llm.end", "step": 2, "ts": 2.0}),  # no tokens key
    ]))
    steps = mod.parse_profile(ndjson)
    assert len(steps) == 2
    for s in steps:
        assert s.prompt_tokens is None
        assert s.completion_tokens is None


def test_match_in_order_pairs_by_index(mod):
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=10, completion_tokens=49),
        mod.ProfileStep(step=2, ts=2.0, prompt_tokens=20, completion_tokens=30),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        model="m", status="success"),
        mod.DynamoEntry(line_no=2, input_tokens=20, output_tokens=30,
                        model="m", status="success"),
    ]
    pairs = mod.match_in_order(profile, dynamo)
    assert len(pairs) == 2
    assert pairs[0][0].step == 1 and pairs[0][1].output_tokens == 49
    assert pairs[1][0].step == 2 and pairs[1][1].output_tokens == 30


def test_match_in_order_marks_excess_profile_unmatched(mod):
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=10, completion_tokens=49),
        mod.ProfileStep(step=2, ts=2.0, prompt_tokens=20, completion_tokens=30),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        model="m", status="success"),
    ]
    pairs = mod.match_in_order(profile, dynamo)
    assert pairs[1][1] is None


def test_match_in_order_excess_dynamo_ignored(mod):
    """Order-based pairing drops dynamo entries past the profile end."""
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=10, completion_tokens=49),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        model="m", status="success"),
        mod.DynamoEntry(line_no=2, input_tokens=20, output_tokens=30,
                        model="m", status="success"),
    ]
    pairs = mod.match_in_order(profile, dynamo)
    assert len(pairs) == 1
    assert pairs[0][1].line_no == 1


def test_render_table_flags_token_mismatch(mod):
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=99, completion_tokens=49),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        model="m", status="success"),
    ]
    table, mismatches = mod.render_table(mod.match_in_order(profile, dynamo))
    assert "PROMPT_DIFF(99!=10)" in table
    assert mismatches == 1


def test_render_table_status_not_counted_as_mismatch(mod):
    """status != "success" surfaces as a flag string but does NOT bump the
    mismatch count (token agreement is the strict check; status is purely
    informational)."""
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=10, completion_tokens=49),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        model="m", status="error"),
    ]
    table, mismatches = mod.render_table(mod.match_in_order(profile, dynamo))
    assert "STATUS=error" in table
    assert mismatches == 0


def test_render_table_no_flags_on_clean_match(mod):
    profile = [
        mod.ProfileStep(step=1, ts=1.0, prompt_tokens=10, completion_tokens=49),
    ]
    dynamo = [
        mod.DynamoEntry(line_no=1, input_tokens=10, output_tokens=49,
                        model="m", status="success"),
    ]
    table, mismatches = mod.render_table(mod.match_in_order(profile, dynamo))
    assert mismatches == 0


def test_main_end_to_end_clean(mod, tmp_path, capsys):
    log = tmp_path / "frontend.log"
    log.write_text(SAMPLE_DYNAMO_LINE + "\n")

    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text(json.dumps({
        "ev": "llm.end", "step": 1, "ts": 1.0,
        "tokens": {"prompt_tokens": 15473, "completion_tokens": 49},
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
        'request_id=b model="m" input_tokens=1 output_tokens=1'
    )
    real = (
        'INFO http-request: dynamo_llm::http::service::metrics: request completed '
        "request_id=c model=m endpoint=chat_completions request_type=stream "
        "status=success elapsed_ms=1500 method=POST uri=/ version=HTTP/1.1 "
        'request_id=d model="m" input_tokens=200 output_tokens=49'
    )
    log = tmp_path / "frontend.log"
    log.write_text(probe + "\n" + real + "\n")

    ndjson = tmp_path / "ses.ndjson"
    ndjson.write_text(json.dumps({
        "ev": "llm.end", "step": 1, "ts": 1.0,
        "tokens": {"prompt_tokens": 200, "completion_tokens": 49},
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
        "tokens": {"prompt_tokens": 99999, "completion_tokens": 49},  # mismatch
    }) + "\n")

    rc = mod.main(["--profile", str(ndjson), "--dynamo-log", str(log), "--strict"])
    assert rc == 1
