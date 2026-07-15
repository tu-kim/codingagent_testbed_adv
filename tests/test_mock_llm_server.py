"""Tests for scripts/mock_llm_server.py (response engine, no sockets).

The engine is a pure function (request dict -> ResponsePlan) so tests
cover: assistant-turn counting, tool selection from the request's own
tools array, OpenAI chunk framing (role delta first, tool_calls
name/arguments split, finish + nvext.timing, usage chunk), latency
plan (delays recorded, not slept), and the non-stream collapse.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "mock_llm_server.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("mock_llm_server", _SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["mock_llm_server"] = module
    spec.loader.exec_module(module)
    return module


def _tools(*names):
    return [{"type": "function", "function": {"name": n, "parameters": {}}} for n in names]


def _body(n_assistant=0, tools=None, content="hello"):
    messages = [{"role": "user", "content": content}]
    for _ in range(n_assistant):
        messages.append({"role": "assistant", "content": "prev"})
        messages.append({"role": "tool", "content": "out"})
    return {"messages": messages, "tools": tools if tools is not None else _tools("bash")}


def _plan(mod, body, **kw):
    defaults = dict(model="local", tool_turns=4, tool_cmd="ls",
                    ttft_ms=0.0, itl_ms=0.0, output_tokens=8,
                    now_unix_ms=1_000_000.0)
    defaults.update(kw)
    return mod.build_plan(body, **defaults)


# ---------- turn counting / tool pick ----------


def test_count_assistant_turns(mod):
    assert mod.count_assistant_turns(_body(0)["messages"]) == 0
    assert mod.count_assistant_turns(_body(3)["messages"]) == 3


def test_pick_tool_prefers_bash(mod):
    t = mod.pick_tool(_tools("read", "bash", "grep"))
    assert t["name"] == "bash"


def test_pick_tool_falls_back_to_first(mod):
    t = mod.pick_tool(_tools("read", "grep"))
    assert t["name"] == "read"


def test_pick_tool_none_when_empty(mod):
    assert mod.pick_tool([]) is None
    assert mod.pick_tool(None) is None


# ---------- plan structure: tool turn ----------


def test_tool_turn_chunk_order(mod):
    plan = _plan(mod, _body(0))
    assert plan.mode == "tool"
    payloads = [c.payload for c in plan.chunks]
    # 1) role delta first
    assert payloads[0]["choices"][0]["delta"] == {"role": "assistant"}
    # 2) tool_calls: first delta has id+name, later deltas stream arguments
    first_tc = payloads[1]["choices"][0]["delta"]["tool_calls"][0]
    assert first_tc["id"].startswith("call_")
    assert first_tc["function"]["name"] == "bash"
    args = "".join(
        tc["function"].get("arguments", "")
        for p in payloads[1:-2]
        for ch in p.get("choices") or []
        for tc in (ch.get("delta") or {}).get("tool_calls") or []
    )
    parsed = json.loads(args)
    assert parsed["command"] == "ls"
    # 3) finish chunk: finish_reason=tool_calls + nvext.timing
    finish = payloads[-2]
    assert finish["choices"][0]["finish_reason"] == "tool_calls"
    assert finish["nvext"]["timing"]["request_received_ms"] == 1_000_000.0
    # 4) usage chunk last, empty choices
    usage = payloads[-1]
    assert usage["choices"] == []
    assert usage["usage"]["completion_tokens"] == plan.completion_tokens
    assert usage["usage"]["total_tokens"] == (
        usage["usage"]["prompt_tokens"] + usage["usage"]["completion_tokens"])


def test_chat_id_is_chatcmpl_prefixed_and_stable(mod):
    plan = _plan(mod, _body(0))
    ids = {c.payload["id"] for c in plan.chunks}
    assert len(ids) == 1
    (cid,) = ids
    assert cid == f"chatcmpl-{plan.request_id}"


def test_tool_turn_uses_request_tool_names_only(mod):
    plan = _plan(mod, _body(0, tools=_tools("mytool")))
    tc = plan.chunks[1].payload["choices"][0]["delta"]["tool_calls"][0]
    assert tc["function"]["name"] == "mytool"


# ---------- plan structure: final turn ----------


def test_final_turn_after_tool_turns(mod):
    plan = _plan(mod, _body(4), tool_turns=4, output_tokens=5)
    assert plan.mode == "final"
    payloads = [c.payload for c in plan.chunks]
    text = "".join(
        (ch.get("delta") or {}).get("content") or ""
        for p in payloads for ch in p.get("choices") or [])
    assert text.startswith("Task complete:")
    assert payloads[-2]["choices"][0]["finish_reason"] == "stop"
    assert plan.completion_tokens == 5


def test_no_tools_in_request_forces_final(mod):
    plan = _plan(mod, _body(0, tools=[]))
    assert plan.mode == "final"


# ---------- latency plan ----------


def test_latency_delays_recorded_not_slept(mod):
    plan = _plan(mod, _body(0), ttft_ms=200.0, itl_ms=10.0)
    assert plan.chunks[0].delay_ms == 200.0          # ttft before role delta
    mids = [c.delay_ms for c in plan.chunks[1:-1]]
    assert all(d == 10.0 for d in mids)
    assert plan.chunks[-1].delay_ms == 0.0           # usage chunk immediate


def test_zero_latency_default(mod):
    plan = _plan(mod, _body(0))
    assert all(c.delay_ms == 0.0 for c in plan.chunks)


# ---------- non-stream collapse ----------


def test_non_stream_collapse_tool_call(mod):
    plan = _plan(mod, _body(0))
    resp = mod.plan_to_non_stream(plan, "local")
    assert resp["object"] == "chat.completion"
    msg = resp["choices"][0]["message"]
    (tc,) = msg["tool_calls"]
    assert tc["function"]["name"] == "bash"
    assert json.loads(tc["function"]["arguments"])["command"] == "ls"
    assert resp["choices"][0]["finish_reason"] == "tool_calls"
    assert resp["usage"]["completion_tokens"] == plan.completion_tokens


def test_non_stream_collapse_text(mod):
    plan = _plan(mod, _body(4), output_tokens=3)
    resp = mod.plan_to_non_stream(plan, "local")
    msg = resp["choices"][0]["message"]
    assert msg["content"].startswith("Task complete:")
    assert "tool_calls" not in msg
    assert resp["choices"][0]["finish_reason"] == "stop"


# ---------- prompt token estimate ----------


def test_prompt_tokens_from_string_and_parts(mod):
    body = {"messages": [
        {"role": "user", "content": "x" * 400},
        {"role": "user", "content": [{"type": "text", "text": "y" * 400}]},
    ]}
    assert mod._estimate_prompt_tokens(body) == 200


# ---------- argparse defaults ----------


def test_arg_defaults(mod):
    args = mod.build_arg_parser().parse_args([])
    assert args.port == 8000
    assert args.tool_turns == 4
    assert args.ttft_ms == 0.0
    assert args.log == "logs/mock_llm.ndjson"
    assert args.pid_file == "logs/mock_llm.pid"
