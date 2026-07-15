#!/usr/bin/env python3
"""OpenAI-compatible mock LLM server for CPU-contention experiments.

Replaces the Dynamo frontend so very high --max-in-flight runner sweeps
land their concurrency on the CPU side (opencode tool execution /
scaffold), instead of being throttled by GPU KV-cache pressure and
inflated TTFT. Point opencode at it by setting testbed.yaml
`.dynamo.host/.dynamo.port` (or TESTBED__DYNAMO__HOST/PORT) to this
server before `testbed.sh up opencode` — the rendered opencode.json's
{{DYNAMO_BASE_URL}} then targets the mock; nothing else changes.

Behavior (template responder, stateless):
  * Turn depth = number of `role:"assistant"` entries in the request
    `messages` (each request self-describes its position in the loop).
  * depth < --tool-turns  -> stream ONE tool call. The tool is chosen
    from the request's own `tools` array (prefer `bash`, else the first
    listed) so opencode never sees an unknown tool name. Arguments come
    from --tool-cmd (bash) or a generic {} for other tools.
  * depth >= --tool-turns -> stream --output-tokens words of text and
    finish with "stop", ending the agent loop.

Synthetic latency: --ttft-ms before the first content chunk, --itl-ms
between subsequent chunks. Defaults 0 = as fast as possible (that is
the point: LLM time ~0 so CPU-side effects dominate).

Wire format: standard OpenAI chat.completion.chunk SSE (role delta ->
tool_calls/name+arguments deltas or text deltas -> finish chunk ->
usage chunk -> [DONE]).  A dynamo-shaped `nvext.timing` block rides on
the finish chunk so the opencode profile patch's llm.end.dynamo fields
stay populated. `stream:false` requests (e.g. title/summary agents) get
a plain JSON completion. GET /v1/models and /health also served.

Observability: NDJSON request log (--log, default logs/mock_llm.ndjson)
and a PID file (--pid-file, default logs/mock_llm.pid) so
monitor_resources.py picks the server up via its *.pid scan.

Usage:
  python scripts/mock_llm_server.py --port 8000 \\
      [--host 127.0.0.1] [--tool-turns 4] [--tool-cmd 'ls'] \\
      [--ttft-ms 0] [--itl-ms 0] [--output-tokens 32] \\
      [--model local] [--log logs/mock_llm.ndjson] [--pid-file logs/mock_llm.pid]

Stdlib only (asyncio + minimal HTTP/1.1 handling); no aiohttp.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path


# ---------- response engine (pure: request dict -> plan) ----------


@dataclass
class Chunk:
    payload: dict            # chat.completion.chunk body
    delay_ms: float = 0.0    # sleep BEFORE writing this chunk


@dataclass
class ResponsePlan:
    chunks: list[Chunk] = field(default_factory=list)
    mode: str = "final"          # "tool" | "final"
    turn_index: int = 0
    completion_tokens: int = 0
    request_id: str = ""


def count_assistant_turns(messages: list[dict]) -> int:
    return sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "assistant")


def pick_tool(tools: list[dict] | None, preferred: str = "bash") -> dict | None:
    """Pick a tool from the request's own tools array; never invent names."""
    if not tools:
        return None
    named = {}
    for t in tools:
        fn = (t or {}).get("function") or {}
        name = fn.get("name")
        if name:
            named[name] = fn
    if preferred in named:
        return named[preferred]
    first = tools[0].get("function") or {}
    return first if first.get("name") else None


def tool_arguments(tool_name: str, tool_cmd: str) -> str:
    if tool_name == "bash":
        return json.dumps({"command": tool_cmd, "description": "mock workload command"})
    # generic minimal args; opencode tools reject bad schemas loudly, which
    # is fine for an experiment harness — prefer bash-capable agents.
    return json.dumps({})


def _estimate_prompt_tokens(body: dict) -> int:
    n = 0
    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    n += len(part["text"])
    return max(1, n // 4)


def build_plan(
    body: dict,
    *,
    model: str,
    tool_turns: int,
    tool_cmd: str,
    ttft_ms: float,
    itl_ms: float,
    output_tokens: int,
    now_unix_ms: float,
) -> ResponsePlan:
    rid = uuid.uuid4().hex
    chat_id = f"chatcmpl-{rid}"
    created = int(now_unix_ms / 1000)
    messages = body.get("messages") or []
    depth = count_assistant_turns(messages)
    prompt_tokens = _estimate_prompt_tokens(body)

    def chunk(delta: dict, finish_reason=None, extra: dict | None = None,
              delay_ms: float = 0.0) -> Chunk:
        payload = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if extra:
            payload.update(extra)
        return Chunk(payload=payload, delay_ms=delay_ms)

    plan = ResponsePlan(turn_index=depth, request_id=rid)
    tool = pick_tool(body.get("tools")) if depth < tool_turns else None
    plan.mode = "tool" if tool else "final"

    chunks: list[Chunk] = [chunk({"role": "assistant"}, delay_ms=ttft_ms)]

    if tool:
        call_id = f"call_{uuid.uuid4().hex[:24]}"
        name = tool["name"]
        args = tool_arguments(name, tool_cmd)
        # name first, then arguments split across >=2 deltas (mirrors real
        # OpenAI framing the @ai-sdk/openai-compatible parser expects).
        chunks.append(chunk({"tool_calls": [{
            "index": 0, "id": call_id, "type": "function",
            "function": {"name": name, "arguments": ""},
        }]}, delay_ms=itl_ms))
        half = max(1, len(args) // 2)
        for frag in (args[:half], args[half:]):
            if frag:
                chunks.append(chunk({"tool_calls": [{
                    "index": 0, "function": {"arguments": frag},
                }]}, delay_ms=itl_ms))
        finish_reason = "tool_calls"
        plan.completion_tokens = max(1, len(args) // 4)
    else:
        n = max(1, output_tokens)
        for i in range(n):
            chunks.append(chunk({"content": ("done " if i else "Task complete: ")},
                                delay_ms=itl_ms))
        finish_reason = "stop"
        plan.completion_tokens = n

    # finish chunk carries a dynamo-shaped nvext.timing block so profile
    # llm.end.dynamo stays populated against the mock.
    total_ms = ttft_ms + itl_ms * max(0, len(chunks) - 1)
    chunks.append(chunk({}, finish_reason=finish_reason, extra={
        "nvext": {"timing": {
            "request_received_ms": now_unix_ms,
            "total_time_ms": total_ms,
        }},
    }, delay_ms=itl_ms))
    # usage chunk (opencode requests stream_options.include_usage; emit always)
    chunks.append(Chunk(payload={
        "id": chat_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": plan.completion_tokens,
            "total_tokens": prompt_tokens + plan.completion_tokens,
        },
    }))
    plan.chunks = chunks
    return plan


def plan_to_non_stream(plan: ResponsePlan, model: str) -> dict:
    """Collapse a streaming plan into a single chat.completion response."""
    text_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish = "stop"
    usage = None
    chat_id = None
    created = None
    for ch in plan.chunks:
        p = ch.payload
        chat_id = chat_id or p.get("id")
        created = created or p.get("created")
        if p.get("usage"):
            usage = p["usage"]
        for choice in p.get("choices") or []:
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(tc.get("index", 0), {
                    "id": None, "type": "function",
                    "function": {"name": None, "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
    message: dict = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
    return {
        "id": chat_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage,
    }


# ---------- HTTP layer (stdlib asyncio) ----------


class MockServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.log_path: Path | None = Path(args.log) if args.log else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _log(self, rec: dict) -> None:
        if not self.log_path:
            return
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._handle(reader, writer)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        while True:
            request_line = await reader.readline()
            if not request_line:
                return
            try:
                method, path, _ = request_line.decode("latin-1").split(" ", 2)
            except ValueError:
                return
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, _, v = line.decode("latin-1").partition(":")
                headers[k.strip().lower()] = v.strip()
            length = int(headers.get("content-length", "0") or "0")
            body_bytes = await reader.readexactly(length) if length else b""

            keep_alive = headers.get("connection", "keep-alive").lower() != "close"

            if method == "GET" and path.startswith("/health"):
                await self._respond_json(writer, {"status": "ok"}, keep_alive)
            elif method == "GET" and path.startswith("/v1/models"):
                await self._respond_json(writer, {
                    "object": "list",
                    "data": [{"id": self.args.model, "object": "model",
                              "created": 0, "owned_by": "mock"}],
                }, keep_alive)
            elif method == "POST" and path.startswith("/v1/chat/completions"):
                try:
                    body = json.loads(body_bytes.decode("utf-8") or "{}")
                except json.JSONDecodeError:
                    await self._respond_json(
                        writer, {"error": {"message": "invalid JSON"}}, keep_alive, status=400)
                    continue
                await self._chat(writer, body, keep_alive)
            else:
                await self._respond_json(
                    writer, {"error": {"message": f"no route {method} {path}"}},
                    keep_alive, status=404)
            if not keep_alive:
                return

    async def _respond_json(self, writer, obj: dict, keep_alive: bool, status: int = 200) -> None:
        data = json.dumps(obj).encode("utf-8")
        reason = {200: "OK", 400: "Bad Request", 404: "Not Found"}.get(status, "OK")
        head = (
            f"HTTP/1.1 {status} {reason}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(data)}\r\n"
            f"Connection: {'keep-alive' if keep_alive else 'close'}\r\n\r\n"
        )
        writer.write(head.encode("latin-1") + data)
        await writer.drain()

    async def _chat(self, writer, body: dict, keep_alive: bool) -> None:
        a = self.args
        t0 = time.time()
        plan = build_plan(
            body,
            model=a.model,
            tool_turns=a.tool_turns,
            tool_cmd=a.tool_cmd,
            ttft_ms=a.ttft_ms,
            itl_ms=a.itl_ms,
            output_tokens=a.output_tokens,
            now_unix_ms=t0 * 1000.0,
        )
        stream = bool(body.get("stream", True))
        if stream:
            head = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Transfer-Encoding: chunked\r\n"
                f"Connection: {'keep-alive' if keep_alive else 'close'}\r\n\r\n"
            )
            writer.write(head.encode("latin-1"))
            for ch in plan.chunks:
                if ch.delay_ms > 0:
                    await asyncio.sleep(ch.delay_ms / 1000.0)
                self._write_sse(writer, "data: " + json.dumps(ch.payload) + "\n\n")
                await writer.drain()
            self._write_sse(writer, "data: [DONE]\n\n")
            writer.write(b"0\r\n\r\n")  # end chunked body
            await writer.drain()
        else:
            total_delay = plan.chunks[0].delay_ms if plan.chunks else 0.0
            if total_delay > 0:
                await asyncio.sleep(total_delay / 1000.0)
            await self._respond_json(writer, plan_to_non_stream(plan, a.model), keep_alive)
        self._log({
            "ts": t0,
            "path": "/v1/chat/completions",
            "stream": stream,
            "n_messages": len(body.get("messages") or []),
            "isl_chars": sum(
                len(m.get("content")) for m in body.get("messages") or []
                if isinstance(m.get("content"), str)),
            "turn_index": plan.turn_index,
            "mode": plan.mode,
            "request_id": plan.request_id,
            "injected_ttft_ms": a.ttft_ms,
            "injected_itl_ms": a.itl_ms,
            "completion_tokens": plan.completion_tokens,
            "wall_s": round(time.time() - t0, 4),
        })

    @staticmethod
    def _write_sse(writer, text: str) -> None:
        data = text.encode("utf-8")
        writer.write(f"{len(data):x}\r\n".encode("latin-1") + data + b"\r\n")


async def serve(args: argparse.Namespace) -> None:
    srv = MockServer(args)
    server = await asyncio.start_server(srv.handle, args.host, args.port)
    if args.pid_file:
        p = Path(args.pid_file)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    addr = ", ".join(str(s.getsockname()) for s in server.sockets)
    print(f"mock_llm_server listening on {addr} "
          f"(tool_turns={args.tool_turns} ttft={args.ttft_ms}ms itl={args.itl_ms}ms)",
          flush=True)
    async with server:
        await server.serve_forever()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model", default="local",
                    help="model id echoed in responses (match model.served_name)")
    ap.add_argument("--tool-turns", type=int, default=4,
                    help="number of tool-calling turns before the final text turn")
    ap.add_argument("--tool-cmd", default="ls",
                    help="bash command the mock asks the agent to run each tool turn")
    ap.add_argument("--ttft-ms", type=float, default=0.0)
    ap.add_argument("--itl-ms", type=float, default=0.0)
    ap.add_argument("--output-tokens", type=int, default=32,
                    help="words emitted on the final text turn")
    ap.add_argument("--log", default="logs/mock_llm.ndjson")
    ap.add_argument("--pid-file", default="logs/mock_llm.pid")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass
    finally:
        if args.pid_file and Path(args.pid_file).exists():
            try:
                Path(args.pid_file).unlink()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
