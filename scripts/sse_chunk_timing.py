#!/usr/bin/env python3
"""Print per-chunk timing for an SSE capture produced by:

  curl -N ... | while IFS= read -r line; do
      printf '%s.%06d  %s\\n' "$(date +%s)" "$(date +%N | cut -c1-6)" "$line"
    done | tee /tmp/turn.timed

Each non-blank line is `<unix_seconds>.<microseconds>  <raw line>`.
We filter to `data: {...}` and `data: [DONE]`, parse each JSON chunk,
and report the wall delta from the previous event plus a short label
describing what's in the chunk (text content / tool_call / finish_reason
/ usage / nvext timing).

Usage:
  scripts/sse_chunk_timing.py /tmp/turn.timed
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_LINE_RE = re.compile(r"^(?P<ts>\d+\.\d+)\s+data:\s*(?P<body>.*)$")


def _label(chunk: dict) -> str:
    """One-line description of what's in this SSE chunk."""
    # Usage-only chunk (last before [DONE] when stream_options.include_usage)
    if not chunk.get("choices") and chunk.get("usage"):
        u = chunk["usage"]
        return (
            f"[USAGE] prompt={u.get('prompt_tokens')} "
            f"completion={u.get('completion_tokens')} total={u.get('total_tokens')}"
        )

    choices = chunk.get("choices") or []
    if not choices:
        return "[EMPTY]"

    c0 = choices[0]
    delta = c0.get("delta") or {}
    finish = c0.get("finish_reason")
    bits: list[str] = []

    # tool_calls present?
    tc = delta.get("tool_calls")
    if tc:
        for entry in tc:
            fn = (entry.get("function") or {}).get("name")
            args = (entry.get("function") or {}).get("arguments") or ""
            args_short = args if len(args) <= 60 else args[:57] + "..."
            bits.append(f"[TOOL_CALL name={fn} args={args_short!r}]")

    # text content delta?
    content = delta.get("content")
    if content:
        # Keep print compact — escape newlines so each event is one line
        text = content if len(content) <= 40 else content[:37] + "..."
        text = text.replace("\n", "\\n")
        bits.append(f"[TEXT {text!r}]")

    if finish is not None:
        bits.append(f"[FINISH reason={finish}]")

    # in-band dynamo timing on the finish chunk
    nvext = chunk.get("nvext") or {}
    t = nvext.get("timing")
    if isinstance(t, dict):
        bits.append(
            f"[nvext total={t.get('total_time_ms')}ms "
            f"req_received={t.get('request_received_ms')}]"
        )

    if not bits:
        bits.append("[KEEPALIVE]" if not delta else f"[delta_keys={list(delta.keys())}]")

    return " ".join(bits)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", type=Path, help="Path to .timed capture file")
    args = ap.parse_args(argv)

    if not args.path.exists():
        print(f"file not found: {args.path}", file=sys.stderr)
        return 2

    prev_ts: float | None = None
    first_ts: float | None = None
    n_events = 0
    n_text = 0
    n_tool = 0

    with args.path.open() as f:
        for raw in f:
            raw = raw.rstrip("\n")
            if not raw.strip():
                continue
            m = _LINE_RE.match(raw)
            if not m:
                continue
            ts = float(m.group("ts"))
            body = m.group("body").strip()
            if first_ts is None:
                first_ts = ts

            if body == "[DONE]":
                label = "[DONE]"
            else:
                try:
                    chunk = json.loads(body)
                except json.JSONDecodeError:
                    label = f"[UNPARSEABLE {body[:60]!r}]"
                    chunk = None
                else:
                    label = _label(chunk)
                    if chunk:
                        c0 = (chunk.get("choices") or [{}])[0]
                        delta = c0.get("delta") or {}
                        if delta.get("content"):
                            n_text += 1
                        if delta.get("tool_calls"):
                            n_tool += 1

            delta_s = 0.0 if prev_ts is None else ts - prev_ts
            since_start_s = ts - (first_ts or ts)
            print(f"+{delta_s:7.4f}s  t={since_start_s:7.4f}s  {label}")
            prev_ts = ts
            n_events += 1

    if first_ts is not None and prev_ts is not None:
        print()
        print(
            f"events={n_events}  text_chunks={n_text}  "
            f"tool_call_chunks={n_tool}  total_wall_s={prev_ts - first_ts:.4f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
