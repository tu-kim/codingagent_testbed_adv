#!/usr/bin/env python3
"""Cross-check OpenCode profile NDJSON against Dynamo frontend request logs.

For each `llm.end` event in the profile, pair it with the corresponding
`request completed` line from the Dynamo log and compare:

  * prompt_tokens (profile)      ↔ input_tokens  (dynamo)
  * completion_tokens (profile)  ↔ output_tokens (dynamo)
  * step_duration_s (profile)    ↔ elapsed_ms/1000 (dynamo)
  * duration_s (profile, up to first tool.start)   <-- where in elapsed
                                                       did the tool_call land
  * expected_decode_s = output_tokens * avg_itl_ms / 1000

The duration delta (step_duration_s - elapsed_ms/1000) reflects AI-SDK /
processor.ts overhead on top of the LLM stream as seen by Dynamo.

Matching is order-based: profile llm.end events sorted by ts are paired
1:1 with `request completed` lines in the dynamo log. Assumes a single
OpenCode session (no concurrent runs against the same Dynamo). If tokens
don't match for a pair, it's flagged in the output.

Usage:
  scripts/match_profile_vs_dynamo.py \\
      --profile <workspace_root>/profiles/ses_xxx.ndjson \\
      --dynamo-log logs/frontend.log \\
      [--model qwen3-coder-30b-a3b-instruct-fp8] \\
      [--strict]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


# `request completed` line shape (one example, fields appear twice from
# nested tracing spans -- we only grab what's unique to the metrics
# middleware: input_tokens, output_tokens, ttft_ms, avg_itl_ms).
_DYNAMO_TOKENS_RE = re.compile(
    r"\binput_tokens=(?P<input_tokens>\d+)\s+"
    r"output_tokens=(?P<output_tokens>\d+)\s+"
    r'ttft_ms="(?P<ttft_ms>[\d.]+)"\s+'
    r'avg_itl_ms="(?P<avg_itl_ms>[\d.]+)"'
)
_ELAPSED_RE = re.compile(r"\belapsed_ms=(?P<elapsed_ms>\d+)\b")
_MODEL_QUOTED_RE = re.compile(r'\bmodel="(?P<model>[^"]+)"')
_STATUS_RE = re.compile(r"\bstatus=(?P<status>\w+)\b")


@dataclass
class DynamoEntry:
    line_no: int
    input_tokens: int
    output_tokens: int
    elapsed_ms: int
    ttft_ms: float
    avg_itl_ms: float
    model: str | None
    status: str | None

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ms / 1000.0

    @property
    def expected_decode_s(self) -> float:
        # ITL is the gap *between* tokens — N tokens have N-1 gaps.
        # First token's latency is captured separately in ttft_ms.
        gaps = max(0, self.output_tokens - 1)
        return (gaps * self.avg_itl_ms) / 1000.0


@dataclass
class ProfileStep:
    step: int
    ts: float
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_s: float | None
    step_duration_s: float | None
    finish: str | None


def parse_dynamo_log(path: Path, model_filter: str | None) -> list[DynamoEntry]:
    out: list[DynamoEntry] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if "request completed" not in line:
                continue
            m = _DYNAMO_TOKENS_RE.search(line)
            if not m:
                continue
            elapsed_m = _ELAPSED_RE.search(line)
            model_m = _MODEL_QUOTED_RE.search(line)
            status_m = _STATUS_RE.search(line)
            model = model_m.group("model") if model_m else None
            if model_filter and model != model_filter:
                continue
            out.append(
                DynamoEntry(
                    line_no=i,
                    input_tokens=int(m.group("input_tokens")),
                    output_tokens=int(m.group("output_tokens")),
                    elapsed_ms=int(elapsed_m.group("elapsed_ms")) if elapsed_m else 0,
                    ttft_ms=float(m.group("ttft_ms")),
                    avg_itl_ms=float(m.group("avg_itl_ms")),
                    model=model,
                    status=status_m.group("status") if status_m else None,
                )
            )
    return out


def parse_profile(path: Path) -> list[ProfileStep]:
    out: list[ProfileStep] = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("ev") != "llm.end":
                continue
            tokens = ev.get("tokens") or {}
            out.append(
                ProfileStep(
                    step=ev.get("step", -1),
                    ts=float(ev.get("ts", 0.0)),
                    prompt_tokens=tokens.get("prompt_tokens") or tokens.get("promptTokens"),
                    completion_tokens=tokens.get("completion_tokens") or tokens.get("completionTokens"),
                    duration_s=ev.get("duration_s"),
                    step_duration_s=ev.get("step_duration_s"),
                    finish=ev.get("finish"),
                )
            )
    out.sort(key=lambda p: p.ts)
    return out


def match_in_order(
    profile: list[ProfileStep], dynamo: list[DynamoEntry]
) -> list[tuple[ProfileStep, DynamoEntry | None]]:
    pairs: list[tuple[ProfileStep, DynamoEntry | None]] = []
    for idx, p in enumerate(profile):
        d = dynamo[idx] if idx < len(dynamo) else None
        pairs.append((p, d))
    return pairs


def _fmt(v: object, width: int = 7, prec: int = 3) -> str:
    if v is None or v == "":
        return " " * (width - 3) + "N/A"
    if isinstance(v, float):
        return f"{v:>{width}.{prec}f}"
    return f"{v:>{width}}"


def render_table(pairs: Iterable[tuple[ProfileStep, DynamoEntry | None]]) -> tuple[str, int]:
    headers = [
        ("step", 4),
        ("p_prompt", 8),
        ("d_input", 7),
        ("p_compl", 7),
        ("d_out", 5),
        ("p_dur_s", 8),
        ("p_step_s", 9),
        ("d_elap_s", 9),
        ("d_ttft_s", 9),
        ("d_decode_s", 11),
        ("framework_s", 12),
        ("flags", 0),
    ]
    head_line = " | ".join(f"{name:>{w}}" if w else name for name, w in headers)
    sep_line = "-" * len(head_line)
    lines = [head_line, sep_line]

    mismatches = 0
    for p, d in pairs:
        flags: list[str] = []
        if d is None:
            flags.append("NO_DYNAMO")
            mismatches += 1
            row = [
                _fmt(p.step, 4),
                _fmt(p.prompt_tokens, 8),
                _fmt(None, 7),
                _fmt(p.completion_tokens, 7),
                _fmt(None, 5),
                _fmt(p.duration_s, 8),
                _fmt(p.step_duration_s, 9),
                _fmt(None, 9),
                _fmt(None, 9),
                _fmt(None, 11),
                _fmt(None, 12),
                " ".join(flags),
            ]
            lines.append(" | ".join(row))
            continue

        if p.prompt_tokens is not None and p.prompt_tokens != d.input_tokens:
            flags.append(f"PROMPT_DIFF({p.prompt_tokens}!={d.input_tokens})")
            mismatches += 1
        if p.completion_tokens is not None and p.completion_tokens != d.output_tokens:
            flags.append(f"COMPL_DIFF({p.completion_tokens}!={d.output_tokens})")
            mismatches += 1
        if d.status and d.status != "success":
            flags.append(f"STATUS={d.status}")

        framework_s: float | None = None
        if p.step_duration_s is not None:
            framework_s = p.step_duration_s - d.elapsed_s
            if framework_s < -0.05:
                # Profile bracket shorter than dynamo elapsed by >50ms means
                # the LLM call started before start-step was hooked, or
                # finish-step fired before stream actually closed --
                # either way suspicious.
                flags.append(f"NEG_FRAMEWORK({framework_s:+.2f}s)")
                mismatches += 1

        row = [
            _fmt(p.step, 4),
            _fmt(p.prompt_tokens, 8),
            _fmt(d.input_tokens, 7),
            _fmt(p.completion_tokens, 7),
            _fmt(d.output_tokens, 5),
            _fmt(p.duration_s, 8),
            _fmt(p.step_duration_s, 9),
            _fmt(d.elapsed_s, 9),
            _fmt(d.ttft_ms / 1000.0, 9),
            _fmt(d.expected_decode_s, 11),
            _fmt(framework_s, 12),
            " ".join(flags),
        ]
        lines.append(" | ".join(row))

    return "\n".join(lines), mismatches


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Match profile NDJSON llm.end events against Dynamo `request completed` log lines.",
    )
    ap.add_argument("--profile", required=True, type=Path, help="Profile NDJSON for ONE session")
    ap.add_argument("--dynamo-log", required=True, type=Path, help="Dynamo frontend log path")
    ap.add_argument("--model", default=None, help="Filter Dynamo entries by model= value")
    ap.add_argument("--strict", action="store_true", help="Exit 1 if any mismatch is found")
    args = ap.parse_args(argv)

    if not args.profile.exists():
        print(f"profile not found: {args.profile}", file=sys.stderr)
        return 2
    if not args.dynamo_log.exists():
        print(f"dynamo log not found: {args.dynamo_log}", file=sys.stderr)
        return 2

    profile = parse_profile(args.profile)
    dynamo = parse_dynamo_log(args.dynamo_log, args.model)

    print(f"profile llm.end events:   {len(profile)}")
    print(f"dynamo request completed: {len(dynamo)}")
    if len(profile) != len(dynamo):
        print(
            "  WARN: count mismatch — order-based pairing will leave excess entries unmatched.",
            file=sys.stderr,
        )
    print()

    table, mismatches = render_table(match_in_order(profile, dynamo))
    print(table)
    print()
    print(f"flagged rows: {mismatches}")

    # Aggregate framework overhead (profile step_duration vs dynamo elapsed).
    deltas = []
    for p, d in match_in_order(profile, dynamo):
        if d is None or p.step_duration_s is None:
            continue
        deltas.append(p.step_duration_s - d.elapsed_s)
    if deltas:
        deltas.sort()
        n = len(deltas)
        med = deltas[n // 2]
        print(
            f"framework overhead (step_duration_s - dynamo_elapsed_s): "
            f"n={n}, median={med:+.3f}s, min={deltas[0]:+.3f}s, max={deltas[-1]:+.3f}s"
        )

    return 1 if (args.strict and mismatches) else 0


if __name__ == "__main__":
    sys.exit(main())
