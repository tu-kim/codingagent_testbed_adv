#!/usr/bin/env python3
"""Compare token counts between OpenCode profile NDJSON and Dynamo frontend logs.

For each `llm.end` event in the profile, pair it with the corresponding
`request completed` line from the Dynamo log and compare token counts only:

  * prompt = tokens.input + tokens.cache.read (profile, normalized AI SDK
    shape from opencode session.ts getUsage) ↔ input_tokens (dynamo ISL)
  * completion = tokens.output (profile)      ↔ output_tokens (dynamo OSL)

Matching is order-based: profile llm.end events sorted by ts are paired
1:1 with `request completed` lines in the dynamo log. Assumes a single
OpenCode session (no concurrent runs against the same Dynamo).

Usage:
  scripts/match_profile_vs_dynamo.py \\
      --profile <workspace_root>/profiles/ses_xxx.ndjson \\
      --dynamo-log logs/frontend.log \\
      [--model qwen3-coder-30b-a3b-instruct-fp8] \\
      [--skip-dynamo-leading N] \\
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


# Tracing-subscriber's pretty formatter wraps keys and `=` separators in
# ANSI SGR codes even when stdout is not a TTY. Strip them per line.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJ]")


def _field(name: str, valpat: str) -> re.Pattern[str]:
    """Field accessor tolerant of both tracing's pretty `k=v` / `k="v"`
    and JSON-formatter `"k":v` / `"k":"v"` shapes."""
    return re.compile(
        rf'(?:\b|")\b{name}\b"?\s*[=:]\s*"?(?P<v>{valpat})"?'
    )


_INPUT_TOKENS_RE = _field("input_tokens", r"\d+")
_OUTPUT_TOKENS_RE = _field("output_tokens", r"\d+")
_STATUS_RE = _field("status", r"\w+")
# `model=` appears twice on a `request completed` line (bare from
# InflightGuard core, quoted from MetricsCollector::finalize). Take the
# LAST occurrence -- it's the span-recorded one set right next to the
# token counts.
_MODEL_RE = _field("model", r"[^\s\",}]+")


def _last_model(line: str) -> str | None:
    matches = _MODEL_RE.findall(line)
    return matches[-1] if matches else None


@dataclass
class DynamoEntry:
    line_no: int
    input_tokens: int | None
    output_tokens: int | None
    model: str | None
    status: str | None


@dataclass
class ProfileStep:
    step: int
    ts: float
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass
class DynamoParseStats:
    """Diagnostic counters surfaced by parse_dynamo_log."""
    total_completed: int = 0          # `request completed` lines seen
    matched: int = 0                  # entries returned (passed filter)
    filtered_by_model: int = 0
    missing_input_tokens: int = 0     # error/cancel path; no token info
    missing_output_tokens: int = 0
    first_dropped_sample: str | None = None  # first line dropped at parse, for debugging


def parse_dynamo_log(
    path: Path,
    model_filter: str | None,
    stats: DynamoParseStats | None = None,
) -> list[DynamoEntry]:
    out: list[DynamoEntry] = []
    s = stats if stats is not None else DynamoParseStats()
    with path.open(encoding="utf-8", errors="replace") as f:
        for i, raw_line in enumerate(f, start=1):
            line = _ANSI_RE.sub("", raw_line)
            if "request completed" not in line:
                continue
            s.total_completed += 1

            model = _last_model(line)
            if model_filter and model != model_filter:
                s.filtered_by_model += 1
                continue

            input_m = _INPUT_TOKENS_RE.search(line)
            output_m = _OUTPUT_TOKENS_RE.search(line)
            status_m = _STATUS_RE.search(line)

            if not input_m:
                s.missing_input_tokens += 1
                if s.first_dropped_sample is None:
                    s.first_dropped_sample = line.rstrip()
            if not output_m:
                s.missing_output_tokens += 1

            out.append(
                DynamoEntry(
                    line_no=i,
                    input_tokens=int(input_m.group("v")) if input_m else None,
                    output_tokens=int(output_m.group("v")) if output_m else None,
                    model=model,
                    status=status_m.group("v") if status_m else None,
                )
            )
            s.matched += 1
    return out


def _extract_profile_tokens(tokens: dict) -> tuple[int | None, int | None]:
    """Pull (prompt, completion) from the AI SDK v5/v6 normalized shape
    that opencode emits (session.ts getUsage):
        { total, input, output, reasoning, cache: {read, write} }
    where `input` is already adjusted (cache tokens subtracted). To compare
    against dynamo's ISL we must add cache.read back. Falls back to the
    classic OpenAI shape (prompt_tokens / completion_tokens) and camelCase
    so older NDJSON files still parse.
    """
    if not tokens:
        return None, None

    p = tokens.get("prompt_tokens") or tokens.get("promptTokens")
    c = tokens.get("completion_tokens") or tokens.get("completionTokens")
    if p is not None or c is not None:
        return p, c

    inp = tokens.get("input")
    out = tokens.get("output")
    cache = tokens.get("cache") or {}
    cache_read = cache.get("read") if isinstance(cache, dict) else None

    prompt = None
    if inp is not None:
        prompt = inp + (cache_read or 0)

    return prompt, out


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
            prompt_tok, compl_tok = _extract_profile_tokens(ev.get("tokens") or {})
            out.append(
                ProfileStep(
                    step=ev.get("step", -1),
                    ts=float(ev.get("ts", 0.0)),
                    prompt_tokens=prompt_tok,
                    completion_tokens=compl_tok,
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


def _fmt(v: object, width: int) -> str:
    if v is None:
        return " " * (width - 3) + "N/A"
    return f"{v:>{width}}"


def render_table(pairs: Iterable[tuple[ProfileStep, DynamoEntry | None]]) -> tuple[str, int]:
    headers = [("step", 4), ("p_prompt", 8), ("d_input", 8),
               ("p_compl", 7), ("d_out", 7), ("flags", 0)]
    head_line = " | ".join(f"{name:>{w}}" if w else name for name, w in headers)
    lines = [head_line, "-" * len(head_line)]

    mismatches = 0
    for p, d in pairs:
        flags: list[str] = []
        if d is None:
            flags.append("NO_DYNAMO")
            mismatches += 1
            lines.append(" | ".join([
                _fmt(p.step, 4),
                _fmt(p.prompt_tokens, 8), _fmt(None, 8),
                _fmt(p.completion_tokens, 7), _fmt(None, 7),
                " ".join(flags),
            ]))
            continue

        if (p.prompt_tokens is not None and d.input_tokens is not None
                and p.prompt_tokens != d.input_tokens):
            flags.append(f"PROMPT_DIFF({p.prompt_tokens}!={d.input_tokens})")
            mismatches += 1
        if (p.completion_tokens is not None and d.output_tokens is not None
                and p.completion_tokens != d.output_tokens):
            flags.append(f"COMPL_DIFF({p.completion_tokens}!={d.output_tokens})")
            mismatches += 1
        if d.input_tokens is None and d.output_tokens is None:
            flags.append("NO_DYNAMO_TOKENS")
        if d.status and d.status != "success":
            flags.append(f"STATUS={d.status}")

        lines.append(" | ".join([
            _fmt(p.step, 4),
            _fmt(p.prompt_tokens, 8), _fmt(d.input_tokens, 8),
            _fmt(p.completion_tokens, 7), _fmt(d.output_tokens, 7),
            " ".join(flags),
        ]))

    return "\n".join(lines), mismatches


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Token-count cross-check: profile NDJSON ↔ Dynamo `request completed` log.",
    )
    ap.add_argument("--profile", required=True, type=Path, help="Profile NDJSON for ONE session")
    ap.add_argument("--dynamo-log", required=True, type=Path, help="Dynamo frontend log path")
    ap.add_argument("--model", default=None, help="Filter Dynamo entries by model= value")
    ap.add_argument(
        "--skip-dynamo-leading", type=int, default=0, metavar="N",
        help="Drop the first N matched Dynamo entries before pairing. Use when the "
             "frontend log has leading probe/warmup requests that don't appear in the "
             "profile NDJSON (common when opencode validates the model at startup).",
    )
    ap.add_argument("--strict", action="store_true", help="Exit 1 if any mismatch is found")
    args = ap.parse_args(argv)

    if not args.profile.exists():
        print(f"profile not found: {args.profile}", file=sys.stderr)
        return 2
    if not args.dynamo_log.exists():
        print(f"dynamo log not found: {args.dynamo_log}", file=sys.stderr)
        return 2

    profile = parse_profile(args.profile)
    stats = DynamoParseStats()
    dynamo = parse_dynamo_log(args.dynamo_log, args.model, stats=stats)

    if args.skip_dynamo_leading > 0:
        dropped = dynamo[: args.skip_dynamo_leading]
        dynamo = dynamo[args.skip_dynamo_leading :]
        print(
            f"--skip-dynamo-leading={args.skip_dynamo_leading}: dropped "
            f"{len(dropped)} leading entries "
            f"(line_nos: {[e.line_no for e in dropped]})"
        )

    print(f"profile llm.end events:   {len(profile)}")
    print(f"dynamo request completed: {stats.total_completed} (matched {stats.matched})")
    if stats.filtered_by_model:
        print(f"  filtered by --model:    {stats.filtered_by_model}")
    if stats.missing_input_tokens or stats.missing_output_tokens:
        print(
            f"  token fields missing on matched rows: "
            f"input={stats.missing_input_tokens} output={stats.missing_output_tokens} "
            f"(error/cancel paths — finalize() didn't run)"
        )
        if stats.first_dropped_sample:
            print(f"    first such line: {stats.first_dropped_sample[:240]}")
    if len(profile) != stats.matched:
        print(
            "  WARN: count mismatch — order-based pairing will leave excess entries unmatched.",
            file=sys.stderr,
        )
    print()

    table, mismatches = render_table(match_in_order(profile, dynamo))
    print(table)
    print()
    print(f"flagged rows: {mismatches}")

    return 1 if (args.strict and mismatches) else 0


if __name__ == "__main__":
    sys.exit(main())
