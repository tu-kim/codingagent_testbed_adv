#!/usr/bin/env python3
"""Cross-check OpenCode profile NDJSON against Dynamo frontend request logs.

For each `llm.end` event in the profile, pair it with the corresponding
`request completed` line from the Dynamo log and compare:

  * prompt = tokens.input + tokens.cache.read (profile, normalized AI SDK
    shape from opencode session.ts getUsage) ↔ input_tokens (dynamo ISL)
  * completion = tokens.output (profile)      ↔ output_tokens (dynamo OSL)
  * step_duration_s (profile)                 ↔ elapsed_ms/1000 (dynamo)
  * duration_s (profile, up to first tool.start)   <-- where in elapsed
                                                       did the tool_call land
  * post_ttft_s = elapsed_ms - ttft_ms              (directly measured,
                                                     NOT derived from avg_itl_ms)

avg_itl_ms in dynamo is `(elapsed - ttft) / itl_count` where itl_count is
the count of tokens delivered AFTER the first SSE chunk (not output_tokens),
so multiplying it by (N-1) overestimates decode time when the first chunk
carries many tokens. Use post_ttft_s for honest comparison.

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


# Token/timing fields are injected via tracing `span.record(...)` at
# dynamo/lib/llm/src/http/service/metrics.rs:1466-1484 and only appear
# conditionally:
#   - input_tokens / output_tokens: only when MetricsCollector.finalize()
#     ran (success path); error/cancel paths may omit them entirely.
#   - ttft_ms:    only when TTFT was measured (>=1 generated token).
#   - avg_itl_ms: only when itl_count > 0 (= output_tokens >= 2).
# Values are sometimes quoted (the `format!("{:.2}", ...)` fields:
# ttft_ms, avg_itl_ms) and sometimes bare (integers). Parse each field
# independently so a missing optional field doesn't drop the whole line.
# Field accessors tolerate both tracing's pretty format (`k=v` or `k="v"`)
# and JSON-formatter output (`"k":v` or `"k":"v"`). Production deployments
# often configure tracing-subscriber for JSON to feed log aggregators.
def _field(name: str, valpat: str) -> re.Pattern[str]:
    return re.compile(
        rf'(?:\b|")\b{name}\b"?\s*[=:]\s*"?(?P<v>{valpat})"?'
    )


# Tracing-subscriber's pretty formatter writes ANSI SGR colour codes around
# both keys and `=` separators even when stdout is not a TTY (it only checks
# at startup). The codes break `key=value` parsing, so we strip them per
# line before applying field regexes. Disable on the dynamo side with
# RUST_LOG_STYLE=never or `--log-no-color`, but parsing tolerant is safer.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mGKHJ]")


_INPUT_TOKENS_RE = _field("input_tokens", r"\d+")
_OUTPUT_TOKENS_RE = _field("output_tokens", r"\d+")
_TTFT_RE = _field("ttft_ms", r"[\d.]+")
_AVG_ITL_RE = _field("avg_itl_ms", r"[\d.]+")
_ELAPSED_RE = _field("elapsed_ms", r"\d+")
_STATUS_RE = _field("status", r"\w+")
# `model=` appears twice on a `request completed` line: bare from
# InflightGuard::Drop's `tracing::info!`, then quoted from the span-
# recorded MetricsCollector::finalize. Take the LAST occurrence so we
# get the authoritative span-recorded value -- they should be equal in
# practice but the span-recorded one is set by finalize() right next
# to the token counts we care about.
_MODEL_RE = _field("model", r"[^\s\",}]+")


def _last_model(line: str) -> str | None:
    matches = _MODEL_RE.findall(line)
    return matches[-1] if matches else None


@dataclass
class DynamoEntry:
    line_no: int
    input_tokens: int | None
    output_tokens: int | None
    elapsed_ms: int
    ttft_ms: float | None
    avg_itl_ms: float | None
    model: str | None
    status: str | None

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ms / 1000.0

    @property
    def post_ttft_s(self) -> float | None:
        """Wall time spent streaming after TTFT: elapsed - ttft. Directly
        measured (no assumption about per-token latency)."""
        if self.ttft_ms is None:
            return None
        return max(0.0, (self.elapsed_ms - self.ttft_ms) / 1000.0)


@dataclass
class ProfileStep:
    step: int
    ts: float
    prompt_tokens: int | None
    completion_tokens: int | None
    duration_s: float | None
    step_duration_s: float | None
    finish: str | None


@dataclass
class DynamoParseStats:
    """Diagnostic counters surfaced by parse_dynamo_log."""
    total_completed: int = 0          # `request completed` lines seen
    matched: int = 0                  # entries returned (passed filter)
    filtered_by_model: int = 0
    missing_elapsed_ms: int = 0
    missing_input_tokens: int = 0     # error/cancel path; no token info
    missing_output_tokens: int = 0
    missing_ttft_ms: int = 0          # short / failed requests
    missing_avg_itl_ms: int = 0       # output_tokens < 2
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

            elapsed_m = _ELAPSED_RE.search(line)
            if not elapsed_m:
                s.missing_elapsed_ms += 1
                if s.first_dropped_sample is None:
                    s.first_dropped_sample = line.rstrip()
                continue
            elapsed_ms = int(elapsed_m.group("v"))

            model = _last_model(line)
            if model_filter and model != model_filter:
                s.filtered_by_model += 1
                continue

            input_m = _INPUT_TOKENS_RE.search(line)
            output_m = _OUTPUT_TOKENS_RE.search(line)
            ttft_m = _TTFT_RE.search(line)
            itl_m = _AVG_ITL_RE.search(line)
            status_m = _STATUS_RE.search(line)

            if not input_m:
                s.missing_input_tokens += 1
            if not output_m:
                s.missing_output_tokens += 1
            if not ttft_m:
                s.missing_ttft_ms += 1
            if not itl_m:
                s.missing_avg_itl_ms += 1

            out.append(
                DynamoEntry(
                    line_no=i,
                    input_tokens=int(input_m.group("v")) if input_m else None,
                    output_tokens=int(output_m.group("v")) if output_m else None,
                    elapsed_ms=elapsed_ms,
                    ttft_ms=float(ttft_m.group("v")) if ttft_m else None,
                    avg_itl_ms=float(itl_m.group("v")) if itl_m else None,
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

    # Classic OpenAI shape -> use directly.
    p = tokens.get("prompt_tokens") or tokens.get("promptTokens")
    c = tokens.get("completion_tokens") or tokens.get("completionTokens")
    if p is not None or c is not None:
        return p, c

    # AI SDK v5/v6 normalized shape used by opencode.
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
        ("d_post_ttft_s", 14),
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
                _fmt(None, 14),
                _fmt(None, 12),
                " ".join(flags),
            ]
            lines.append(" | ".join(row))
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
            # Likely an error/cancel path where MetricsCollector.finalize()
            # didn't run -- token comparison is impossible for this row.
            flags.append("NO_DYNAMO_TOKENS")
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
            _fmt(d.ttft_ms / 1000.0 if d.ttft_ms is not None else None, 9),
            _fmt(d.post_ttft_s, 14),
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
    if stats.missing_elapsed_ms:
        print(f"  skipped (no elapsed_ms): {stats.missing_elapsed_ms}")
        if stats.first_dropped_sample:
            print(f"    first dropped line: {stats.first_dropped_sample[:240]}")
    if stats.missing_input_tokens or stats.missing_output_tokens:
        print(
            f"  token fields missing on matched rows: "
            f"input={stats.missing_input_tokens} output={stats.missing_output_tokens} "
            f"(error/cancel paths — finalize() didn't run)"
        )
    if stats.missing_ttft_ms or stats.missing_avg_itl_ms:
        print(
            f"  timing fields missing on matched rows: "
            f"ttft_ms={stats.missing_ttft_ms} avg_itl_ms={stats.missing_avg_itl_ms} "
            f"(avg_itl_ms only recorded when output_tokens >= 2)"
        )
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
