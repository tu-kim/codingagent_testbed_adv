#!/usr/bin/env python3
"""Convert NDJSON (one JSON object per line) to a single JSON array.

Useful when a downstream tool (jq's `.[]`, a notebook's `json.load`,
`pandas.read_json(... orient='records')`, etc.) wants a single
array rather than line-delimited objects.

Input:
  - file path passed as the first positional arg, OR
  - "-" / no arg  → read from stdin

Output:
  --output <path>  write to file
  (default)        write to stdout

Flags:
  --pretty         indent=2; default emits compact (no whitespace)
  --strict         raise on malformed JSON lines; default skips them
                   with a stderr warning

Usage:
  scripts/jsonl_to_json.py logs/resource.ndjson --output logs/resource.json
  scripts/jsonl_to_json.py logs/vllm_metrics.ndjson --pretty > metrics.json
  cat logs/foo.ndjson | scripts/jsonl_to_json.py - > foo.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import IO, Iterable


def iter_objects(stream: IO[str], *, strict: bool = False) -> Iterable[object]:
    """Yield one parsed object per non-blank line. With strict=False
    (default), log malformed lines to stderr and continue; with
    strict=True, raise json.JSONDecodeError on the first bad line."""
    for lineno, raw in enumerate(stream, start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as e:
            if strict:
                raise
            print(f"jsonl_to_json: skipping malformed line {lineno}: {e}",
                  file=sys.stderr)


def convert(in_stream: IO[str], out_stream: IO[str],
            *, pretty: bool, strict: bool) -> int:
    """Stream-write a JSON array. Returns count of objects emitted."""
    indent = 2 if pretty else None
    sep_item = ",\n  " if pretty else ","
    n = 0
    first = True
    out_stream.write("[\n  " if pretty else "[")
    for obj in iter_objects(in_stream, strict=strict):
        if not first:
            out_stream.write(sep_item)
        # Each element keeps its own indentation; this matches what
        # json.dumps([...], indent=2) would produce for the array.
        out_stream.write(json.dumps(obj, indent=indent, ensure_ascii=False))
        first = False
        n += 1
    out_stream.write("\n]\n" if pretty else "]")
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("input", nargs="?", default="-",
                    help="Input NDJSON file. '-' or omitted = stdin.")
    ap.add_argument("--output", "-o", default=None, type=Path,
                    help="Output JSON file. Default: stdout.")
    ap.add_argument("--pretty", action="store_true",
                    help="Indent output (indent=2). Default: compact.")
    ap.add_argument("--strict", action="store_true",
                    help="Fail on malformed JSON lines. Default: skip "
                         "with stderr warning.")
    args = ap.parse_args(argv)

    if args.input == "-":
        in_ctx = sys.stdin
        close_in = False
    else:
        in_path = Path(args.input)
        if not in_path.exists():
            print(f"input not found: {in_path}", file=sys.stderr)
            return 2
        # utf-8-sig strips an optional BOM (﻿) so a BOM-prefixed
        # NDJSON doesn't make the first line silently fail JSON parse.
        in_ctx = in_path.open(encoding="utf-8-sig", errors="replace")
        close_in = True

    if args.output is None:
        out_ctx = sys.stdout
        close_out = False
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out_ctx = args.output.open("w", encoding="utf-8")
        close_out = True

    try:
        n = convert(in_ctx, out_ctx, pretty=args.pretty, strict=args.strict)
    finally:
        if close_in:
            in_ctx.close()
        if close_out:
            out_ctx.close()

    if args.output is not None:
        print(f"jsonl_to_json: wrote {n} objects to {args.output}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
