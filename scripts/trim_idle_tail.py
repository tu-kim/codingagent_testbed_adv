#!/usr/bin/env python3
"""Trim trailing (and leading) idle samples from monitor/scrape NDJSON.

`monitor_resources.py` and `scrape_vllm_metrics.py` keep writing every
tick until SIGTERM. If you forget to `testbed.sh down monitor /
scrape_metrics` immediately when the workload finishes, the NDJSON
tail is padded with idle samples that skew downstream stats.

This script derives the workload's [first, last] ts window from the
opencode profile NDJSON (`query.start` / `query.end` events across
every session under --profile-dir) and filters one or more target
NDJSON files to that window. Output goes to `<target>.trimmed.ndjson`
unless `--in-place` is passed (which renames the original to .bak
first so nothing is lost).

Usage:
  # auto-detect window from profile dir, write .trimmed siblings
  scripts/trim_idle_tail.py \\
      --profile-dir /tmp/testbed-workspaces/profiles \\
      --target logs/resource.ndjson \\
      --target logs/vllm_metrics.ndjson

  # or specify the window explicitly (unix seconds)
  scripts/trim_idle_tail.py \\
      --start 1779200000.0 --end 1779203600.0 \\
      --target logs/resource.ndjson \\
      --in-place
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def detect_window(profile_dir: Path) -> tuple[float, float]:
    """Return (first_query_start_ts, last_query_end_ts) across all
    session NDJSON files under profile_dir. Falls back to last-seen ts
    per session when query.end is missing (mid-flight / crashed run).

    Raises SystemExit when no sessions found."""
    starts: list[float] = []
    ends: list[float] = []
    last_seen: dict[str, float] = {}
    started: set[str] = set()
    ended: set[str] = set()

    for jsonl in sorted(profile_dir.glob("*.jsonl")):
        with jsonl.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = ev.get("sessionID")
                ts = ev.get("ts")
                if not sid or not isinstance(ts, (int, float)):
                    continue
                last_seen[sid] = max(last_seen.get(sid, ts), ts)
                if ev.get("ev") == "query.start":
                    starts.append(ts)
                    started.add(sid)
                elif ev.get("ev") == "query.end":
                    ends.append(ts)
                    ended.add(sid)

    # Fall back to last-seen ts for sessions that started but never
    # emitted query.end (mid-flight when monitor was stopped, crash).
    for sid in started - ended:
        if sid in last_seen:
            ends.append(last_seen[sid])

    if not starts:
        raise SystemExit(
            f"no query.start events found under {profile_dir} — "
            "pass --start/--end manually if the profile dir is elsewhere"
        )
    return min(starts), max(ends) if ends else max(starts)


def trim_file(target: Path, start: float, end: float,
              in_place: bool = False) -> tuple[int, int, Path]:
    """Read `target`, keep rows where start <= row.ts <= end.
    Return (rows_kept, rows_total, output_path)."""
    if not target.exists():
        raise SystemExit(f"target not found: {target}")
    kept_lines: list[str] = []
    total = 0
    with target.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            total += 1
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                # Preserve malformed rows so trim never silently drops data
                # unrelated to the time filter.
                kept_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            ts = row.get("ts")
            if not isinstance(ts, (int, float)):
                continue
            if start <= ts <= end:
                kept_lines.append(line if line.endswith("\n") else line + "\n")

    if in_place:
        bak = target.with_suffix(target.suffix + ".bak")
        shutil.move(str(target), str(bak))
        target.write_text("".join(kept_lines))
        return len(kept_lines), total, target
    out = target.with_suffix(target.suffix + ".trimmed" + target.suffix
                              if not target.name.endswith(".ndjson")
                              else "")
    # The .with_suffix dance gets weird for compound suffixes; build
    # explicitly: <stem>.trimmed<.ndjson|.jsonl|...>
    out = target.parent / f"{target.stem}.trimmed{target.suffix}"
    out.write_text("".join(kept_lines))
    return len(kept_lines), total, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--profile-dir", type=Path,
                     help="Directory of opencode profile NDJSON files. "
                          "Window = (min query.start.ts, max query.end.ts).")
    src.add_argument("--window", nargs=2, type=float, metavar=("START", "END"),
                     help="Explicit window in unix seconds.")
    ap.add_argument("--target", action="append", required=True, type=Path,
                    help="NDJSON file to trim. Repeat for multiple files.")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite the target after saving a .bak. "
                         "Default writes to <stem>.trimmed<.ext>.")
    args = ap.parse_args(argv)

    if args.profile_dir:
        if not args.profile_dir.is_dir():
            print(f"profile dir not found: {args.profile_dir}", file=sys.stderr)
            return 2
        start, end = detect_window(args.profile_dir)
        print(f"window from profile dir: {start:.3f} → {end:.3f} ({end - start:.2f}s)")
    else:
        start, end = args.window
        print(f"window from --window: {start:.3f} → {end:.3f} ({end - start:.2f}s)")

    for target in args.target:
        kept, total, out_path = trim_file(target, start, end, in_place=args.in_place)
        dropped = total - kept
        pct = (dropped / total * 100.0) if total else 0.0
        print(f"  {target} → {out_path}: kept {kept}/{total} (dropped {dropped}, {pct:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
