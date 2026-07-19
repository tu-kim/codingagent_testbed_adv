#!/usr/bin/env bash
# Concatenate per-session OpenCode profile NDJSON files into one stream.
#
# OpenCode writes <OPENCODE_PROFILE_DIR>/<sessionID>.jsonl when
# OPENCODE_PROFILE is enabled. testbed.sh up_opencode points
# OPENCODE_PROFILE_DIR at <workspace_root>/profiles, so:
#
#   scripts/aggregate_profiles.sh /tmp/testbed-workspaces            \
#     > results/run1/profiles.jsonl                                  # aggregate
#   scripts/aggregate_profiles.sh /tmp/testbed-workspaces ses_abcdef \
#     | jq -s 'sort_by(.ts)'                                          # filter by session
#
# Each row already carries `sessionID` so the merged stream is self-describing.

set -Eeuo pipefail

usage() {
  cat >&2 <<EOF
usage: $(basename "$0") <workspace_root_or_profile_dir> [session_substring] [--trace trace.jsonl]

Reads <workspace_root>/profiles/*.jsonl (or <profile_dir>/*.jsonl when the
path already points directly at a profiles directory) and writes the
concatenation to stdout. Rows preserve their original order within each
session; use \`jq -s 'sort_by(.ts)'\` downstream if you want a globally
time-sorted merge.

  --trace <trace.jsonl>  Keep only the run's MAIN sessions (session_ids in
                         the trace). profiles/ also holds title-generation
                         and \`task\` sub-agent sessions; this drops them so
                         the aggregate matches the trace-filtered analyses.
EOF
}

ROOT=""
FILTER=""
TRACE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --trace) TRACE="${2-}"; shift 2 || { usage; exit 2; } ;;
    --trace=*) TRACE="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1" >&2; usage; exit 2 ;;
    *)
      if [[ -z "$ROOT" ]]; then ROOT="$1"
      elif [[ -z "$FILTER" ]]; then FILTER="$1"
      else echo "too many positional args" >&2; usage; exit 2
      fi
      shift ;;
  esac
done

if [[ -z "$ROOT" ]]; then
  usage; exit 2
fi

if [[ ! -d "$ROOT" ]]; then
  echo "not a directory: $ROOT" >&2
  exit 1
fi

# Auto-detect: if <ROOT>/profiles exists use that, else assume ROOT itself
# already points at the profile dir.
PROFILE_DIR="$ROOT"
if [[ -d "$ROOT/profiles" ]]; then
  PROFILE_DIR="$ROOT/profiles"
fi

shopt -s nullglob
files=("$PROFILE_DIR"/*.jsonl)
shopt -u nullglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "no *.jsonl under $PROFILE_DIR" >&2
  exit 1
fi

# Build the MAIN-session allowlist from the trace into a newline-delimited
# temp file (avoids bash-4 associative arrays for portability; membership
# is tested with grep -Fxq). basename minus .jsonl is the sessionID. Uses
# python3 (a hard dep of the repo) so the JSONL parse tolerates blank lines.
KEEP_FILE=""
if [[ -n "$TRACE" ]]; then
  if [[ ! -f "$TRACE" ]]; then
    echo "trace not found: $TRACE" >&2; exit 1
  fi
  KEEP_FILE="$(mktemp)"
  trap 'rm -f "$KEEP_FILE"' EXIT
  python3 -c '
import json, sys
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = line.strip()
    if not line:
        continue
    try:
        rec = json.loads(line)
    except Exception:
        continue
    s = rec.get("session_id")
    if s:
        print(s)
' "$TRACE" > "$KEEP_FILE"
  if [[ ! -s "$KEEP_FILE" ]]; then
    echo "no session_id in $TRACE" >&2; exit 1
  fi
fi

for f in "${files[@]}"; do
  base="$(basename "$f")"
  if [[ -n "$FILTER" ]]; then
    case "$base" in
      *"$FILTER"*) ;;
      *) continue ;;
    esac
  fi
  # --trace: keep only files whose sessionID (basename minus .jsonl) is a
  # main session in the trace.
  if [[ -n "$KEEP_FILE" ]] && ! grep -Fxq -- "${base%.jsonl}" "$KEEP_FILE"; then
    continue
  fi
  cat "$f"
done
