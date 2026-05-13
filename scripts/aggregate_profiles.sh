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
usage: $(basename "$0") <workspace_root_or_profile_dir> [session_substring]

Reads <workspace_root>/profiles/*.jsonl (or <profile_dir>/*.jsonl when the
path already points directly at a profiles directory) and writes the
concatenation to stdout. Rows preserve their original order within each
session; use \`jq -s 'sort_by(.ts)'\` downstream if you want a globally
time-sorted merge.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage; exit 2
fi

ROOT="$1"
FILTER="${2-}"

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

for f in "${files[@]}"; do
  if [[ -n "$FILTER" ]]; then
    case "$(basename "$f")" in
      *"$FILTER"*) ;;
      *) continue ;;
    esac
  fi
  cat "$f"
done
