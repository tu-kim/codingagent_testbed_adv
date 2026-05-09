#!/usr/bin/env bash
# Pretty-print a results/<run>/trace.jsonl produced by `python -m testbed run`.
#
# Usage:
#   scripts/view_trace.sh <trace.jsonl>                       # all instances
#   scripts/view_trace.sh <trace.jsonl> <instance_id_substr>  # filter
#
# Output (one block per task):
#   === <instance_id>  rtt=<rtt>s  ok=<bool>  error=<stage|->  ===
#   [user]      <first 120 chars of prompt>
#   [assistant] tool: bash {"command":"ls -la"}
#   [tool]      bash → "manage.py settings.py..."
#   [assistant] text: "I've patched ..."
#
# Field source: opencode list_messages returns [{info:{role,...}, parts:[...]}]
# (per CLAUDE.md "Module contracts"). Part shape is heterogeneous; we extract
# the most common keys (text, tool/name, input/output, file path) and fall
# back to the part type if nothing useful matches.

set -Eeuo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <trace.jsonl> [instance_id_substring]" >&2
  exit 2
fi

TRACE="$1"
FILTER="${2-}"

[[ -f "$TRACE" ]] || { echo "trace not found: $TRACE" >&2; exit 1; }
command -v jq >/dev/null 2>&1 || { echo "jq not in PATH" >&2; exit 1; }

# --filter applied via jq if non-empty.
JQ_FILTER='
def shorten(n): if (. | length) <= n then . else .[0:n] + "..." end;

def part_summary:
  . as $p
  | (.type // "?") as $t
  | (
      if   $p.text             then "text: " + ($p.text | tostring | shorten(160))
      elif $p.tool             then "tool: " + $p.tool +
                                    (if $p.input  then " " + ($p.input  | tojson | shorten(160)) else "" end) +
                                    (if $p.output then " → " + ($p.output | tostring | shorten(160)) else "" end)
      elif $p.name             then "tool: " + $p.name +
                                    (if $p.input  then " " + ($p.input  | tojson | shorten(160)) else "" end)
      elif $p.path             then "file: " + $p.path
      elif $p.command          then "cmd: " + ($p.command | shorten(160))
      else $t
      end
    );

def task:
  ( "=== " + .instance_id
    # null rtt_s (clone/session failures) renders as "-" so it does not look
    # like an instant-success "rtt=0s".
    + "  rtt=" + ((.rtt_s // "-") | tostring) + "s"
    + "  ok=" + (.success | tostring)
    + "  error=" + ((.error.stage // "-") | tostring)
    + "  session=" + ((.session_id // "-") | tostring)
    + " ===" ),
  ( (.messages // [])[]
    | "[" + ((.info.role // "?") | tostring) + "] "
      + ((.parts // []) | map(part_summary) | join("  |  "))
  ),
  "";

if $filter == "" then .
else select(.instance_id | contains($filter))
end | task
'

jq -r --arg filter "$FILTER" "$JQ_FILTER" "$TRACE"
