#!/usr/bin/env bash
# Reproduce the "large tool-call buffers, text streams" phenomenon at the WIRE
# level, to locate where opencode's late `start-step` comes from.
#
# Fires two streaming /v1/chat/completions requests at the Dynamo frontend --
# one FORCED to emit a large tool-call, one plain text -- timestamps every SSE
# line as curl receives it, then runs scripts/sse_chunk_timing.py on each so the
# inter-chunk deltas show the delivery pattern.
#
# Interpretation:
#   * TOOL-CALL capture = long silence (generation) then a BURST of [TOOL_CALL]
#     chunks at the end, text_chunks=0  -> the frontend/vLLM BUFFERS structured
#     tool-calls (qwen3_coder parser can't emit partial JSON). That buffered
#     delivery is why opencode's start-step (llm.start) fires only at the end
#     and the real LLM time leaks into "others".
#   * TEXT capture = [TEXT] chunks with small steady deltas -> text streams fine.
#   * If TOOL-CALL ALSO streams steadily, the buffering is client-side
#     (AI-SDK/opencode), not the server -> a different harness is needed.
#
# Requires the stack up (workers + frontend) and GNU date (`date +%N`).
#
# Usage:
#   scripts/repro_toolcall_buffering.sh
#   MAX_TOKENS=8192 OUT_DIR=logs/repro scripts/repro_toolcall_buffering.sh
set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="${TESTBED_CONFIG:-$REPO_ROOT/deploy/testbed.yaml}"
TIMER="$REPO_ROOT/scripts/sse_chunk_timing.py"

for t in curl yq python3; do
  command -v "$t" >/dev/null 2>&1 || { echo "missing tool: $t" >&2; exit 1; }
done
[[ -f "$CFG" ]]    || { echo "config not found: $CFG" >&2; exit 1; }
[[ -f "$TIMER" ]]  || { echo "missing $TIMER" >&2; exit 1; }

cfg() { yq -r "$1" "$CFG"; }
DYN="http://$(cfg .dynamo.host):$(cfg .dynamo.port)"
SERVED="$(cfg .model.served_name)"
MAX_TOKENS="${MAX_TOKENS:-4096}"
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
mkdir -p "$OUT_DIR"

# Stream a request body, tagging each received SSE line with a wall timestamp.
timed() {  # timed <body-json> <out.timed>
  local body="$1" out="$2"
  curl -N -sS -H 'content-type: application/json' -d "$body" \
       "$DYN/v1/chat/completions" \
  | while IFS= read -r line; do
      printf '%s.%06d  %s\n' "$(date +%s)" "$(date +%N | cut -c1-6)" "$line"
    done | tee "$out" >/dev/null
}

# (A) FORCE a large tool-call (tool_choice pins the function).
read -r -d '' TOOL_BODY <<JSON || true
{"model":"$SERVED","stream":true,"stream_options":{"include_usage":true},
 "max_tokens":$MAX_TOKENS,
 "messages":[{"role":"user","content":"Create hello.py: a Python file with 200 functions f1..f200, each printing its own number. Emit the ENTIRE file content via the write_file tool."}],
 "tools":[{"type":"function","function":{"name":"write_file","description":"write a file",
   "parameters":{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}}}],
 "tool_choice":{"type":"function","function":{"name":"write_file"}}}
JSON

# (B) Control: a similarly large PLAIN-TEXT response, no tools.
read -r -d '' TEXT_BODY <<JSON || true
{"model":"$SERVED","stream":true,"stream_options":{"include_usage":true},
 "max_tokens":$MAX_TOKENS,
 "messages":[{"role":"user","content":"Write a 200-line Python file with functions f1..f200, each printing its own number. Output the full code as plain text, do not use any tools."}]}
JSON

echo "DYN=$DYN  SERVED=$SERVED  MAX_TOKENS=$MAX_TOKENS  OUT_DIR=$OUT_DIR"
echo
echo "== firing TOOL-CALL request =="
timed "$TOOL_BODY" "$OUT_DIR/toolcall.timed"
echo "== firing TEXT (control) request =="
timed "$TEXT_BODY" "$OUT_DIR/text.timed"

echo
echo "############ TOOL-CALL chunk timing ############"
python3 "$TIMER" "$OUT_DIR/toolcall.timed"
echo
echo "############ TEXT (control) chunk timing ############"
python3 "$TIMER" "$OUT_DIR/text.timed"

echo
echo "captures: $OUT_DIR/{toolcall,text}.timed"
echo "verdict: a long gap then a [TOOL_CALL] burst (with text_chunks=0) confirms"
echo "         server-side buffering of tool-calls == opencode's late start-step."
