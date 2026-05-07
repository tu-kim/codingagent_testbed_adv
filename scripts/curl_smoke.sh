#!/usr/bin/env bash
# Single-request smoke tests against the running testbed stack.
#
# Subcommands:
#   routes    list OpenCode OpenAPI routes
#   dynamo    one /v1/chat/completions to Dynamo
#   opencode  full OpenCode session+message round trip
#   swebench  send a real SWE-bench prompt through OpenCode
#   all       run all four sequentially, fail-fast

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$REPO_ROOT/deploy/testbed.yaml"
PYTHON="${PYTHON:-python3}"

cfg() { yq -r "$1" "$CFG"; }

OC_HOST=$(cfg .opencode.host)
OC_PORT=$(cfg .opencode.port)
DYN_HOST=$(cfg .dynamo.host)
DYN_PORT=$(cfg .dynamo.port)
SERVED=$(cfg .model.served_name)

OC="http://${OC_HOST}:${OC_PORT}"
DYN="http://${DYN_HOST}:${DYN_PORT}"

require() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing tool: $1" >&2; exit 1; }
}
require curl
require jq

# Use a stable per-run directory so multiple smokes don't trample.
DIR_NAME="smoke-$(date -u +%Y%m%dT%H%M%S)-$$"
WORKSPACE_ROOT=$(cfg .workspace_root)
mkdir -p "$WORKSPACE_ROOT/$DIR_NAME"

smoke_routes() {
  echo "== routes =="
  curl -fsS "$OC/openapi.json" | jq '.paths | keys'
}

smoke_dynamo() {
  echo "== dynamo =="
  curl -fsS -H 'content-type: application/json' \
    -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
    "$DYN/v1/chat/completions" | jq '.choices[0].message // .'
}

smoke_opencode() {
  echo "== opencode =="
  local sid
  sid=$(curl -fsS -H 'content-type: application/json' \
    -d '{}' \
    "$OC/session?directory=$DIR_NAME" | jq -r '.id')
  echo "session: $sid"
  curl -fsS -H 'content-type: application/json' \
    -d '{"parts":[{"type":"text","text":"Reply with the single word: hello"}]}' \
    "$OC/session/$sid/message?directory=$DIR_NAME" | jq '.info.id // .'
  curl -fsS "$OC/session/$sid/message?directory=$DIR_NAME" | jq 'length'
}

smoke_swebench() {
  echo "== swebench =="
  local prompt
  prompt=$("$PYTHON" -m testbed.swebench)
  local payload
  payload=$(jq -nc --arg t "$prompt" '{parts:[{type:"text", text:$t}]}')
  local sid
  sid=$(curl -fsS -H 'content-type: application/json' \
    -d '{}' \
    "$OC/session?directory=$DIR_NAME" | jq -r '.id')
  echo "session: $sid"
  curl -fsS -H 'content-type: application/json' \
    -d "$payload" \
    "$OC/session/$sid/message?directory=$DIR_NAME" | jq '.info.id // .'
}

case "${1-all}" in
  routes)   smoke_routes ;;
  dynamo)   smoke_dynamo ;;
  opencode) smoke_opencode ;;
  swebench) smoke_swebench ;;
  all)      smoke_routes; smoke_dynamo; smoke_opencode; smoke_swebench ;;
  *) echo "usage: $0 [routes|dynamo|opencode|swebench|all]" >&2; exit 2 ;;
esac
