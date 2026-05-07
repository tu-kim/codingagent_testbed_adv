#!/usr/bin/env bash
# Lifecycle for the testbed: up / down / status / logs.
#
# Component glossary:
#   workers   = vllm.prefill_workers[] + vllm.decode_workers[] (python -m dynamo.vllm)
#   frontend  = python -m dynamo.frontend (OpenAI-compatible)
#   opencode  = bun run dev serve (OpenCode headless HTTP server)
#   nats      = single-node convenience (NATS event/request plane)
#   etcd      = single-node convenience (discovery)
#
# All PID files and component logs live under ./logs/ (relative to this script).

set -Eeuo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs"
CFG="$REPO_ROOT/deploy/testbed.yaml"
OPENCODE_DIR="$REPO_ROOT/opencode"
mkdir -p "$LOG_DIR"

# ---------- helpers ----------

# Run yq (kislyuk Python yq, jq-compatible filters) against testbed.yaml.
cfg_get() {
  yq -r "$1" "$CFG"
}

# CLI flag > TESTBED__SECTION__KEY env > yaml.
# Usage: cfg_get_env DYNAMO__ROUTER_MODE .dynamo.router_mode
cfg_get_env() {
  local var="TESTBED__$1"
  if [[ -n "${!var-}" ]]; then
    printf '%s\n' "${!var}"
  else
    cfg_get "$2"
  fi
}

pid_file() { printf '%s/%s.pid' "$LOG_DIR" "$1"; }
log_file() { printf '%s/%s.log' "$LOG_DIR" "$1"; }

# spawn <name> -- <cmd...>
# Optionally precede `--` with KEY=VALUE pairs to inject into the child env.
spawn() {
  local name="$1"; shift
  local -a envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    envs+=("$1"); shift
  done
  if [[ "${1-}" == "--" ]]; then
    shift
  else
    echo "spawn: missing -- before command for $name" >&2
    return 2
  fi
  if [[ -f "$(pid_file "$name")" ]] && kill -0 "$(cat "$(pid_file "$name")")" 2>/dev/null; then
    echo "$name: already running (pid $(cat "$(pid_file "$name")"))" >&2
    return 0
  fi
  local logp; logp="$(log_file "$name")"
  : > "$logp"
  setsid env "${envs[@]}" "$@" >>"$logp" 2>&1 &
  local pid=$!
  echo "$pid" > "$(pid_file "$name")"
  echo "$name: spawned pid=$pid log=$logp"
}

# Send TERM to the whole process group, then KILL after grace period.
kill_pgid() {
  local name="$1"
  local pf; pf="$(pid_file "$name")"
  [[ -f "$pf" ]] || { echo "$name: no pidfile"; return 0; }
  local pid; pid=$(cat "$pf")
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$pf"; echo "$name: not running"; return 0
  fi
  # PID == PGID because we used setsid above.
  kill -TERM "-$pid" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    sleep 0.5
  done
  if kill -0 "$pid" 2>/dev/null; then
    kill -KILL "-$pid" 2>/dev/null || true
  fi
  rm -f "$pf"
  echo "$name: stopped"
}

# Backstop: kill anything still on the given TCP port.
kill_port() {
  local port="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k -TERM "$port/tcp" 2>/dev/null || true
  fi
}

# ---------- worker arg construction ----------

# spawn one vllm worker. Args:
#   $1 role (prefill|decode), $2 entry json (compact), $3 rank
spawn_worker() {
  local role="$1" entry="$2" rank="$3"
  local name host gpus tp pp
  name=$(echo "$entry" | yq -r '.name')
  host=$(echo "$entry" | yq -r '.host // "127.0.0.1"')
  gpus=$(echo "$entry" | yq -r '.gpus')
  tp=$(echo "$entry" | yq -r '.tp')
  pp=$(echo "$entry" | yq -r '.pp')

  local gpu_count
  gpu_count=$(awk -F, '{print NF}' <<<"$gpus")
  if [[ "$gpu_count" != "$((tp * pp))" ]]; then
    echo "worker $name: gpus=$gpus has $gpu_count entries but tp*pp=$((tp * pp))" >&2
    return 2
  fi

  local model_name model_served
  model_name=$(cfg_get .model.name)
  model_served=$(cfg_get .model.served_name)

  local nixl_base
  nixl_base=$(cfg_get .vllm.nixl_port_base)

  local mml mnbt mns gmu kvdtype
  mml=$(cfg_get ".vllm.${role}.max_model_len")
  mnbt=$(cfg_get ".vllm.${role}.max_num_batched_tokens")
  mns=$(cfg_get ".vllm.${role}.max_num_seqs")
  gmu=$(cfg_get ".vllm.${role}.gpu_memory_utilization")
  kvdtype=$(cfg_get ".vllm.${role}.kv_cache_dtype")

  local extra_args
  extra_args=$(cfg_get '.vllm.extra_args // ""')

  # Role-specific kv-transfer-config JSON (kept as a single arg).
  local disagg_mode kv_role
  if [[ "$role" == "prefill" ]]; then
    disagg_mode=prefill; kv_role=kv_producer
  else
    disagg_mode=decode;  kv_role=kv_consumer
  fi
  local kv_cfg="{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"$kv_role\"}"

  local nats_url etcd_endpoints
  nats_url=$(cfg_get_env DYNAMO__NATS_URL .dynamo.nats_url)
  etcd_endpoints=$(cfg_get_env DYNAMO__ETCD_ENDPOINTS .dynamo.etcd_endpoints)

  local nixl_port=$((nixl_base + rank * 100))

  # shellcheck disable=SC2206  # word splitting is intended for extra_args
  local extra_array=($extra_args)

  spawn "vllm-${name}" \
    "CUDA_VISIBLE_DEVICES=$gpus" \
    "VLLM_NIXL_SIDE_CHANNEL_HOST=$host" \
    "VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port" \
    "NATS_SERVER=$nats_url" \
    "ETCD_ENDPOINTS=$etcd_endpoints" \
    -- \
    python -m dynamo.vllm \
      --model "$model_name" \
      --served-model-name "$model_served" \
      --tensor-parallel-size "$tp" \
      --pipeline-parallel-size "$pp" \
      --max-model-len "$mml" \
      --max-num-batched-tokens "$mnbt" \
      --max-num-seqs "$mns" \
      --gpu-memory-utilization "$gmu" \
      --kv-cache-dtype "$kvdtype" \
      --disaggregation-mode "$disagg_mode" \
      --kv-transfer-config "$kv_cfg" \
      "${extra_array[@]}"
}

# ---------- up verbs ----------

up_workers() {
  local rank=0
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    spawn_worker prefill "$entry" "$rank"
    rank=$((rank + 1))
  done < <(yq -c '.vllm.prefill_workers[]' "$CFG")
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    spawn_worker decode "$entry" "$rank"
    rank=$((rank + 1))
  done < <(yq -c '.vllm.decode_workers[]' "$CFG")
}

up_frontend() {
  local host port mode db rp ep nats_url etcd_endpoints
  host=$(cfg_get_env DYNAMO__HOST .dynamo.host)
  port=$(cfg_get_env DYNAMO__PORT .dynamo.port)
  mode=$(cfg_get_env DYNAMO__ROUTER_MODE .dynamo.router_mode)
  db=$(cfg_get_env DYNAMO__DISCOVERY_BACKEND .dynamo.discovery_backend)
  rp=$(cfg_get_env DYNAMO__REQUEST_PLANE .dynamo.request_plane)
  ep=$(cfg_get_env DYNAMO__EVENT_PLANE .dynamo.event_plane)
  nats_url=$(cfg_get_env DYNAMO__NATS_URL .dynamo.nats_url)
  etcd_endpoints=$(cfg_get_env DYNAMO__ETCD_ENDPOINTS .dynamo.etcd_endpoints)

  spawn frontend \
    "NATS_SERVER=$nats_url" \
    "ETCD_ENDPOINTS=$etcd_endpoints" \
    -- \
    python -m dynamo.frontend \
      --http-host "$host" \
      --http-port "$port" \
      --router-mode "$mode" \
      --discovery-backend "$db" \
      --request-plane "$rp" \
      --event-plane "$ep"
}

render_opencode_config() {
  local oc_cfg="$OPENCODE_DIR/opencode.json"
  local tmpl="$REPO_ROOT/deploy/opencode.json.tmpl"
  local dynamo_host dynamo_port served name provider_id
  dynamo_host=$(cfg_get_env DYNAMO__HOST .dynamo.host)
  dynamo_port=$(cfg_get_env DYNAMO__PORT .dynamo.port)
  served=$(cfg_get .model.served_name)
  name=$(cfg_get .model.name)
  provider_id=dynamo

  sed \
    -e "s|{{DYNAMO_BASE_URL}}|http://${dynamo_host}:${dynamo_port}/v1|g" \
    -e "s|{{MODEL_SERVED_NAME}}|${served}|g" \
    -e "s|{{MODEL_NAME}}|${name}|g" \
    -e "s|{{PROVIDER_ID}}|${provider_id}|g" \
    "$tmpl" > "$oc_cfg"
  echo "rendered $oc_cfg"
}

up_opencode() {
  render_opencode_config
  local host port experimental
  host=$(cfg_get_env OPENCODE__HOST .opencode.host)
  port=$(cfg_get_env OPENCODE__PORT .opencode.port)
  experimental=$(cfg_get_env OPENCODE__EXPERIMENTAL_WORKSPACES .opencode.experimental_workspaces)

  local exp_env="false"
  [[ "$experimental" == "true" ]] && exp_env="true"

  pushd "$OPENCODE_DIR" >/dev/null
  spawn opencode \
    "OPENCODE_EXPERIMENTAL_WORKSPACES=$exp_env" \
    -- \
    bun run dev serve --hostname "$host" --port "$port"
  popd >/dev/null
}

up_nats() {
  if ! command -v nats-server >/dev/null 2>&1; then
    echo "up nats: nats-server not in PATH (install or use external NATS)" >&2
    return 1
  fi
  spawn nats -- nats-server --addr 127.0.0.1 --port 4222
}

up_etcd() {
  if ! command -v etcd >/dev/null 2>&1; then
    echo "up etcd: etcd not in PATH (install or use external etcd)" >&2
    return 1
  fi
  spawn etcd -- etcd \
    --listen-client-urls http://127.0.0.1:2379 \
    --advertise-client-urls http://127.0.0.1:2379 \
    --data-dir "$LOG_DIR/etcd-data"
}

up_all() {
  up_workers
  up_frontend
  up_opencode
}

# ---------- down verbs ----------

down_one() {
  local name="$1"
  case "$name" in
    opencode)
      kill_pgid opencode
      kill_port "$(cfg_get_env OPENCODE__PORT .opencode.port)"
      ;;
    frontend)
      kill_pgid frontend
      kill_port "$(cfg_get_env DYNAMO__PORT .dynamo.port)"
      ;;
    workers)
      # Find any pidfile prefixed with vllm-.
      shopt -s nullglob
      for pf in "$LOG_DIR"/vllm-*.pid; do
        local n; n="$(basename "${pf%.pid}")"
        kill_pgid "$n"
      done
      shopt -u nullglob
      ;;
    nats|etcd)
      kill_pgid "$name"
      ;;
    *) echo "down: unknown component $name" >&2; return 2 ;;
  esac
}

down_all() {
  down_one opencode
  down_one frontend
  down_one workers
}

# ---------- status / logs ----------

status() {
  shopt -s nullglob
  for pf in "$LOG_DIR"/*.pid; do
    local n; n="$(basename "${pf%.pid}")"
    local pid; pid="$(cat "$pf")"
    if kill -0 "$pid" 2>/dev/null; then
      printf '%-20s pid=%-7s running=yes\n' "$n" "$pid"
    else
      printf '%-20s pid=%-7s running=no\n' "$n" "$pid"
    fi
  done
  shopt -u nullglob
}

logs() {
  local n="$1"
  local lf; lf="$(log_file "$n")"
  if [[ ! -f "$lf" ]]; then
    echo "no log file at $lf" >&2; return 1
  fi
  exec tail -F "$lf"
}

# ---------- dispatch ----------

usage() {
  cat <<USAGE
usage: $0 <verb> [target]

  up [nats|etcd|workers|frontend|opencode|all]   default: all (workers + frontend + opencode)
  down [nats|etcd|workers|frontend|opencode|all] default: all
  status
  logs <component>
USAGE
}

verb="${1-}"; shift || true
case "$verb" in
  up)
    target="${1:-all}"
    case "$target" in
      nats)     up_nats ;;
      etcd)     up_etcd ;;
      workers)  up_workers ;;
      frontend) up_frontend ;;
      opencode) up_opencode ;;
      all)      up_all ;;
      *) echo "up: unknown target $target" >&2; usage; exit 2 ;;
    esac
    ;;
  down)
    target="${1:-all}"
    case "$target" in
      all) down_all ;;
      *)   down_one "$target" ;;
    esac
    ;;
  status) status ;;
  logs)
    [[ $# -ge 1 ]] || { usage; exit 2; }
    logs "$1"
    ;;
  ""|-h|--help|help) usage ;;
  *) echo "unknown verb: $verb" >&2; usage; exit 2 ;;
esac
