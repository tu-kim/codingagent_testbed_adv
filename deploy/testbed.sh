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

# Tri-state bool toggle: env (TESTBED__$1) > yaml ($2), normalized to
# canonical lowercase true/false. config.py parses env values through
# yaml.safe_load (YAML 1.1 booleans: True/yes/on/y, False/no/off/n) and
# pydantic coerces 1/0 -> bool, so a user could set e.g.
# TESTBED__VLLM__ENFORCE_EAGER=True and have config.json record `true`.
# Without this normalization the caller's lowercase-only `case` arms would
# miss that spelling -> no flag passed -> config.json/worker drift (the
# exact bug this whole helper-conversion fixes). Anything else (null/"")
# passes through and stays unmatched downstream -> flag omitted.
# Usage: case "$(cfg_bool VLLM__ENFORCE_EAGER '.vllm.enforce_eager // ""')" in ...
cfg_bool() {
  local v; v=$(cfg_get_env "$1" "$2")
  case "${v,,}" in
    y|yes|on|true|1)   printf 'true\n' ;;
    n|no|off|false|0)  printf 'false\n' ;;
    *)                 printf '%s\n' "$v" ;;
  esac
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
  local name host gpus tp pp dp ep
  name=$(echo "$entry" | yq -r '.name')
  host=$(echo "$entry" | yq -r '.host // "127.0.0.1"')
  gpus=$(echo "$entry" | yq -r '.gpus')
  tp=$(echo "$entry" | yq -r '.tp')
  pp=$(echo "$entry" | yq -r '.pp')
  dp=$(echo "$entry" | yq -r '.dp // 1')
  ep=$(echo "$entry" | yq -r '.ep // false')

  # dp feeds arithmetic ($((tp*pp*dp))); a non-numeric value (e.g. dp: auto,
  # or a yaml typo) would be dereferenced as a varname and abort under
  # `set -u` before any clear message. Validate to a clean return 2 instead.
  if [[ ! "$dp" =~ ^[0-9]+$ ]] || [[ "$dp" -lt 1 ]]; then
    echo "worker $name: dp=$dp is not a positive integer" >&2
    return 2
  fi

  local gpu_count
  gpu_count=$(awk -F, '{print NF}' <<<"$gpus")
  if [[ "$gpu_count" != "$((tp * pp * dp))" ]]; then
    echo "worker $name: gpus=$gpus has $gpu_count entries but tp*pp*dp=$((tp * pp * dp))" >&2
    return 2
  fi

  # Parallelism flags forwarded to AsyncEngineArgs.add_cli_args
  # (dynamo/components/src/dynamo/vllm/args.py:93). --data-parallel-size is
  # only passed when dp>1 to keep the single-rank path byte-identical and
  # avoid perturbing dynamo's DP-range handling (llm_engine.py:282-283).
  # --enable-expert-parallel shards MoE experts across the tp*dp ranks
  # (e.g. MiniMax M2); vLLM derives EP size, so it's a bare toggle.
  local -a dp_flag=()
  if [[ "$dp" -gt 1 ]]; then
    dp_flag=(--data-parallel-size "$dp")
  fi
  local -a ep_flag=()
  case "$ep" in
    true) ep_flag=(--enable-expert-parallel) ;;
  esac

  # All worker knobs read via cfg_get_env (env > yaml) so a TESTBED__* env
  # override is honored on the SHELL side too -- config.py honors it on the
  # Python side, so plain cfg_get here would let a run's config.json record
  # the env value while the worker launched with the yaml value (silent
  # provenance drift). Role-nested fields use the ${role^^} segment to match
  # config.py's TESTBED__VLLM__PREFILL__<KEY> / __DECODE__<KEY> convention.
  local model_name model_served
  model_name=$(cfg_get_env MODEL__NAME .model.name)
  model_served=$(cfg_get_env MODEL__SERVED_NAME .model.served_name)

  # nixl_port_base is a DISAGG-only knob (feeds the NIXL side-channel port);
  # agg workers never use it, so a pure-agg yaml may omit it entirely. Read
  # with a "" fallback and validate only for prefill/decode roles.
  local nixl_base
  nixl_base=$(cfg_get_env VLLM__NIXL_PORT_BASE '.vllm.nixl_port_base // ""')
  # nixl_base feeds arithmetic ($((nixl_base + rank*100))); a non-numeric value
  # (yaml omission -> "", or a TESTBED__VLLM__NIXL_PORT_BASE typo) would be
  # dereferenced as a varname and abort under set -u. Guard like dp above.
  if [[ "$role" != "agg" && ! "$nixl_base" =~ ^[0-9]+$ ]]; then
    echo "worker $name: nixl_port_base=$nixl_base is not a non-negative integer" >&2
    return 2
  fi

  local role_uc="${role^^}"
  local mml mnbt mns gmu kvdtype
  mml=$(cfg_get_env "VLLM__${role_uc}__MAX_MODEL_LEN" ".vllm.${role}.max_model_len")
  mnbt=$(cfg_get_env "VLLM__${role_uc}__MAX_NUM_BATCHED_TOKENS" ".vllm.${role}.max_num_batched_tokens")
  mns=$(cfg_get_env "VLLM__${role_uc}__MAX_NUM_SEQS" ".vllm.${role}.max_num_seqs")
  gmu=$(cfg_get_env "VLLM__${role_uc}__GPU_MEMORY_UTILIZATION" ".vllm.${role}.gpu_memory_utilization")
  kvdtype=$(cfg_get_env "VLLM__${role_uc}__KV_CACHE_DTYPE" ".vllm.${role}.kv_cache_dtype")
  # Fail fast if the role section is missing/incomplete (yq yields "null"):
  # otherwise vLLM would be launched with literal `--max-model-len null`.
  # config.py validates this too, but testbed.sh reads the yaml directly.
  local knob
  for knob in "max_model_len=$mml" "max_num_batched_tokens=$mnbt" \
              "max_num_seqs=$mns" "gpu_memory_utilization=$gmu" \
              "kv_cache_dtype=$kvdtype"; do
    if [[ "${knob#*=}" == "null" || -z "${knob#*=}" ]]; then
      echo "worker $name: vllm.${role}.${knob%%=*} is missing (yaml lacks the" \
           "'${role}:' role section or the key) -- refusing to launch" >&2
      return 2
    fi
  done

  local extra_args
  extra_args=$(cfg_get_env VLLM__EXTRA_ARGS '.vllm.extra_args // ""')

  # Role-specific disaggregation flags.
  #   prefill/decode: --disaggregation-mode + NixlConnector kv-transfer-config
  #     (kv_producer/kv_consumer) + NIXL side-channel env.
  #   agg (PD colocation): --disaggregation-mode agg, NO --kv-transfer-config
  #     and NO NIXL env -- a single aggregated worker keeps KV in-engine
  #     across prefill+decode; dynamo only REQUIRES kv-transfer-config for
  #     prefill (dynamo/components/src/dynamo/vllm/args.py:213-223), and
  #     passing NixlConnector without a transfer peer just stands up dead
  #     transfer machinery.
  local disagg_mode kv_role=""
  local -a kv_args=()
  if [[ "$role" == "prefill" ]]; then
    disagg_mode=prefill; kv_role=kv_producer
  elif [[ "$role" == "decode" ]]; then
    disagg_mode=decode;  kv_role=kv_consumer
  else
    disagg_mode=agg
  fi
  if [[ -n "$kv_role" ]]; then
    # kv_connector was hardcoded; read it (env > yaml, default NixlConnector so
    # yaml omission keeps the old behavior) so config.py's vllm.kv_connector
    # field is actually honored shell-side instead of being a dead config knob.
    local kv_connector
    kv_connector=$(cfg_get_env VLLM__KV_CONNECTOR '.vllm.kv_connector // "NixlConnector"')
    kv_args=(--kv-transfer-config "{\"kv_connector\":\"$kv_connector\",\"kv_role\":\"$kv_role\"}")
  fi

  # --dyn-tool-call-parser applies to every NON-PREFILL worker: the branch in
  # dynamo/components/src/dynamo/vllm/main.py:722-724 is
  # `if worker_type != WorkerType.Prefill`, which covers decode AND
  # aggregated workers (both are the OpenAI surface that emits tool calls).
  # Empty value => skip the flag entirely.
  local -a tool_parser_args=()
  if [[ "$role" != "prefill" ]]; then
    local tool_parser
    tool_parser=$(cfg_get_env VLLM__TOOL_CALL_PARSER '.vllm.tool_call_parser // ""')
    if [[ -n "$tool_parser" ]]; then
      tool_parser_args=(--dyn-tool-call-parser "$tool_parser")
    fi
  fi

  # --dyn-reasoning-parser rides the same non-prefill branch (main.py:724).
  # Strips in-band reasoning spans (e.g. MiniMax M3 <mm:think>...</mm:think>)
  # so they don't reach the agent as content. Empty value => skip the flag.
  local -a reasoning_parser_args=()
  if [[ "$role" != "prefill" ]]; then
    local reasoning_parser
    reasoning_parser=$(cfg_get_env VLLM__REASONING_PARSER '.vllm.reasoning_parser // ""')
    if [[ -n "$reasoning_parser" ]]; then
      reasoning_parser_args=(--dyn-reasoning-parser "$reasoning_parser")
    fi
  fi

  local nats_url etcd_endpoints
  nats_url=$(cfg_get_env DYNAMO__NATS_URL .dynamo.nats_url)
  etcd_endpoints=$(cfg_get_env DYNAMO__ETCD_ENDPOINTS .dynamo.etcd_endpoints)

  # NIXL side-channel env: only meaningful when KV actually transfers
  # between workers (disagg). Omitted entirely for agg workers -- and the
  # port arithmetic stays inside the branch because a pure-agg yaml may
  # leave nixl_base empty ($(("" + ...)) aborts under set -u/-e).
  local -a nixl_env=()
  if [[ "$role" != "agg" ]]; then
    local nixl_port=$((nixl_base + rank * 100))
    nixl_env=(
      "VLLM_NIXL_SIDE_CHANNEL_HOST=$host"
      "VLLM_NIXL_SIDE_CHANNEL_PORT=$nixl_port"
    )
  fi

  # DYN_SYSTEM_PORT exposes per-worker Prometheus /metrics + /health
  # via dynamo's system status server (lib/runtime/src/system_status_server.rs).
  # Worker rank N → port = base + N (DCGM-style; collisions if base+N
  # overlaps another listener). Set system_port_base <= 0 to disable.
  local sys_port_base sys_port=-1
  sys_port_base=$(cfg_get_env VLLM__SYSTEM_PORT_BASE '.vllm.system_port_base // -1')
  # Feeds `-gt` and $((...)); a non-numeric env value would abort under set -u.
  # Allow a leading '-' so the -1 disable sentinel (and negatives) stay valid.
  if [[ ! "$sys_port_base" =~ ^-?[0-9]+$ ]]; then
    echo "worker $name: system_port_base=$sys_port_base is not an integer" >&2
    return 2
  fi
  if [[ "$sys_port_base" -gt 0 ]]; then
    sys_port=$((sys_port_base + rank))
  fi

  # vLLM tri-state toggles forwarded as paired CLI flags. Null/missing
  # = don't pass either; vLLM v1 default is enable_prefix_caching=True
  # (dynamo/components/src/dynamo/vllm/args.py:225-230).
  # `// ""` is the no-value sentinel: `null` and `""` are both unmatched by
  # the true/false case arms so the flag is omitted (vLLM default kept).
  # cfg_bool (env > yaml, NOT plain cfg_get) so a TESTBED__VLLM__* env
  # override is honored on the SHELL side too -- config.py already honors it
  # on the Python side, so without this the run's config.json (Python
  # snapshot) could record false while the workers actually launched with
  # vLLM's default True. cfg_bool also normalizes boolean spelling to match
  # config.py's yaml.safe_load parsing. Same precedence as the rest: env > yaml.
  local -a prefix_flag=()
  case "$(cfg_bool VLLM__ENABLE_PREFIX_CACHING '.vllm.enable_prefix_caching // ""')" in
    true)  prefix_flag=(--enable-prefix-caching) ;;
    false) prefix_flag=(--no-enable-prefix-caching) ;;
  esac
  local -a chunked_flag=()
  case "$(cfg_bool VLLM__ENABLE_CHUNKED_PREFILL '.vllm.enable_chunked_prefill // ""')" in
    true)  chunked_flag=(--enable-chunked-prefill) ;;
    false) chunked_flag=(--no-enable-chunked-prefill) ;;
  esac
  # Reproducibility knobs forwarded to AsyncEngineArgs.add_cli_args
  # (dynamo/components/src/dynamo/vllm/args.py:93).
  local seed
  seed=$(cfg_get_env VLLM__SEED '.vllm.seed // 42')
  local -a eager_flag=()
  case "$(cfg_bool VLLM__ENFORCE_EAGER '.vllm.enforce_eager // ""')" in
    true)  eager_flag=(--enforce-eager) ;;
    false) eager_flag=(--no-enforce-eager) ;;
  esac
  local -a dcar_flag=()
  case "$(cfg_bool VLLM__DISABLE_CUSTOM_ALL_REDUCE '.vllm.disable_custom_all_reduce // ""')" in
    true)  dcar_flag=(--disable-custom-all-reduce) ;;
    # vLLM's flag is `--disable-custom-all-reduce` (boolean store_true);
    # there's no `--no-` partner. Skip the flag to keep custom all-reduce
    # enabled (vLLM default).
  esac

  # --override-generation-config '<json>' merges into the model's
  # generation_config.json BEFORE per-request SamplingParams are built.
  # Neutralizes Qwen's `repetition_penalty: 1.05` (which would tilt logits
  # even under greedy decoding). yq -c emits compact single-line JSON; the
  # `// empty` fallback returns "" when the field is null or absent, so
  # older testbed.yaml files (or an explicit null) cleanly skip the flag.
  # Env override: deliberately bypasses cfg_get_env (which only knows
  # scalars). Pass the WHOLE dict as a JSON string via the env var --
  # this is whole-value replacement, NOT a per-key merge. (On the Python
  # side, _apply_env_overrides + _walk_set can express per-key overrides
  # like TESTBED__VLLM__OVERRIDE_GENERATION_CONFIG__REPETITION_PENALTY,
  # but those only merge cleanly when the yaml already provides the
  # dict; if yaml omits it, _walk_set creates a fresh dict containing
  # only the env-overridden key and pydantic's default_factory is
  # bypassed. So per-key overrides assume the yaml-supplied baseline.)
  # `{}` is NOT filtered -- it's a valid "merge nothing extra" signal
  # that matches Python's pass-through semantics; vLLM treats it as a
  # no-op flag. Only `null` / empty string skip the flag entirely.
  local oge_json
  local -a oge_flag=()
  if [[ -n "${TESTBED__VLLM__OVERRIDE_GENERATION_CONFIG-}" ]]; then
    oge_json="$TESTBED__VLLM__OVERRIDE_GENERATION_CONFIG"
  else
    oge_json=$(yq -c '.vllm.override_generation_config // empty' "$CFG")
  fi
  if [[ -n "$oge_json" && "$oge_json" != "null" ]]; then
    oge_flag=(--override-generation-config "$oge_json")
  fi

  # shellcheck disable=SC2206  # word splitting is intended for extra_args
  local extra_array=($extra_args)

  local -a sys_env=()
  if [[ "$sys_port" -gt 0 ]]; then
    # Bind to 0.0.0.0 so scrape_vllm_metrics.py can hit it from the
    # host running the testbed driver (typically same node anyway).
    sys_env=("DYN_SYSTEM_HOST=0.0.0.0" "DYN_SYSTEM_PORT=$sys_port")
  fi

  # ENV-gated engine-prompt capture (deploy/patches/dynamo-prompt-dump.patch).
  # When DYN_PROMPT_DUMP is truthy, each worker dumps the EXACT prompt it
  # hands to the vLLM engine -- the chat-template-applied token_ids the
  # frontend produced, detokenized back to text -- as NDJSON to
  # <workspace_root>/prompts/prompt-<pid>.jsonl, one record per request.
  # This is the prompt as the ENGINE sees it, distinct from OPENCODE_PROFILE
  # (which snapshots OpenCode's pre-template wire messages). Forcing a single
  # shared dir collects every prefill+decode worker's file in one place.
  # NOTE: detokenizing every prompt adds hot-path latency -- do NOT combine
  # with timing-sensitive profile/scheduling runs.
  local -a prompt_dump_envs=()
  local prompt_dump_enabled="${DYN_PROMPT_DUMP:-}"
  if [[ -n "$prompt_dump_enabled" && "$prompt_dump_enabled" != "0" && "$prompt_dump_enabled" != "false" ]]; then
    local ws_root prompt_dir
    # plain cfg_get (NOT cfg_get_env): config.py's TESTBED__ env regex requires
    # >=2 __-separated segments, so TESTBED__WORKSPACE_ROOT is NOT honored
    # Python-side (runner.py clones + records config.json from the yaml value).
    # Honoring it shell-side would split prompt dirs from where runner put the
    # session workspaces -- the inverse of the drift this sweep fixes.
    ws_root=$(cfg_get .workspace_root)
    prompt_dir="${DYN_PROMPT_DUMP_DIR:-${ws_root}/prompts}"
    mkdir -p "$prompt_dir"
    prompt_dump_envs+=(
      "DYN_PROMPT_DUMP=$prompt_dump_enabled"
      "DYN_PROMPT_DUMP_DIR=$prompt_dir"
    )
    [[ -n "${DYN_PROMPT_DUMP_TEXT:-}" ]] && prompt_dump_envs+=("DYN_PROMPT_DUMP_TEXT=$DYN_PROMPT_DUMP_TEXT")
    [[ -n "${DYN_PROMPT_DUMP_TOKENS:-}" ]] && prompt_dump_envs+=("DYN_PROMPT_DUMP_TOKENS=$DYN_PROMPT_DUMP_TOKENS")
    echo "vllm-${name}: engine-prompt dump enabled, DYN_PROMPT_DUMP_DIR=$prompt_dir"
  fi

  # `${arr[@]+"${arr[@]}"}` guards against unbound-array errors when
  # the array is empty under `set -u` (bash <4.4 quirk; harmless on
  # newer bash but kept for consistency with the rest of the script).
  spawn "vllm-${name}" \
    "CUDA_VISIBLE_DEVICES=$gpus" \
    ${nixl_env[@]+"${nixl_env[@]}"} \
    "NATS_SERVER=$nats_url" \
    "ETCD_ENDPOINTS=$etcd_endpoints" \
    ${sys_env[@]+"${sys_env[@]}"} \
    ${prompt_dump_envs[@]+"${prompt_dump_envs[@]}"} \
    -- \
    python -m dynamo.vllm \
      --model "$model_name" \
      --served-model-name "$model_served" \
      --tensor-parallel-size "$tp" \
      --pipeline-parallel-size "$pp" \
      ${dp_flag[@]+"${dp_flag[@]}"} \
      ${ep_flag[@]+"${ep_flag[@]}"} \
      --max-model-len "$mml" \
      --max-num-batched-tokens "$mnbt" \
      --max-num-seqs "$mns" \
      --gpu-memory-utilization "$gmu" \
      --kv-cache-dtype "$kvdtype" \
      --disaggregation-mode "$disagg_mode" \
      ${kv_args[@]+"${kv_args[@]}"} \
      --seed "$seed" \
      ${prefix_flag[@]+"${prefix_flag[@]}"} \
      ${chunked_flag[@]+"${chunked_flag[@]}"} \
      ${eager_flag[@]+"${eager_flag[@]}"} \
      ${dcar_flag[@]+"${dcar_flag[@]}"} \
      ${oge_flag[@]+"${oge_flag[@]}"} \
      ${tool_parser_args[@]+"${tool_parser_args[@]}"} \
      ${reasoning_parser_args[@]+"${reasoning_parser_args[@]}"} \
      ${extra_array[@]+"${extra_array[@]}"}
}

# ---------- up verbs ----------

up_workers() {
  # `// []` so a topology's absent list (pure-agg yaml omits
  # prefill_workers/decode_workers; pure-disagg omits agg_workers) iterates
  # zero times instead of yq erroring on null. Spawn order prefill ->
  # decode -> agg defines the global rank (NIXL/system port math) and MUST
  # match scrape_vllm_metrics.py:load_workers.
  local rank=0
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    spawn_worker prefill "$entry" "$rank"
    rank=$((rank + 1))
  done < <(yq -c '.vllm.prefill_workers // [] | .[]' "$CFG")
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    spawn_worker decode "$entry" "$rank"
    rank=$((rank + 1))
  done < <(yq -c '.vllm.decode_workers // [] | .[]' "$CFG")
  while IFS= read -r entry; do
    [[ -z "$entry" ]] && continue
    spawn_worker agg "$entry" "$rank"
    rank=$((rank + 1))
  done < <(yq -c '.vllm.agg_workers // [] | .[]' "$CFG")
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
  local dynamo_host dynamo_port served name provider_id temperature top_p
  dynamo_host=$(cfg_get_env DYNAMO__HOST .dynamo.host)
  dynamo_port=$(cfg_get_env DYNAMO__PORT .dynamo.port)
  served=$(cfg_get_env MODEL__SERVED_NAME .model.served_name)
  name=$(cfg_get_env MODEL__NAME .model.name)
  # Sampling overrides applied to every primary agent so reproducible
  # runs don't get the provider-transform default (qwen → 0.55).
  # `// 0.0` / `// 1.0` fallbacks mirror the `// ""` pattern at
  # lines 133/149: if the user's local testbed.yaml predates these
  # fields, yq would otherwise emit literal "null" and the rendered
  # opencode.json becomes invalid (`"temperature": null`).
  temperature=$(cfg_get_env MODEL__TEMPERATURE '.model.temperature // 0.0')
  top_p=$(cfg_get_env MODEL__TOP_P '.model.top_p // 1.0')
  local model_seed
  model_seed=$(cfg_get_env MODEL__SEED '.model.seed // 42')
  provider_id=dynamo

  sed \
    -e "s|{{DYNAMO_BASE_URL}}|http://${dynamo_host}:${dynamo_port}/v1|g" \
    -e "s|{{MODEL_SERVED_NAME}}|${served}|g" \
    -e "s|{{MODEL_NAME}}|${name}|g" \
    -e "s|{{PROVIDER_ID}}|${provider_id}|g" \
    -e "s|{{TEMPERATURE}}|${temperature}|g" \
    -e "s|{{TOP_P}}|${top_p}|g" \
    -e "s|{{SEED}}|${model_seed}|g" \
    "$tmpl" > "$oc_cfg"
  echo "rendered $oc_cfg"
}

up_opencode() {
  render_opencode_config
  local host port experimental oc_cfg
  host=$(cfg_get_env OPENCODE__HOST .opencode.host)
  port=$(cfg_get_env OPENCODE__PORT .opencode.port)
  experimental=$(cfg_get_env OPENCODE__EXPERIMENTAL_WORKSPACES .opencode.experimental_workspaces)
  # Match render_opencode_config's output path (kept stable: rendered file is
  # already in .gitignore at opencode/opencode.json).
  oc_cfg="$OPENCODE_DIR/opencode.json"

  local exp_env="false"
  [[ "$experimental" == "true" ]] && exp_env="true"

  # ENV-gated profiling (opencode/packages/opencode/src/profile/profile.ts).
  # When OPENCODE_PROFILE is set to a truthy value, force OPENCODE_PROFILE_DIR
  # to <workspace_root>/profiles so per-session NDJSON files from every
  # concurrent task land in one flat directory and can be aggregated with
  # scripts/aggregate_profiles.sh.
  local -a profile_envs=()
  local profile_enabled="${OPENCODE_PROFILE:-}"
  if [[ -n "$profile_enabled" && "$profile_enabled" != "0" && "$profile_enabled" != "false" ]]; then
    local workspace_root profile_dir
    # plain cfg_get (NOT cfg_get_env): see the spawn_worker prompt-dump note --
    # TESTBED__WORKSPACE_ROOT is not honored by config.py, so the shell must
    # read the same yaml value runner.py uses or profiles split from sessions.
    workspace_root=$(cfg_get .workspace_root)
    profile_dir="${OPENCODE_PROFILE_DIR:-${workspace_root}/profiles}"
    mkdir -p "$profile_dir"
    profile_envs+=(
      "OPENCODE_PROFILE=$profile_enabled"
      "OPENCODE_PROFILE_DIR=$profile_dir"
    )
    if [[ -n "${OPENCODE_PROFILE_MESSAGES:-}" ]]; then
      profile_envs+=("OPENCODE_PROFILE_MESSAGES=$OPENCODE_PROFILE_MESSAGES")
    fi
    echo "opencode: profiling enabled, OPENCODE_PROFILE_DIR=$profile_dir"
  fi

  # OPENCODE_CONFIG forces OpenCode to load this exact file regardless of
  # launch CWD or per-request ?directory= (config.ts:563-566). Without it,
  # OpenCode walks up from ?directory=<workspace> looking for opencode.json
  # (paths.ts:10-21), never reaching our rendered file -- which causes the
  # provider/model config to be ignored and HF (auto-enabled by HF_TOKEN) to
  # take over, routing inference to router.huggingface.co.
  #
  # OPENCODE_CLIENT=server disables the interactive `question` tool. It
  # defaults to "cli" (core/flag/flag.ts), and registry.ts:193-194 exposes
  # `question` whenever OPENCODE_CLIENT is one of app|cli|desktop. In a
  # headless run nobody can answer: the model's question.ask() blocks on a
  # Deferred (question/index.ts:174) until POST /question/:id/reply arrives,
  # which the runner never sends -- so the task hangs until --task-timeout-s.
  # "server" is outside the app|cli|desktop set, so the tool is not offered.
  pushd "$OPENCODE_DIR" >/dev/null
  # ${profile_envs[@]+"${profile_envs[@]}"} expands to nothing when the array
  # is empty -- needed because `set -u` is on and unguarded "${arr[@]}" errors
  # on older bash (macOS 3.2 / pre-4.4) when the array has zero elements.
  spawn opencode \
    "OPENCODE_EXPERIMENTAL_WORKSPACES=$exp_env" \
    "OPENCODE_CONFIG=$oc_cfg" \
    "OPENCODE_CLIENT=server" \
    ${profile_envs[@]+"${profile_envs[@]}"} \
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

up_monitor() {
  local interval dcgm_update_freq output pids_from py bindings_path
  interval=$(cfg_get_env MONITOR__INTERVAL_S '.monitor.interval_s // 1.0')
  dcgm_update_freq=$(cfg_get_env MONITOR__DCGM_UPDATE_FREQ_S '.monitor.dcgm_update_freq_s // 0.1')
  output=$(cfg_get_env MONITOR__OUTPUT '.monitor.output // "logs/resource.ndjson"')
  pids_from=$(cfg_get_env MONITOR__PIDS_FROM '.monitor.pids_from // "logs/"')
  # Resolve relative paths against REPO_ROOT so a `monitor:` field
  # like `logs/resource.ndjson` doesn't fly off into the user's CWD.
  [[ "$output" = /* ]] || output="$REPO_ROOT/$output"
  [[ "$pids_from" = /* ]] || pids_from="$REPO_ROOT/$pids_from"
  # dcgm_py + dcgm_bindings_path read from yaml (not $DCGM_PY env) so
  # `sudo testbed.sh up monitor` works without sudo's env-keep dance.
  py=$(cfg_get_env MONITOR__DCGM_PY '.monitor.dcgm_py // ""')
  if [[ -z "$py" || "$py" == "null" ]]; then
    echo "up monitor: monitor.dcgm_py is empty in testbed.yaml" >&2
    echo "  Set it to the Python interpreter that has DCGM bindings" >&2
    echo "  installed (or that can find them via monitor.dcgm_bindings_path)." >&2
    return 1
  fi
  if [[ ! -f "$REPO_ROOT/scripts/monitor_resources.py" ]]; then
    echo "up monitor: $REPO_ROOT/scripts/monitor_resources.py is missing" >&2
    return 1
  fi
  bindings_path=$(cfg_get_env MONITOR__DCGM_BINDINGS_PATH '.monitor.dcgm_bindings_path // ""')
  local dcgm_env=()
  if [[ -n "$bindings_path" && "$bindings_path" != "null" ]]; then
    dcgm_env+=("DCGM_BINDINGS_PATH=$bindings_path")
  fi
  spawn monitor ${dcgm_env[@]+"${dcgm_env[@]}"} -- "$py" "$REPO_ROOT/scripts/monitor_resources.py" \
    --output "$output" \
    --interval "$interval" \
    --dcgm-update-freq "$dcgm_update_freq" \
    --pids-from "$pids_from"
}

up_scrape_metrics() {
  local interval output py metric_names
  # Env names mirror the yaml path so TESTBED__MONITOR__SCRAPE_INTERVAL_S
  # is consistent with TESTBED__MONITOR__INTERVAL_S / __OUTPUT (same
  # section prefix, key in snake_case).
  interval=$(cfg_get_env MONITOR__SCRAPE_INTERVAL_S '.monitor.scrape_interval_s // 1.0')
  output=$(cfg_get_env MONITOR__SCRAPE_OUTPUT '.monitor.scrape_output // "logs/vllm_metrics.ndjson"')
  [[ "$output" = /* ]] || output="$REPO_ROOT/$output"
  py="${PYTHON:-python3}"
  if ! command -v "$py" >/dev/null 2>&1; then
    echo "up scrape_metrics: python interpreter '$py' not in PATH" >&2
    return 1
  fi
  if [[ ! -f "$REPO_ROOT/scripts/scrape_vllm_metrics.py" ]]; then
    echo "up scrape_metrics: $REPO_ROOT/scripts/scrape_vllm_metrics.py is missing" >&2
    return 1
  fi
  # monitor.vllm_metric_names: list -> comma-joined for the --metric-names
  # CLI arg. null/absent -> empty string -> script falls back to its
  # DEFAULT_METRIC_NAMES allowlist. `// []` ensures join doesn't error
  # on the null case.
  metric_names=$(yq -r '(.monitor.vllm_metric_names // []) | join(",")' "$CFG")
  local -a metric_args=()
  if [[ -n "$metric_names" ]]; then
    metric_args=(--metric-names "$metric_names")
  fi
  spawn scrape_metrics -- "$py" "$REPO_ROOT/scripts/scrape_vllm_metrics.py" \
    --testbed-yaml "$CFG" \
    --output "$output" \
    --interval "$interval" \
    ${metric_args[@]+"${metric_args[@]}"}
}

up_all() {
  # Core inference stack only. `monitor` (DCGM/psutil sampler) and
  # `scrape_metrics` (vLLM /metrics poller) are opt-in -- bring them
  # up explicitly with `testbed.sh up monitor` / `up scrape_metrics`
  # before a run you want resource data for.
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
    nats|etcd|monitor|scrape_metrics)
      kill_pgid "$name"
      ;;
    *) echo "down: unknown component $name" >&2; return 2 ;;
  esac
}

down_all() {
  # monitor and scrape_metrics are excluded: monitor runs under sudo and is
  # brought up/down separately. Use `down monitor` / `down scrape_metrics`.
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

  up [nats|etcd|workers|frontend|opencode|monitor|scrape_metrics|all]   default: all (= workers + frontend + opencode; monitor/scrape_metrics are opt-in, bring up separately)
  down [nats|etcd|workers|frontend|opencode|monitor|scrape_metrics|all] default: all (= opencode + frontend + workers; monitor/scrape_metrics excluded — stop them separately)
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
      monitor)  up_monitor ;;
      scrape_metrics) up_scrape_metrics ;;
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
