# CLAUDE.md

Reference for future Claude Code sessions on this repo. Keep edits in sync with reality — readers trust this over the README.

## What this is

Testbed that drives **SWE-bench** problems through an **OpenCode** agent server pointed at a **NVIDIA Dynamo** OpenAI-compatible frontend whose backend is a **vLLM PD-disaggregated** worker pool. The goal is to measure router/scheduling decisions under realistic coding-agent workloads.

```
SWE-bench sample
   └─ runner.py (Poisson)──┐
       │ git clone+checkout │   # pre-clone repo@base_commit into <workspace_root>/<session_id>
       ▼                     │   # before any OpenCode call (avoids agent-loop hangs)
                             ▼
                 OpenCode server (:4096, headless HTTP)
                  OPENCODE_EXPERIMENTAL_WORKSPACES=true
                  POST /session, POST /session/:id/message, GET /session/:id/message
                  (per-request workspace via ?directory=)
                           │
                           ▼ (OpenAI Chat Completions)
                 Dynamo frontend (:8000/v1)
                  --router-mode {round-robin|least-loaded|kv|...}
                  worker discovery via etcd (default); NATS is the request/event plane
                           │
                           ▼ (KV cache transfer via vLLM's NixlConnector)
                 vLLM workers
                   prefill (kv_producer) ──► decode (kv_consumer)
```

OpenCode runs as a **headless HTTP server** (not the TUI), launched from `opencode/` with `OPENCODE_EXPERIMENTAL_WORKSPACES=true bun dev serve`. Concurrent runner requests **must not share a working directory**, so every session/message call carries a unique `?directory=` and lives in its own subfolder under the shared `workspace_root`.

The runner pre-clones the SWE-bench repo at `base_commit` into that subfolder **before** opening the OpenCode session — letting the agent perform the clone during its loop has been observed to hang.

## Layout

```
<project-root>/
  README.md
  pyproject.toml

  opencode/                # vendored OpenCode source (built/run in-tree)
  dynamo/                  # vendored Dynamo source (built with vLLM backend)

  src/testbed/
    cli.py                 # click entrypoint: run / smoke
    runner.py              # Poisson workload driver; clones repo, calls OpenCode
    poisson.py             # arrival_offsets / arrivals (rate=qps, seed-deterministic)
    swebench.py            # load_samples(split, seed, n) + render_prompt(sample)
    opencode.py            # async HTTP client for /session, /session/:id/message, /event SSE
    config.py              # loads deploy/testbed.yaml + .env overrides; schema validation

  deploy/
    testbed.yaml.example   # committed template; copy to testbed.yaml (gitignored) and edit
    testbed.yaml           # SINGLE source of truth: workspace, model, vLLM PD, dynamo, opencode (gitignored)
    testbed.sh             # self-contained: spawn/kill/log/status all inline (no _lib.sh)
    opencode.json.tmpl     # rendered to opencode/opencode.json on launch

  scripts/
    curl_smoke.sh          # opencode | dynamo | routes | swebench | all — single-request smoke

  tests/                   # pytest, no network. Mocks only.
  logs/                    # PID + log files. Created at runtime by testbed.sh.
  .env.example             # secrets/overrides only (passwords, api keys)
```

## Module contracts

These are the shapes downstream code depends on. Pinned here so changes are deliberate.

**`opencode.py`** — async client. All methods take/return JSON-native types; no transformation of payloads.
```python
async def create_session(directory: str) -> str
    # POST /session?directory=<abs_dir> with body {}.
    # Returns the SERVER-ASSIGNED session id (regex ^ses.* per the SDK schema —
    # we don't pick the id; we use it as :sessionID in subsequent calls).

async def send_message(session_id: str, prompt: str, directory: str) -> dict
    # POST /session/:id/message?directory=<abs_dir> with body
    #   {"parts":[{"type":"text","text":prompt}]}.
    # Blocks until agent loop completes; returns the raw JSON envelope
    # ({info, parts}) — only the FINAL assistant message — see per-task flow.

async def abort_session(session_id: str, directory: str) -> bool
    # POST /session/:id/abort?directory=<abs_dir> with NO body.
    # Cancels the server-side agent loop (route handler runs svc.cancel and
    # returns a bare boolean). Returns that boolean. Used by the runner's
    # timeout path to kill the zombie loop — see "Concurrency, errors, cleanup".

async def list_messages(session_id: str, directory: str) -> list[dict]
    # GET /session/:id/message?directory=<abs_dir>; returns the full message
    # list as-is. This is the canonical source of intermediate tool-loop steps.

async def stream_events(directory: str) -> AsyncIterator[dict]
    # GET /event SSE. Exposed for debugging only; runner does NOT consume this.
```

`?directory=` is honored on every instance route by `InstanceMiddleware`
(`opencode/packages/opencode/src/server/routes/instance/middleware.ts`); it can
also be passed as the `x-opencode-directory` header. The middleware runs
`AppFileSystem.resolve(...)` on the value, which is Node's `path.resolve()` —
**a relative value is anchored on OpenCode's CWD, NOT on `workspace_root`**.
The runner therefore sends the **absolute path** of the pre-cloned dir;
sending just the subfolder name silently mis-anchors onto `opencode/<name>/`.

**Auth** (when `OPENCODE_SERVER_PASSWORD` is set on the server side):
HTTP Basic, with username defaulting to `"opencode"`
(`opencode/packages/opencode/src/server/auth.ts`). The client sends
`Authorization: Basic <base64(username:password)>`. There is **no**
`x-opencode-server-password` header — that name is from no version of OpenCode
and any code that sets it is a bug.

**`swebench.py:load_samples(split, seed, n)`** — sample selection is **fully deterministic** given `(split, seed, n)`:
1. Load all samples for the split.
2. Sort by `instance_id`.
3. `random.Random(seed).sample(sorted, n)` to pick `n`.

No "first N" mode. No reshuffling between runs with the same seed.

**`poisson.py`** — `arrival_offsets(rate: float, n: int, seed: int) -> list[float]` returns monotonic-increasing arrival times in seconds. Same `(rate, n, seed)` always yields identical offsets.

## Single config: `deploy/testbed.yaml`

All component settings live here. Loaded by `config.py` (Python side) and by `testbed.sh` via `yq` (shell side). `.env` carries **only secrets and developer-local overrides** — anything that would normally be committed belongs in `testbed.yaml`.

Default shape:

```yaml
# Shared by OpenCode (parent of session subdirs) and runner (clone target).
workspace_root: /tmp/testbed-workspaces

model:
  name: qwen3-coder-30b-a3b      # HF id or local path; default
  served_name: local              # OpenCode provider id used in opencode.json
  temperature: 0.0                # forwarded into opencode.json.tmpl's agent.* blocks
  top_p: 1.0                      # see Reproducibility note below

vllm:
  kv_connector: NixlConnector     # vLLM kv-transfer connector class name (matches what vLLM expects)
  nixl_port_base: 6000            # rank N → VLLM_NIXL_SIDE_CHANNEL_PORT = base + N*100
  tool_call_parser: qwen3_coder   # → --dyn-tool-call-parser on DECODE workers only ("" disables)
  reasoning_parser: ""            # → --dyn-reasoning-parser on DECODE workers only ("" disables); minimax_m3 for MiniMax M3
  override_generation_config:     # → --override-generation-config '<json>'; set null/omit to skip (see Conventions/gotchas)
    temperature: 0.0
    top_p: 1.0
    top_k: -1
    repetition_penalty: 1.0       # neutralizes Qwen's 1.05 baked into generation_config.json (logits tilt under greedy)

  # Each worker entry:
  #   host: where this worker is deployed. Default 127.0.0.1.
  #         Doubles as VLLM_NIXL_SIDE_CHANNEL_HOST for that worker.
  #         Single-node only at present; multi-node SSH spawn is TBD. Keep host=127.0.0.1 until then.
  #   gpus: comma-separated GPU ids. Must satisfy len(split(',')) == tp*pp*dp.
  #   dp:   --data-parallel-size (optional, default 1; emitted only when >1).
  #   ep:   --enable-expert-parallel (optional, default false). MoE expert
  #         sharding (e.g. MiniMax M2); EP size = tp*dp, so it's a bare toggle
  #         and does NOT add GPUs to the tp*pp*dp count.
  # Topology is EITHER disagg (prefill_workers + decode_workers + prefill/decode
  # sections) OR colocation (agg_workers + agg section). config.py rejects
  # mixing, an empty union, and a half-configured disagg pair (a lone decode
  # pool never becomes ready -- dynamo readiness gating needs a Prefill peer).
  prefill_workers:
    - { name: p0, host: 127.0.0.1, gpus: "0,1", tp: 2, pp: 1, dp: 1, ep: true }
  decode_workers:
    - { name: d0, host: 127.0.0.1, gpus: "2,3", tp: 2, pp: 1, dp: 1, ep: true }
  agg_workers: []                 # PD colocation (aggregated): one worker type does
                                  # prefill AND decode. Requires the `agg:` role
                                  # section; mutually exclusive with the two lists above.

  prefill:
    max_model_len: 32768
    max_num_batched_tokens: 8192
    max_num_seqs: 16
    gpu_memory_utilization: 0.90  # fraction of GPU mem used by engine; most goes to KV cache
    kv_cache_dtype: auto          # auto | fp8 | fp8_e4m3 | fp8_e5m2

  decode:
    max_model_len: 32768
    max_num_batched_tokens: 2048
    max_num_seqs: 64
    gpu_memory_utilization: 0.90
    kv_cache_dtype: auto

  kvbm:                           # dynamo KV Block Manager: host/disk KV tiering (AGG-ONLY; see Worker role injection)
    enabled: false
    cpu_cache_gb: 0               # host tier GB; REQUIRED > 0 when enabled
    disk_cache_gb: 0              # disk tier GB; 0 = no disk tier
    metrics_port_base: 6880       # per-worker KVBM /metrics = base + rank; <=0 disables
    leader_zmq_pub_port_base: 56001  # KVBM leader ZMQ; unique per worker (base + rank)
    leader_zmq_ack_port_base: 56101

  lmcache:                        # LMCache CPU/disk KV offloading (AGG-ONLY; mutually exclusive with kvbm; see Worker role injection)
    enabled: false
    chunk_size: 256               # LMCACHE_CHUNK_SIZE
    cpu_cache_gb: 0               # LMCACHE_MAX_LOCAL_CPU_SIZE GB; REQUIRED > 0 when enabled
    disk_cache_gb: 0              # LMCACHE_MAX_LOCAL_DISK_SIZE GB; 0 = no disk tier
    disk_path: /tmp/lmcache       # LMCACHE_LOCAL_DISK=file://<path>/ when disk_cache_gb > 0

  extra_args: ""                  # appended to every vLLM serve invocation

dynamo:
  host: 127.0.0.1                 # → --http-host on dynamo.frontend
  port: 8000                      # → --http-port on dynamo.frontend
  router_mode: round-robin        # round-robin | least-loaded | kv | random | power-of-two | direct | device-aware-weighted
  discovery_backend: etcd         # → --discovery-backend (kubernetes | etcd | file | mem)
  etcd_endpoints: http://127.0.0.1:2379   # → ETCD_ENDPOINTS env on every dynamo process
  nats_url: nats://127.0.0.1:4222         # → NATS_SERVER env (request/event plane, NOT discovery)
  request_plane: tcp              # → --request-plane (tcp | nats)
  event_plane: nats               # → --event-plane (nats | zmq)

opencode:
  host: 127.0.0.1
  port: 4096
  experimental_workspaces: true   # → OPENCODE_EXPERIMENTAL_WORKSPACES=true

monitor:                          # DCGM GPU + psutil CPU/process sampler (opt-in; not in `up all` / `down all`)
  dcgm_py: ""                     # REQUIRED for `up monitor`: path to Python with DCGM bindings (read from yaml so sudo doesn't strip it)
  dcgm_bindings_path: ""          # optional: dir with dcgm_fields.py if not pip-discoverable from dcgm_py
  interval_s: 1.0                 # NDJSON drain cadence (window length per row)
  dcgm_update_freq_s: 0.1         # DCGM internal sampling period; window aggregates ~interval/this samples per field
  output: logs/resource.ndjson    # NDJSON output (`ts` matches profile NDJSON)
  pids_from: logs/                # per-process tracking via *.pid files
  scrape_interval_s: 1.0          # vLLM /metrics scrape cadence (separate component)
  scrape_output: logs/vllm_metrics.ndjson
  vllm_metric_names: null         # null = script's DEFAULT_METRIC_NAMES (~13: KV-cache + queue + token + request_queue_time histogram = prefill/decode scheduling delay); list overrides
```

There is **no `runner:` section**. Runner-side defaults (`num_samples=10`, `qps=0.5`, `seed=42`) live in `cli.py`. CLI flag > env override > yaml default.

### Worker role injection (kv_role + disaggregation_mode + tool_call_parser + reasoning_parser + override_generation_config)

Each worker needs role-specific Dynamo/vLLM args. `testbed.sh` injects these at launch:
- `prefill_workers[]` → `--disaggregation-mode prefill` plus `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'`
- `decode_workers[]`  → `--disaggregation-mode decode` plus `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'` plus (if `vllm.tool_call_parser` is non-empty) `--dyn-tool-call-parser <name>` plus (if `vllm.reasoning_parser` is non-empty) `--dyn-reasoning-parser <name>`
- `agg_workers[]` (**PD colocation**) → `--disaggregation-mode agg` (a first-class `DisaggregationMode` enum value, `dynamo/components/src/dynamo/common/constants.py:9-15`; omitting the flag resolves to the same AGGREGATED default, `backend_args.py:378-379`), **NO `--kv-transfer-config`** (only required for prefill, `args.py:213-223`; KV stays in-engine across both phases) and **NO `VLLM_NIXL_SIDE_CHANNEL_*` env**. Parsers ARE applied (see below). Aggregated workers register as component `backend` with `needs=[]` (`args.py:183-185`, `worker_factory.py:563-569`) so they serve the moment they register — the frontend needs zero changes, and `router_mode: kv` still works because agg workers publish KV events (only DECODE disables them, `args.py:338-343`). Do NOT mix agg and prefill/decode pools in one namespace (same `backend` component, ambiguous readiness) — `config.py` rejects it. **EXCEPTION — `vllm.kvbm.enabled: true`**: KVBM (dynamo KV Block Manager, host/disk KV tiering) is selected by giving the agg worker a `--kv-transfer-config` naming connector class `DynamoConnector` (`{"kv_connector":"DynamoConnector","kv_connector_module_path":"kvbm.vllm_integration.connector","kv_role":"kv_both"}`, exact shape from `dynamo/examples/backends/vllm/launch/agg_kvbm.sh`) — this is the *offload connector*, not disagg transport, so the "agg has no kv-transfer-config" rule applies only to NixlConnector. Tier sizes + observability ride env (`DYN_KVBM_CPU_CACHE_GB`, `DYN_KVBM_DISK_CACHE_GB`, `DYN_KVBM_METRICS`/`_PORT` = `kvbm.metrics_port_base + rank`, `DYN_KVBM_LEADER_ZMQ_{PUB,ACK}_PORT` = base + rank — names from `dynamo/lib/runtime/src/config/environment_names.rs:180-269`). KVBM is **agg-only** in this testbed (`config.py` rejects it with disagg — the disagg shape needs PdConnector-nested connectors with `kv_role=kv_both`, a different role contract from our kv_producer/kv_consumer wiring; see `disagg_kvbm.sh`). PREREQ: the `kvbm` python extension (`dynamo/lib/bindings/kvbm`, separate CUDA maturin wheel — NOT built by our dynamo build steps) must be importable; missing wheel = consolidator warning (`main.py:593-599`) then vLLM connector failure. SIZING: `cpu_cache_gb` must meaningfully exceed the GPU KV pool or churn degrades perf (`docs/components/kvbm/kvbm-guide.md:272`). `scrape_vllm_metrics.py` auto-adds a `role: "kvbm"` scrape target per agg worker (port `metrics_port_base + rank`, own `KVBM_METRIC_NAMES` allowlist: `kvbm_host/disk_cache_hit_rate`, offload/onboard block counters). **SECOND EXCEPTION — `vllm.lmcache.enabled: true`** (mutually exclusive with kvbm; both claim the offload-connector slot): LMCache CPU/disk offloading via vLLM's NATIVE `LMCacheConnectorV1` (`--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'`, shape from `dynamo/tests/lmcache/deploy-lmcache_enabled-dynamo.sh:31`). Tier config rides `LMCACHE_*` env: `LMCACHE_CHUNK_SIZE`, `LMCACHE_LOCAL_CPU=True`, `LMCACHE_MAX_LOCAL_CPU_SIZE` (GB; names from the vendored disag script :37-39) + disk `LMCACHE_LOCAL_DISK=file://<disk_path>/`, `LMCACHE_MAX_LOCAL_DISK_SIZE` (GB; LMCache-documented, NOT vendored — verify against the installed lmcache on the GPU host). No kvbm wheel, no consolidator, no leader-ZMQ ports, and **no separate metrics endpoint**: LMCache registers `lmcache:`-prefixed Prometheus metrics on the worker's own registry (= the `DYN_SYSTEM_PORT` `/metrics` already scraped; `scrape_vllm_metrics.py` keeps them by `LMCACHE_METRIC_PREFIX` since exact names live in the external pip package). PREREQ: `pip install lmcache` importable on the worker host (vLLM's connector factory imports it lazily at engine init). Motivation: KVBM's DynamoConnector has a preemption stale-slot bug (reset ordering) that crashes the scheduler assert under KV pressure; LMCache's stateless per-call matching avoids that class (verify on first preemption-heavy run).
- All roles → `--data-parallel-size <dp>` (only when the worker's `dp > 1`) plus `--enable-expert-parallel` (when the worker's `ep: true`). These are per-worker parallelism knobs forwarded straight to `AsyncEngineArgs` (so roles can differ); EP shards MoE experts across the `tp*dp` ranks without changing the `tp*pp*dp` GPU count.
- All roles → `--override-generation-config '<json>'` (if `vllm.override_generation_config` is a non-null dict) so the model's `generation_config.json` defaults are merged with our reproducibility-pinned baseline (see the bullet in "Conventions / gotchas" for the rationale).

The flag schema and connector class name come from the vendored Dynamo (`dynamo/components/src/dynamo/vllm/{args,backend_args}.py`). The tool-call/reasoning-parser branch is **non-prefill, not decode-only**: `dynamo/components/src/dynamo/vllm/main.py:722-724` (`if worker_type != WorkerType.Prefill: runtime_config.tool_call_parser = ...; runtime_config.reasoning_parser = ...`) applies the parsers to decode AND aggregated workers — both are the OpenAI surface that emits tool calls; only prefill skips (applying one there is a no-op but a smell). `testbed.sh` mirrors this with a `role != prefill` guard. The **reasoning parser** (`vllm.reasoning_parser` → `--dyn-reasoning-parser`) rides the same branch and is wired identically to `tool_call_parser` — needed for models that emit in-band reasoning the frontend must strip (e.g. MiniMax M3's `<mm:think>…</mm:think>` → `reasoning_parser: minimax_m3`).

**Agg mode + testbed patches**: both `dynamo-scheduling-log.patch` and `dynamo-prompt-dump.patch` hook the `DecodeWorkerHandler` path that aggregated workers execute (`worker_factory.py:177-185` routes AGGREGATED through `_create_decode_worker`), so SCHED_DELAY lines and prompt dumps still fire in colocation — but always with `role=decode` (one line per request, no separate prefill record). The disagg-era guidance "filter prompt dumps to `role=prefill` for the canonical prompt" applies ONLY to disagg; in agg mode use the single `decode` record.

### Discovery vs. NATS

Two separate concerns:
- **Discovery** is how the frontend learns about workers. Default `--discovery-backend etcd`; choices: `kubernetes | etcd | file | mem`. Set `ETCD_ENDPOINTS` on every dynamo process. (CLAUDE.md previously said "discovery via NATS" — that was wrong; NATS does not appear in the discovery enum.)
- **Request/Event plane** is how requests + KV events flow once workers are discovered. NATS is the default event plane and an option for the request plane. Set `NATS_SERVER` on every dynamo process.

Both etcd and NATS are treated as **external prerequisites**. `testbed.sh up etcd` and `testbed.sh up nats` exist as single-node conveniences only.

## Lifecycle: one script, four verbs

`deploy/testbed.sh` is the only thing you need to remember. It is self-contained — no `_lib.sh`, no Makefile, no docker-compose.

```
deploy/testbed.sh up     [nats|etcd|workers|frontend|opencode|monitor|scrape_metrics|all]   # default: all (= workers + frontend + opencode; monitor/scrape_metrics are opt-in)
deploy/testbed.sh down   [nats|etcd|workers|frontend|opencode|monitor|scrape_metrics|all]   # default: all (= opencode + frontend + workers; monitor/scrape_metrics excluded)
deploy/testbed.sh status
deploy/testbed.sh logs   <component>
```

All PID files and component logs are written to **`./logs/`** (created relative to wherever `testbed.sh` is invoked). PGID-based teardown logic and port-kill backstops are inlined into the script.

`up workers` reads `vllm.prefill_workers` / `vllm.decode_workers` from `testbed.yaml`. For each worker, the script:

1. validates `len(split(gpus, ',')) == tp*pp*dp` (and that `dp` is a positive integer),
2. exports `VLLM_NIXL_SIDE_CHANNEL_HOST=<worker.host>`, `VLLM_NIXL_SIDE_CHANNEL_PORT=<nixl_port_base + rank*100>`, `CUDA_VISIBLE_DEVICES=<gpus>`,
3. injects role-specific Dynamo args (kv_role / disaggregation_mode — see above),
4. passes `--tensor-parallel-size <tp>`, `--pipeline-parallel-size <pp>`, `--data-parallel-size <dp>` (only when `dp > 1`), `--enable-expert-parallel` (when `ep: true`), `--max-model-len`, `--max-num-batched-tokens`, `--max-num-seqs`, `--gpu-memory-utilization`, `--kv-cache-dtype`, `--kv-transfer-config` (with `kv_connector`), `--override-generation-config` (from `vllm.override_generation_config`, when non-null) plus `vllm.extra_args`,
5. spawns the worker locally. (Multi-node SSH spawn is TBD; for now keep all worker hosts at 127.0.0.1.)

`up frontend` starts `dynamo.frontend` with `--http-host`, `--http-port`, `--router-mode`, `--discovery-backend`, `--request-plane`, and `--event-plane` from yaml, exporting `NATS_SERVER` and `ETCD_ENDPOINTS` into the child env. Assumes NATS and etcd (or whichever discovery backend is configured) are reachable.

`up opencode` renders `opencode/opencode.json` from the template, then runs `OPENCODE_EXPERIMENTAL_WORKSPACES=true OPENCODE_CLIENT=server bun run dev serve --hostname <host> --port <port>` from inside the vendored `opencode/` directory. (`bun dev` is the bun shorthand for `bun run dev`; the actual flag is `--hostname`, not `--host`, per `opencode/packages/opencode/src/cli/network.ts`.) `OPENCODE_CLIENT=server` suppresses the interactive `question` tool — see the gotcha below.

`up` with no arg brings up `workers → frontend → opencode` in order; `down` with no arg reverses those three. `monitor` and `scrape_metrics` are always opt-in (bring up/down separately; monitor requires `sudo` and `monitor.dcgm_py` set in yaml).

## Prerequisites (host install)

These must be on `PATH` before `testbed.sh up` will work. The exact commands
live in `README.md` under "Prerequisites" — do not duplicate them here, just
the rationale for each pin:

| Tool | Pin | Why |
|------|-----|-----|
| `yq` | apt | `testbed.sh` reads `testbed.yaml` via `yq` (kislyuk's Python yq, jq filters) |
| `nats-server` | apt | `testbed.sh up nats` single-node convenience (port 4222) |
| `etcd` / `etcdctl` | tarball v3.5.17 → `/usr/local/bin` | apt etcd is too old; Dynamo's default `--discovery-backend` is etcd |
| `vllm` | pip `==0.22.1` | matches the `dynamo` submodule pin `v1.3.0-minimax-m3-dev.1`'s vendored AsyncEngineArgs surface; bumping vLLM may break flag pass-through in `dynamo.vllm` (re-run `tests/test_dynamo_interface.py`) |
| `dynamo` submodule | tag `v1.3.0-minimax-m3-dev.1` | adds the `minimax_m3` tool-call + reasoning parsers (Rust) and targets vLLM 0.22.1. Bumping from v1.1.0 required a **cargo rebuild** of the Rust bindings/parsers, not just the Python patches |
| `nixl` | pip | KV-transfer connector class vLLM dlopen's when `--kv-transfer-config` selects `NixlConnector` |
| `kvbm` wheel | maturin build of `dynamo/lib/bindings/kvbm` (OPTIONAL — only for `vllm.kvbm.enabled`) | separate CUDA extension (`kvbm-py3` → python pkg `kvbm` with compiled `_core`; needs cudarc+nixl-sys, cargo feature `block-manager` is default-on). NOT produced by the main dynamo build — build/install on the GPU host before enabling KVBM; verify with `python -c "import kvbm.vllm_integration.connector"` |

If any of these drift (e.g., user upgrades vLLM and `dynamo.vllm` starts rejecting flags), update the pin here AND in the README install block AND re-run `tests/test_dynamo_interface.py` against the new vendored source.

## Build (TBD)

Vendored sources are built per their own install guides:
- **`opencode/`** — bun-based. Follow `opencode/README.md`. Runtime entrypoint: `bun run dev serve` (the `dev` script in the root package.json shells into `packages/opencode` and forwards trailing argv to yargs).
- **`dynamo/`** — must be built with the **vLLM backend** selected. Follow `dynamo/README.md` (or its `BUILD.md` equivalent), passing whichever feature flag the vendored version uses to enable the vLLM backend.

Concrete build commands will be folded into this doc once each vendor folder is analyzed and pinned.

## Running a workload

```
.venv/bin/python -m testbed run \
  --split lite --num-samples 20 --qps 0.5 --seed 42 \
  --out results/run1
```

CLI defaults (set in `cli.py`): `num_samples=10`, `qps=0.5`, `seed=42`, `split=lite`. Override via flags or `TESTBED__*` env vars.

### Workloads (`--workload swebench|apps|terminalbench`)

`run` / `pre-clone` / `smoke` all take `--workload` (default `swebench`, fully backward-compatible). The pluggable surface is `runner.Workload` (frozen dataclass) + the `runner.WORKLOADS` registry: `load_samples(split, seed, n)`, `render_prompt(sample)`, `prepare(sample, dest, *, reset)` — everything else (Poisson/sequential, semaphore, TaskRecord schema, manifest flow, error stages) is workload-agnostic. Samples MUST carry `instance_id` (swebench has it natively; `apps.load_samples` injects `apps-<problem_id:05d>`, `terminalbench.load_samples` injects `terminalbench-<task_id>`). The registry wraps the module functions in deliberate LATE-BINDING lambdas so `monkeypatch.setattr(<module>, "load_samples", ...)` still takes effect through the registry — tests must assert dispatch, not identity.

**APPS** (`src/testbed/apps.py`, dataset `codeparrot/apps`, HF config `"all"`; the dataset is legacy script-based, so on `datasets` >= 4.0 `_load_dataset` falls back to the Hub's `refs/convert/parquet` branch via the packaged parquet builder):
- Splits: `train | test` plus difficulty pseudo-splits `introductory | interview | competition` (= test filtered to that difficulty, filtered BEFORE sort+sample so selection stays deterministic). `--split` defaults to the workload's default (`lite` for swebench, `test` for apps and terminalbench) — it is validated in `cli._resolve_split`, not `click.Choice`.
- **No git**: `apps.prepare_workspace` materializes `PROBLEM.md` (question) + `solution.py` (starter_code for call-based problems, comment scaffold for stdio) into the workspace. Same idempotence contract as `_pre_clone`: existing workspace with `PROBLEM.md` = no-op (reset=False), wiped + rewritten (reset=True). Prepare failures still land as `error.stage="clone"` — that is the documented trace-schema name for stage-1 workspace preparation, for EVERY workload.
- Two judging modes per problem, detected from `input_output` JSON's `fn_name`: **stdio** (run `python solution.py`, stdin→stdout) vs **call-based** (`fn_name` present; LeetCode-style `class Solution` starter code). `render_prompt` tells the agent which mode applies. Hidden tests are NOT written into the workspace (the question text already contains the public examples).
- Manifest name gains a workload prefix for apps (`.workspaces-apps-<split>-s<seed>-n<n>.json`); swebench keeps the legacy name. `run` also rejects a manifest whose `workload` key mismatches.
- `config.json` records `workload`.
- **Scaffold comparison (SWE-agent)**: `scripts/run_sweagent_apps.py` drives the SAME deterministic APPS samples through the SWE-agent scaffold against the SAME Dynamo endpoint (litellm `openai/<served_name>` + `--agent.model.api_base`), strictly sequential. Purpose: isolate why tool-share differs across scaffolds (opencode ~1% vs ~35% reported for SWE-agent) — `--deployment docker` (default) is the upstream-supported path and includes the container round-trip the literature numbers include; `--deployment local` would isolate the pure scaffold factor but FAILS for non-root users (SWE-agent hardcodes copying the repo to the deployment root `/{repo_name}`; upstream PR #1132 adding a configurable base dir was closed unmerged — "use mini-swe-agent for fully local runs"). Workspaces are `apps.prepare_workspace` + `git init`+commit (base for SWE-agent's patch); the produced patch is `git apply`'d back so `evaluate_apps.py` scores the run dir identically. Trace records carry `task_start/end_unix_s` so `scripts/analyze_sweagent_traj.py --run <dir> --frontend logs/frontend.log` can decompose llm/tool/others per task (tool = `.traj` per-step execution times, llm = frontend `request completed` elapsed_ms joined by task window — sequential makes the join unambiguous), plus the env-setup split (`env_head_s` = start→first-request = docker+swe-rex bootstrap+tool install, empirically ~the whole "others" on docker runs; `env_tail_s`; `active_s`) and an **active-time share table** (head/tail excluded — the literature's "share of active time" definition; use THAT row for cross-scaffold comparison against opencode's turn-level shares, which carry no env startup either). ALL sweagent CLI flag names live in `build_sweagent_cmd()`; validate against a new sweagent version with `--dry-run` + `sweagent run --help` before a real run. Install SWE-agent FROM SOURCE on the run host, **editable from a clone** (`git clone … && pip install -e .`, pinned to a release tag) — NOT the `pip install "git+https://…"` one-liner: SWE-agent asserts `CONFIG_DIR = Path(sweagent.__file__).parent.parent / "config"` is a dir at import, and that top-level `config/` is not shipped as package data, so a non-editable install dies with `AssertionError: …/site-packages/config`. The PyPI name `sweagent` is also an unrelated squatted 0.0.1 package whose `togetherunidiff` dependency doesn't resolve. Full runbook (install + docker/proxy + analysis): `docs/sweagent_apps_runbook.md`.
- **Correctness eval**: `scripts/evaluate_apps.py --run results/<dir>` (APPS counterpart of extract+evaluate_predictions): re-derives samples from config.json, executes each workspace's `solution.py` against the hidden `input_output` tests (`--max-tests 20`, `--timeout-s 10` caps), writes `<run>/apps_eval.json` with per-instance verdicts (`resolved|unresolved|no_solution|no_tests|not_in_sample_set`) + `resolve_rate_all` / `resolve_rate_http_ok`. Pragmatic approximation of the official APPS harness (slightly stricter, uniformly). **Executes model-generated code — run in a container.**

**Terminal-Bench** (`src/testbed/terminalbench.py`, HF dataset repo `harborframework/terminal-bench-2.0` pinned at `_HF_REVISION` = `f2e8c75e…`, 2026-04-24, 89 tasks). NOT a parquet dataset — the repo is a flat collection of Harbor-format task dirs, fetched via `huggingface_hub.snapshot_download` (hf_hub is already a hard dep of `datasets`; HF caching/offline work as usual) and parsed from `<task>/task.toml` + `<task>/instruction.md`. Dirs missing `instruction.md` are skipped; a malformed `task.toml` degrades to metadata-less fields instead of failing the load.
- Splits: `test` (all tasks) plus difficulty pseudo-splits `easy | medium | hard` (= tasks whose `[metadata].difficulty` matches, filtered BEFORE sort+sample). Default split `test`. Manifest name: `.workspaces-terminalbench-<split>-s<seed>-n<n>.json`.
- **No git, no Docker**: `terminalbench.prepare_workspace` materializes `TASK.md` (the instruction) ONLY — same idempotence contract as apps (marker `TASK.md`; reset wipes + rewrites). `environment/` (Docker build context incl. challenge-construction scripts), `solution/`, and `tests/` are deliberately NOT copied: solution/tests leak the oracle, and the environment setup scripts CONSTRUCT the task state (equally answer-adjacent). The real benchmark runs each task inside its Docker image; this testbed does not reproduce that container, so `render_prompt` instructs the agent to treat the workspace as the task's root filesystem — re-anchor absolute paths like `/app/…` INSIDE the workspace and never write outside it (keeps the agent loop in-workspace for measurement + host hygiene; the catch-all permission would otherwise let `/app` writes escape silently).
- Purpose: terminal-command-heavy agent traffic for router/scheduling measurement — NOT Terminal-Bench-faithful correctness scoring. There is intentionally no evaluate script for this workload (faithful scoring needs the official Harbor harness + per-task containers).

Router sweeps are just a shell loop:
```
for r in round-robin least-loaded kv; do
  TESTBED__DYNAMO__ROUTER_MODE=$r deploy/testbed.sh down frontend
  TESTBED__DYNAMO__ROUTER_MODE=$r deploy/testbed.sh up   frontend
  .venv/bin/python -m testbed run --num-samples 20 --qps 0.5 --router $r --out results/$r
done
```

`--router` is **only recorded in `config.json`**; the actual router mode is whatever `dynamo.frontend` was started with (env `TESTBED__DYNAMO__ROUTER_MODE` → `testbed.sh`).

### Smoke-testing slices
```
scripts/curl_smoke.sh routes        # list OpenCode endpoints
scripts/curl_smoke.sh dynamo        # one /v1/chat/completions to Dynamo
scripts/curl_smoke.sh opencode      # one full session+message round trip
scripts/curl_smoke.sh swebench      # send a real SWE-bench prompt
```

## Per-task flow (runner.py:_run_one)

1. Compute `directory` (the workspace folder name; **not** the OpenCode session id) and `abs_dir = workspace_root / directory`. Default = `f"session-<instance_id>-<short_uuid>"` (collision-safe, accumulates across runs). With `--reset-workspace`, drops the uuid → `f"session-<instance_id>"` (path is stable across reruns of the same sample; required for reproducible opencode system prompts since opencode embeds cwd).
2. **Pre-clone the repo**: `git clone <sample.repo> <abs_dir>` and `git -C <abs_dir> checkout <sample.base_commit>`. Synchronous, fail-fast — done by runner before any OpenCode call. This sidesteps the in-agent `git clone` hang. Network clones go through `_run_git_retry` (exponential backoff) to ride out transient failures. **Workspace pre-clone (default on, `--pre-clone-workspaces`)**: `run()` assigns ALL workspace directory names up front (`_directory_for`) and clones EVERY task workspace via `prepare_workspaces()` (bounded concurrency 8) BEFORE the first request fires — the workload phase then performs zero clones (`_pre_clone` at task time sees the existing checkout and returns immediately; with `--reset-workspace` it just resets). Workspaces that fail to pre-clone are listed on stderr and retried at their task's arrival (legacy fail-fast path → `error.stage="clone"` if still failing). `--no-pre-clone-workspaces` reverts to per-task clone-at-arrival. **Standalone `pre-clone` command (preferred for flaky networks)**: `python -m testbed pre-clone --split <s> --num-samples <n> --seed <k> [--reset-workspace]` (cli.py → `runner.pre_clone_run`) does ALL workspace clones ahead of the run, **exits 1 if any workspace failed** (re-running resumes: only failures are retried — `_pre_clone` is a no-op on an existing checkout), and writes a manifest at `<workspace_root>/.workspaces-<split>-s<seed>-n<n>.json` (`workspace_manifest_path`) recording `{instance_id: directory}` + the `reset_workspace` flag. `run` with the same `(split, seed, n)` finds the manifest, reuses the exact directories, **skips its clone phase entirely**, records the path as `workspace_manifest` in config.json, and **deletes the manifest (single-use)** — a later run must not silently reuse dirty non-reset workspaces. A manifest with a mismatched `reset_workspace` flag or incomplete instance coverage is ignored (run falls back to its inline clone phase). There is intentionally NO repo-level cache layer (a `.repo-cache` of unique repos existed briefly and was removed) — every workspace is a direct retried network clone; the conservative path for flaky networks is the standalone `pre-clone` gate, not a cache.
3. POST `/session?directory=<abs_dir>` with body `{}` → server returns the assigned `session_id` (matches `^ses.*`). Then POST `/session/<session_id>/message?directory=<abs_dir>` (synchronous; blocks until the agent loop finishes). RTT measured with `time.monotonic()`. The directory query value MUST be the absolute path — see Module contracts above.
4. **Always GET `/session/<session_id>/message?directory=<abs_dir>` after** — the synchronous POST response carries only the FINAL assistant message; intermediate tool-loop steps are only available via the list endpoint.
5. Write a TaskRecord with the raw message dump and basic metadata. The TaskRecord stores `directory` (the folder NAME, relative — for human readability) and `session_id` (server-assigned) as separate fields. The absolute path is recoverable as `<workspace_root>/<directory>`.

OpenCode accepts a `?directory=` that points to an already-existing pre-cloned dir under `OPENCODE_EXPERIMENTAL_WORKSPACES=true` — verified against the vendored middleware (it just `path.resolve()`s the value and uses the resulting directory as the agent's working directory).

## Concurrency, errors, cleanup

**Concurrency**: Poisson generates arrival timestamps; runner fires each task at its arrival via `asyncio.create_task` and gates them with a bounded semaphore. Default `max_in_flight = 16`, set in `cli.py`, overridable via `--max-in-flight`. The semaphore is acquired **after** arrival time (so it represents queueing on top of the system, not an artificial backpressure on the arrival process itself).

**Sequential mode** (`--sequential`): bypasses Poisson entirely — task N+1 starts the moment task N's TaskRecord lands, exactly one request in flight at all times. `--qps` and `--max-in-flight` are recorded in `config.json` for provenance but do NOT influence execution. `arrival_offset_s` in each TaskRecord becomes the elapsed wall-clock from run start at the moment that task started (= cumulative RTT of prior tasks). Pair with `--reset-workspace` for byte-stable workspace state — together they remove the concurrent-batching and the path/state-divergence sources of agent-loop non-determinism. Use this for reproducibility comparisons; the Poisson mode stays the default for realistic workload measurement.

**Error policy**: fail-fast per task, no retry. The run as a whole never aborts on per-task failures.

| Failure point                                              | TaskRecord written? | `success` | `rtt_s`               | `error.stage` |
|------------------------------------------------------------|---------------------|-----------|-----------------------|---------------|
| `git clone` / `git checkout`                               | yes                 | false     | null                  | `clone`       |
| `POST /session` (non-2xx or HTTP error)                    | yes                 | false     | null                  | `session`     |
| `POST /session/:id/message` (non-2xx/network)              | yes                 | false     | wall-clock to failure | `message`     |
| `POST /session/:id/message` exceeded `--task-timeout-s`    | yes                 | false     | wall to abort (≈timeout) | `timeout`  |
| `GET /session/:id/message` (after good POST)               | yes                 | true      | from the POST         | `list`        |

`error.stage = "list"` means RTT is valid but `messages` may be empty/partial. `error.stage = "timeout"` means the agent loop was wall-clock-aborted at `task_timeout_s`. On timeout the runner **first POSTs `/session/:id/abort`** (best-effort, own 30 s cap; recorded as `error.aborted`) — cancelling the message POST alone only drops the HTTP request while opencode keeps running the agent loop server-side (route handlers run `svc.prompt` to completion regardless of client disconnect), and that ZOMBIE loop keeps firing LLM turns that pin GPU KV blocks under later sessions (observed as a growing residual KV-usage floor after a timeout). It then does a **best-effort `GET /session/:id/message`** (bounded by its own 30 s cap) to capture the turns opencode persisted before the abort, so `messages` holds the partial trajectory (not `[]`) and `error.partial_messages` records how many were recovered (0 if that list call also failed). The OpenCode session/workspace may hold further state on disk at `<workspace_root>/<directory>`.

**Cleanup**: none by default. `<workspace_root>/session-<instance_id>-<uuid>/` directories accumulate across runs; prune manually (`rm -rf /tmp/testbed-workspaces/session-*`) between large runs. Per-task cleanup is intentionally avoided so failing runs can be inspected. **With `--reset-workspace`**, the runner instead reuses `<workspace_root>/session-<instance_id>/` and wipes it back to `base_commit` (git reset --hard + git clean -fdx) before each task — same final state as a fresh clone, no network round-trip. Broken checkouts (interrupted prior clone) are detected with `git rev-parse --git-dir` — NOT a bare `.git` presence check, since a half-written `.git` directory passes that but every git command then dies with "fatal: not a git repository" — and are nuked + re-cloned in both modes; a valid-looking repo whose `reset --hard` still fails (missing objects) likewise falls through to a full re-clone. Non-existent dirs get the normal full clone path. `workspace_root` is normalized with `expanduser().resolve()` at every entry point (`run`, `pre_clone_run`, `smoke`) — a relative value in `testbed.yaml` would otherwise anchor `git -C` and the OpenCode `?directory=` on the process CWD.

## Output files (one run, `--out results/<dir>/`)

- `config.json` — invocation parameters (split, num_samples, qps, seed, max_in_flight, **router**, model, resolved `testbed.yaml` snapshot)
- `trace.jsonl` — one TaskRecord per line:
  ```json
  {
    "instance_id": "django__django-12345",
    "session_id": "ses_a1b2c3d4...",
    "directory": "session-django__django-12345-a1b2c3d4",
    "arrival_offset_s": 3.21,
    "rtt_s": 42.31,
    "success": true,
    "error": null,
    "messages": [ /* raw OpenCode list_messages response, JSON-as-is */ ]
  }
  ```
  On failure, `error` is `{"stage": "clone|session|message|list", "type": "...", "msg": "..."}`. `messages` is the raw OpenCode `list_messages` response with no transformation; `[]` if `error.stage` is upstream of the list call.
- `summary.json` — strictly: `rtt_s` p50/p95, `success_rate`, `count`. Nothing else.

### True resolve/fail (SWE-bench correctness)

`trace.jsonl.success` is HTTP-level only. Real resolution is judged post-hoc with the **official SWE-bench harness** via three scripts:
- `scripts/extract_predictions.py --run results/<dir>` — per task: `git -C <ws> add -A && git diff --cached <base_commit>` (junk excluded via `:(exclude)` pathspecs: `__pycache__`, `*.pyc`, `.pytest_cache`, `*.egg-info`, `.opencode`) → `<run>/predictions.jsonl` in the official format. base_commit comes from re-deriving the sample set with `load_samples(split, seed, n)` (deterministic) read out of `config.json`; `--base-commits-json` / `--head-as-base` are the offline fallbacks. Failed tasks emit an EMPTY `model_patch` (harness counts them unresolved — correct semantics). NOTE: mutates the workspace **index** (not the working tree) via `add -A`; idempotent.
- `scripts/evaluate_predictions.sh --run results/<dir>` — wrapper: extract (skipped if predictions.jsonl exists) → `python -m swebench.harness.run_evaluation` run **from inside the run dir** so the report (`<model>.<run_id>.json`) + `logs/run_evaluation/` land next to trace.jsonl. Needs `pip install swebench` + Docker on the eval host.
- `scripts/analyze_eval_results.py --run results/<dir>` — joins report ⋈ trace by instance_id → per-instance verdict (`resolved|unresolved|empty_patch|error|incomplete|not_in_report`; handles both `resolved_ids`- and `resolved`-style report keys) + `resolve_rate_all` vs `resolve_rate_http_ok`.

## Vendored-submodule patches (`deploy/patches/`)

Both submodules stay pinned at their upstream commits; testbed-owned changes live as patches under `deploy/patches/`, prefix-routed to two apply scripts so they never cross-apply:
- `opencode-*.patch` → `scripts/apply_opencode_patches.sh` (opencode submodule)
- `dynamo-*.patch`   → `scripts/apply_dynamo_patches.sh` (dynamo submodule)

Both scripts take no arg (apply, idempotent), `--check` (report applied/pending), `--revert`. Run after `git submodule update --init`. The dynamo patches are **Python-only** (handlers.py) — applying them needs no cargo rebuild, just restart workers (`testbed.sh down workers && up workers`) to pick them up. (Note: a *submodule version bump* itself — e.g. the v1.1.0 → v1.3.0-minimax-m3-dev.1 move — does require a full cargo rebuild of dynamo's Rust core, separate from the patches.)

**`dynamo-scheduling-log.patch`** — adds `BaseWorkerHandler._log_scheduling_delay()` + 3 call sites (decode token/text, prefill) in `dynamo/components/src/dynamo/vllm/handlers.py`. Emits one `SCHED_DELAY request_id=.. role=prefill|decode queue_ms=.. queued_ts=.. scheduled_ts=..` line per request to the worker log, read from vLLM v1's per-request `RequestOutput.metrics.{queued_ts,scheduled_ts}` (the engine scheduler queue-wait = scheduling delay). This is the per-request sink because the value **cannot** ride in-band to the client: the frontend re-serializes `usage` through upstream async-openai types that drop unknown keys, and the `nvext` response block is Rust-only (no Python write path). Parse with `scripts/analyze_worker_scheduling.py --logs logs/` → per-(worker,role) p50/p90/p99. With PD disaggregation the prefill and decode workers each log their own line, so prefill vs decode scheduling delay come out separately. To see queue wait **as a fraction of end-to-end**, `scripts/analyze_request_wait.py --frontend logs/frontend.log --logs logs/` joins the frontend "request completed" line (total elapsed_ms) with the SCHED_DELAY lines by `request_id` (same Context UUID across frontend↔prefill↔decode). Dynamo logs carry NO opencode sessionID (`x-session-affinity` is never surfaced dynamo-side), so per-session needs an external `request_id,session_id` map passed via `--session-map` — produce it with `scripts/analyze_turn_scheduling.py --emit-session-map <csv>` (the profile patch records each turn's `request_id`; see OpenCode profiling below). `scripts/analyze_turn_scheduling.py --profiles <dir> --logs logs/` additionally joins profile TURNS to SCHED_DELAY records: exact join when the profile carries `request_id` (patched runs), timestamp-alignment fallback (`dynamo.request_received_unix_s` ↔ prefill `queued_ts`, greedy 1:1 within `--tolerance-s`, optional `--prompts` ISL corroboration) for legacy runs. Output: per-preceding-tool distributions of output-tokens vs scheduler queue-share (`by_tool.csv`, `turn_sched.csv`) — the small-LLM-turn CPU-offloading analysis. `turn_sched.csv` also carries `cur_tools`, `away_s`, `away_displaced_tokens` (other sessions' KV-token allocation during the away window — the LRU displacement-pressure proxy) and cache-hit columns. Downstream of it: `scripts/compare_prefill_compute.py` (A/B baseline-vs-KVBM `prefill_compute_ms` distributions → per-request onboard-cost estimate; the ONLY handle since this dynamo tag records no per-request transfer duration) and `scripts/cpu_offload_breakeven.py` (GPU-path vs CPU-path cost model per turn + break-even cpu_decode_tps; CPU throughput knobs come from a microbench). Experiment storyline + run matrix: `docs/turn_kv_experiment_plan.md`. `queue_share`'s denominator prefers `dynamo.elapsed_s` and falls back to `llm_wall_s` when nvext timing is absent (`queue_share_basis` column says which); a coverage line reports prefill/decode/elapsed availability (agg/colocation runs have decode-only SCHED_DELAY records — prefill 0/N there is expected, not a bug). The script ALSO emits `away_cache.csv` (profile-only, works with zero SCHED_DELAY records): per-turn `away_s = llm.start(N) − llm.end(N−1)` (time the session was off the GPU running tools) bucketed against `cache_hit_ratio = tokens.cache.read / (input + cache.read)` + re-prefilled token counts + pearson r — the direct measurement of KV-eviction cost of GPU↔CPU turn transitions.

**`dynamo-prompt-dump.patch`** — captures the **exact prompt delivered to the vLLM engine** per request. Adds `import json`, module-level config (`_PROMPT_DUMP_*` read from env at import) + a per-process append-file helper, `BaseWorkerHandler._dump_engine_prompt()`, and 3 call sites — one immediately before each `engine_client.generate(...)` in `handlers.py` (decode token mode `generate_tokens`, decode text mode `_generate_text_mode`, prefill `_generate_token_mode`; each request hits exactly one, so no double-count). What it dumps is the prompt **as the engine sees it**: the Dynamo frontend (Rust) applies the model's chat template and tokenizes, then ships the worker `request["token_ids"]` (a `TokensPrompt`); this hook reads those `prompt_token_ids` and **detokenizes them back to text** via `self.engine_client.tokenizer.decode(...)`. This is fundamentally different from the OpenCode profile snapshot, which records OpenCode's **pre-template** wire `messages` array (see the gotcha below). Gated by `DYN_PROMPT_DUMP` (full no-op otherwise — one bool check on the hot path). Knobs (read once at import): `DYN_PROMPT_DUMP_DIR` (default `/tmp/dynamo-prompt-dump`; `testbed.sh up workers` forces it to `<workspace_root>/prompts`), `DYN_PROMPT_DUMP_TEXT` (default **on** — include detokenized `prompt_text`), `DYN_PROMPT_DUMP_TOKENS` (default **off** — include the raw `prompt_token_ids` array; large). Output: one NDJSON record per request to `<dir>/prompt-<pid>.jsonl` (per-PID so concurrent prefill/decode workers don't clobber), fields `{ts, request_id, role, num_prompt_tokens, prompt_text?, prompt_token_ids?, decode_error?}`. Inspect with `scripts/format_prompt_dump.py --prompts <dir>` (pretty-prints each turn with real newlines, dedups the prefill/decode pair per request_id; `--delta` shows only the text new since the previous turn). For **machine consumption** (importing the trace into an external simulator) use `scripts/export_prompt_turns.py --prompts <dir> --out turns.jsonl` — one JSON line per request with the prompt split into chat-template turns (ChatML `<|im_start|>` framing, `--template qwen3` default) and per-turn `text|think|tool_call|tool_response` segments; offsets are lossless (preamble+turns tile the full prompt_text, segments tile each turn's content span, `text` is the verbatim slice), tool_call segments carry the extracted function `name` (Qwen3-Coder `<function=`, Qwen3 JSON, MiniMax M3 `<invoke name=` — tag strings verified against `dynamo/lib/parsers/src/tool_calling/config.rs`), `--no-text` drops slices for a compact sizes-only trace, `--tokenizer <id|path>` adds per-segment/turn `num_tokens` (runs offline on the dump — the hot-path patch stays untouched, so already-captured dumps can be exported without re-running workers). `role` is `prefill|decode`; with PD both workers receive the same token_ids, so filter to `prefill` for the canonical prompt. Correlate with frontend/scheduling logs by `request_id` (same Context UUID). **Perf caveat**: detokenizing + JSON-serializing a multi-thousand-token prompt per request adds real hot-path latency and synchronous file I/O — enable for prompt-capture runs ONLY, never alongside timing-sensitive profile/scheduling measurements. Enable via `DYN_PROMPT_DUMP=1 testbed.sh up workers` (restart workers to pick it up; Python-only, no cargo rebuild).

## OpenCode profiling (ENV-gated)

The profiler lives as a **testbed-owned patch** at `deploy/patches/opencode-profile.patch` (the `opencode/` submodule stays pinned at its upstream tag, e.g. `v1.14.41`, so the parent repo is portable). One-time setup after `git submodule update --init`:

```
scripts/apply_opencode_patches.sh           # idempotent; safe to re-run
scripts/apply_opencode_patches.sh --check   # report applied/pending
scripts/apply_opencode_patches.sh --revert  # back out cleanly
```

After applying, OpenCode gains `packages/opencode/src/profile/profile.ts` plus hook call sites in `session/prompt.ts` and `session/processor.ts` (`start-step`, `finish-step`, `text-end`). Activate with `OPENCODE_PROFILE=1`; every call is a no-op otherwise. `testbed.sh up_opencode` automatically forces `OPENCODE_PROFILE_DIR=<workspace_root>/profiles` when the env var is truthy so all sessions land in one flat directory regardless of `?directory=`.

Knobs (set before `testbed.sh up opencode`):
- `OPENCODE_PROFILE=1` — enable
- `OPENCODE_PROFILE_DIR=<abs>` — override default (`<workspace_root>/profiles`)
- `OPENCODE_PROFILE_MESSAGES=count|head|full` — snapshot fidelity (default `head` = first 200 chars per part)

Per-session NDJSON layout: one file per `sessionID`, events `query.start, turn.start, llm.start, llm.stream-finish, llm.end, tool.start, tool.end, turn.end, query.end`.

Timing fields on `llm.end` (NOTE: there is NO `stream_end_s` field — the current patch folds the stream anchor into `duration_s`):
- `duration_s` — client-side LLM wall: `start-step → AI-SDK "finish" event` whenever the `Profile.llm.streamFinish` hook fired for the step (`case "finish":` in `processor.ts`; the finish event fires the instant the stream is fully consumed — the **true client-side stream wall**; the sibling `llm.stream-finish` event + non-null `post_stream_overhead_s` are the markers). Fallback when the finish anchor is absent: `start-step → first tool.start` (or last `text-end`) — the legacy approximation that significantly under-measures when closing/finish_reason chunks come AFTER the first tool_call. Consumers detect which anchor applied via `post_stream_overhead_s != null` or a matching `llm.stream-finish` event (see `e7_llm_wall_check.py`).
- `step_duration_s` — full AI SDK step bracket (`start-step → finish-step`). Includes post-stream framework finalization (snapshot.track + snapshot.patch + DB writes in `processor.ts:455-514`); `post_stream_overhead_s` = finish-step − stream finish (recorded explicitly, null when the finish anchor is absent).

`llm.end.dynamo` (also surfaced on `llm.stream-finish`) carries Dynamo's in-band timing, captured from raw SSE chunks (see the response-side precondition below — providerMetadata never carries it):
- `elapsed_s` = dynamo's `total_time_ms / 1000` (server-side wall from HTTP receipt to last chunk; matches the `request completed` log line's `elapsed_ms`).
- `request_received_unix_s` = `request_received_ms / 1000` (wall-clock at Dynamo when the HTTP request landed). Comparing with `llm.start.ts` decomposes "client setup + request upstream" from "dynamo internal".

Because dynamo timings ride in-band, **no log scraping is needed** to cross-reference profile NDJSON with the dynamo frontend log — **PRECONDITION: the request must opt in.** Since dynamo tag `v1.3.0-minimax-m3-dev.1`, `nvext.timing` is **per-request opt-in**: the frontend attaches it (final chunk only, both agg and disagg) ONLY when the request body carries `nvext: {"extra_fields": ["timing"]}` (`dynamo/lib/llm/src/protocols/openai/nvext.rs:215-234,297`; no server-side force flag — `DYN_ENABLE_FRONTEND_NVEXT` defaults on but does not auto-enable timing). `deploy/opencode.json.tmpl` therefore ships `"nvext": {"extra_fields": ["timing"]}` (with `seed`) on the provider **model** `options` — merged into EVERY request regardless of agent (`session/llm.ts:141`), which is what covers task-tool SUBAGENT sessions whose agent name is outside the six agent blocks (agent-level options alone left those sessions wholly null) — and repeats it per-agent, riding the providerOptions→request-body spread. Without it every profile `llm.end.dynamo` is `null` (request_id still populates — it comes from `response.id`, not nvext) and all elapsed-based analyses fall back to client-bracket approximations. Verify after `up opencode` with `scripts/sse_chunk_timing.py` (reads `chunk.nvext.timing`) or by checking a fresh profile's `llm.end.dynamo`. (`scripts/mock_llm_server.py` emits `nvext.timing` unconditionally, so mock runs mask a missing opt-in.) **SECOND PRECONDITION — response side**: the `@ai-sdk/openai-compatible` adapter parses each SSE chunk through a zod schema that STRIPS unknown top-level fields, so `nvext` never reaches `providerMetadata` — reading `providerMetadata.nvext.timing` is structurally dead. The profile patch therefore sets `streamText(includeRawChunks: true)` (session/llm.ts), handles the `raw` stream part in processor.ts (`Profile.llm.rawChunk` stashes the last chunk's `nvext.timing` per step), and the `llm.stream-finish`/`llm.end` hooks prefer that stashed value over providerMetadata. Requires the CURRENT `opencode-profile.patch`; an older applied patch shows the tell-tale combo request_id ✓ / no `llm.stream-finish` events / dynamo ✗. (If `--revert` fails with "cannot reverse-apply", an older patch version is applied — recover with `apply_opencode_patches.sh --reset` then re-apply.)

`llm.end` also records `response_id` (the OpenAI response id from the AI SDK finish-step part) and `request_id` (= `response_id` with the `chatcmpl-` prefix stripped). The Dynamo frontend sets every chunk's id to `chatcmpl-<request_id>` (`dynamo/lib/llm/src/protocols/openai/chat_completions/delta.rs`), so this `request_id` is the SAME Context UUID as the frontend `request completed` and worker `SCHED_DELAY` lines — the exact per-turn join key consumed by `scripts/analyze_turn_scheduling.py`. Both fields are `null` on non-dynamo providers.

**Mock LLM server** (`scripts/mock_llm_server.py`, stdlib-only): OpenAI-compatible SSE responder replacing the Dynamo frontend for CPU-contention experiments (very high `--max-in-flight` without GPU KV pressure). Template responder: N tool-calling turns (tool picked from the request's own `tools` array, bash preferred; `--tool-cmd` sets the command) then a final text turn; `--ttft-ms`/`--itl-ms`/`--output-tokens` inject synthetic latency; emits a dynamo-shaped `nvext.timing` block so profile `llm.end.dynamo` stays populated; writes `logs/mock_llm.ndjson` + `logs/mock_llm.pid` (picked up by `monitor_resources.py`). Wire in by repointing `.dynamo.host/.dynamo.port` (or `TESTBED__DYNAMO__HOST/PORT`) before `up opencode` — no yaml schema or testbed.sh change. Full experiment recipe: `docs/cpu_contention_runbook.md`.

`llm.end.tokens` is the AI-SDK-normalized usage object (`{total, input, output, reasoning, cache:{read,write}}`) from `session.ts:getUsage`; `input` already has cache tokens subtracted so dynamo's ISL = `tokens.input + tokens.cache.read [+ tokens.cache.write]`. `turn.end` carries `duration_s = llm_wall_s + tool_wall_s + post_overhead_s` (clamped at 0 for parallel-tool steps). The `tool.*` pair is fired by `Effect.onExit` so failures + dies still produce a `tool.end` with `ok:false`. Aggregate across sessions with `scripts/aggregate_profiles.sh <workspace_root>` (auto-prepends the `/profiles` subdir). Full step-event lifecycle and decomposition rationale: `docs/opencode_step_events.md`.

## Configuration overrides

`.env` (copy from `.env.example`) — secrets only:
```
OPENCODE_SERVER_PASSWORD=...
DYNAMO_API_KEY=...
HF_TOKEN=...
```

Anything in `testbed.yaml` can be overridden by a `TESTBED__<section>__<key>` env var (e.g. `TESTBED__VLLM__PREFILL__GPU_MEMORY_UTILIZATION=0.85`).

Resolution precedence: **CLI flag > `TESTBED__*` env var > `testbed.yaml` > built-in default**.

Worker and model knobs are consumed at component **launch time** (`up workers`, `up opencode`). Setting or changing a `TESTBED__*` env var or editing `testbed.yaml` after a component is already running has no effect until you restart that component (`down workers && up workers`). The env must be present in the shell that invokes `testbed.sh up workers`, not just exported somewhere else.

## Conventions / gotchas

- **Reproducible sampling targets OUTPUT TEXT, not bit-level intermediate tensors.** The internal floating-point math is allowed to drift across runs as long as the argmax token at each step stays the same. The default recipe in `testbed.yaml` pins JUST what's free (seeds + greedy) and leaves vLLM's throughput optimizations on:
  - `model.temperature: 0.0`, `model.top_p: 1.0` — greedy decoding. Overrides opencode's `ProviderTransform.temperature(qwen) = 0.55` default (`provider/transform.ts:464-480`).
  - `model.seed: 42` — per-request seed. Rendered into `opencode.json`'s `agent.<name>.options.seed` (FLAT — opencode's `ProviderTransform.providerOptions(model, options)` at `provider/transform.ts:1186` wraps the dict under the provider key automatically; pre-nesting under `options.<provider>.seed` produces a double-wrap that vLLM rejects with HTTP 400). The wrapped value reaches the openai-compatible AI SDK adapter as `providerOptions[<provider>].seed`, which gets forwarded as the OpenAI request body's `seed` field, which `dynamo/components/src/dynamo/vllm/handlers.py:411` then maps onto `SamplingParams.seed`. Pins tie-break RNG draws in the sampler.
  - `vllm.seed: 42` — engine-level `--seed`. Pins vLLM's scheduler / sampler RNG across runs.
  - `vllm.enable_prefix_caching` / `enable_chunked_prefill` / `enforce_eager` / `disable_custom_all_reduce` — **default null = leave vLLM defaults on**. Prefix caching, chunked prefill, continuous batching, custom all-reduce, and CUDA-graph capture all introduce only epsilon-level FP variance that rarely flips greedy argmax for typical SWE-bench text. Flip them to false/true ONE AT A TIME and bisect if you observe output divergence with seeds pinned.
  - `vllm.override_generation_config` — dict merged into vLLM's `--override-generation-config '<json>'`. Closes the reproducibility gap that `--seed` alone misses: model-shipped `generation_config.json` defaults (Qwen's is `{temperature:0.7, top_p:0.8, top_k:20, repetition_penalty:1.05}`) flow into the server's default `SamplingParams`, and **`repetition_penalty` is applied to logits BEFORE argmax** — so leaving it at 1.05 makes "greedy" not actually pure-argmax. Default ships as `{temperature:0.0, top_p:1.0, top_k:-1, repetition_penalty:1.0}` to neutralize this. Per-request fields from opencode (`model.temperature`, `model.top_p`, `seed` via providerOptions) still win field-by-field; this only changes the server default for fields the client doesn't set. Wired in `deploy/testbed.sh` via `yq -c '.vllm.override_generation_config'` → `--override-generation-config '<compact-json>'`. Set to `null` or `{}` in yaml to skip the flag.
  Hard limit at the edge: with `--max-in-flight > 1`, concurrent batches mix requests differently across runs which CAN flip argmax on near-tie tokens. For strict per-task output reproducibility the cleanest combination is **`--sequential --reset-workspace`** — sequential mode guarantees only one request in flight at any moment (no concurrent batching) AND drops the Poisson distribution so timing-induced ordering variance also vanishes; `--reset-workspace` makes the workspace path + state byte-stable across reruns (opencode embeds the working-directory path in its system prompt, so the default uuid-suffixed workspace name would otherwise make the prompt different on every run → first token already different → agent loop diverges and turn count varies wildly). `--max-in-flight 1` alone leaves Poisson arrivals in place, so it produces artificial idle gaps without removing the in-flight-overlap concern (the next task can still arrive while the prior is finishing). Multi-GPU TP also retains residual variance in some kernels regardless.
- **Vendored sources are authoritative for any implementation detail.** When in doubt about a CLI flag, an env var, or a wire-format detail of OpenCode/Dynamo/vLLM, read the vendored source (`opencode/`, `dynamo/`) — do not rely on memory or external docs that may not match the pinned version.
- **OpenCode is headless-only**, launched as `OPENCODE_EXPERIMENTAL_WORKSPACES=true bun run dev serve --hostname <h> --port <p>` from `opencode/`. No TUI, no shared global workspace. Every session/message call must include `?directory=<absolute path>` (`InstanceMiddleware` reads it). Without the env var, concurrent agents will trample each other in a single CWD. **Always send the absolute path** — `InstanceMiddleware` calls `AppFileSystem.resolve()` (Node `path.resolve()`) so a bare folder name resolves against OpenCode's CWD (= the `opencode/` repo), not against `workspace_root`.
- **Pass `OPENCODE_CONFIG=<abs path>` when launching OpenCode.** OpenCode's per-instance config loader walks UP from the request's `?directory=` looking for `opencode.json` (`opencode/packages/opencode/src/config/paths.ts:10-21`, consumed at `config/config.ts:568-572`). Our rendered `opencode/opencode.json` is NOT on that walk path because `?directory=` lives under `/tmp/testbed-workspaces/<session>/`. Without `OPENCODE_CONFIG=<abs>` (loaded as "custom config" at `config.ts:563-566`), the rendered file is silently ignored, `cfg.provider` is empty, and `HF_TOKEN` (commonly exported for vLLM) auto-enables the `huggingface` provider via `provider.ts:160-181` (`input.env.some(item => env[item])`) — sending all inference to `https://router.huggingface.co/v1` with whatever `qwen/...` model happens to be in the `huggingface` block of `models.dev/api.json`. Belt-and-suspenders: `opencode.json.tmpl` also pins `disabled_providers: ["huggingface"]` (`provider.ts:1133` consumes this).
- **Launch OpenCode with `OPENCODE_CLIENT=server` to suppress the `question` tool.** `OPENCODE_CLIENT` defaults to `"cli"` (`opencode/packages/core/src/flag/flag.ts`), and `registry.ts:193-194` exposes the interactive `question` tool whenever `OPENCODE_CLIENT ∈ {app, cli, desktop}` (or `OPENCODE_ENABLE_QUESTION_TOOL` is set). In a headless run there is no human to answer: when the model calls `question`, `Question.ask` registers a `Deferred` and **blocks on `Deferred.await` indefinitely** (`opencode/packages/opencode/src/question/index.ts:174`) until a `POST /question/:requestID/reply` (or `/reject`) arrives — which the runner never sends. The agent loop therefore hangs until `--task-timeout-s` fires, producing a `error.stage="timeout"` TaskRecord that is pure measurement noise. `testbed.sh up_opencode` exports `OPENCODE_CLIENT=server` (outside the app|cli|desktop set) so the tool is never offered. Side effect: the `plan` tool is also gated on `OPENCODE_CLIENT === "cli"` (`registry.ts:233`) so it disappears too, but it additionally needs `OPENCODE_EXPERIMENTAL_PLAN_MODE` and is off by default anyway. (To drive question/reply yourself: `GET /question` lists pending, `POST /question/:requestID/reply` with body `{"answers": [["<label>"], ...]}` answers in question order; the `question.asked` bus event also surfaces over the `/event` SSE stream.)
- **OpenCode auth is HTTP Basic** when `OPENCODE_SERVER_PASSWORD` is set on the server. Username defaults to `"opencode"` (override via `OPENCODE_SERVER_USERNAME` server-side). The client sends `Authorization: Basic <base64(user:pass)>`. There is no `x-opencode-server-password` header — do not invent one.
- **Permissions must be pinned to a catch-all `allow` or the agent loop hangs (sibling to the question-tool hang).** opencode's permission `evaluate()` (`opencode/packages/opencode/src/permission/evaluate.ts`) returns `{action:"ask"}` as the DEFAULT when no rule matches, and the asking tools call `ctx.ask({permission:...})` which registers a `Deferred` and **blocks on `Deferred.await` indefinitely** (`permission/index.ts:205-209`) until a `POST /permission/:id/reply` arrives — which the headless runner never sends. The classic trigger is the model writing OUTSIDE the workspace (e.g. `/tmp/test_issue.py`). **Subtle:** an out-of-tree write does NOT ask the `edit` permission first — `tool/write.ts` calls `assertExternalDirectoryEffect()` (`tool/external-directory.ts`) BEFORE the edit ask, which asks a **separate `external_directory`** permission. The built-in default (`agent/agent.ts:271-281`) only allows `external_directory` for `Truncate.GLOB` (the truncation temp dir), NOT arbitrary `/tmp`, so allowing just `edit`/`bash`/`webfetch` is INSUFFICIENT and still hangs. `deploy/opencode.json.tmpl` therefore ships a top-level catch-all `"permission": {"*": "allow"}` that covers every gate (`edit`, `bash`, `webfetch`, `external_directory`, `doom_loop`, future ones). It works because opencode's `Wildcard.match` compiles `*` → `.*` with the dotall flag (`util/wildcard.ts`), so the `*` pattern matches deep external paths with slashes too; and `fromConfig` (`permission/index.ts`) turns the string `"allow"` into a `{permission:"*", pattern:"*", action:"allow"}` rule that the per-agent merge (`agent/agent.ts:248`) applies to every agent. (Immediate override without re-rendering: export `OPENCODE_PERMISSION='{"*":"allow"}'` before `up opencode` — `config.ts:710-711` merges it. Restart opencode to pick up either change.) `tests/test_opencode_template.py:test_rendered_template_permission_is_catchall_allow` guards this invariant.
- **Runner pre-clones; agent does not.** SWE-bench tasks have repo+base_commit. Runner does the clone+checkout synchronously before opening the OpenCode session. The agent operates on a pre-prepared checkout, never `git clone`s itself (this has been observed to hang the agent loop).
- **System prompt lives on the user message** (`info.system`), and OpenCode types it as `string[]`. Consumers should normalize lists to `"\n\n".join(...)`.
- **Two different "prompt" measurement layers — don't conflate them.** (1) The OpenCode **profile** snapshot (`OPENCODE_PROFILE`, `turn.start.{system,messages}`) records the prompt as OpenCode *builds* it: the AI-SDK `messages` array + `system` strings, **before** any chat template — and `OPENCODE_PROFILE_MESSAGES=head` truncates each part to 200 chars. (2) The **engine-prompt dump** (`DYN_PROMPT_DUMP`, `dynamo-prompt-dump.patch`) records the prompt as the **vLLM engine** receives it: the frontend-applied chat-template `token_ids`, detokenized to one flat text string, untruncated. The chat template (role tags, tool-call formatting, BOS/special tokens) is applied **frontend-side in Rust**, so it appears ONLY in layer (2). If the question is "what tokens does the model actually process," it's (2); if it's "what did OpenCode assemble/send," it's (1). They will not match character-for-character.
- **Each PD worker needs a unique NIXL side-channel port.** vLLM defaults all workers to 5600 and they collide on a single host. `testbed.sh` exports `VLLM_NIXL_SIDE_CHANNEL_HOST` (per worker, from `worker.host`) and `VLLM_NIXL_SIDE_CHANNEL_PORT` (`nixl_port_base + rank*100`). Do not rely on vLLM defaults.
- **kv_role + disaggregation_mode are derived, not configured.** Workers in `prefill_workers` get `--disaggregation-mode prefill` and `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'`; `decode_workers` get the `decode`/`kv_consumer` variant.
- **EP/DP are per-worker, not global.** Each worker entry can carry `dp` (`--data-parallel-size`, default 1, flag emitted only when `>1`) and `ep` (`--enable-expert-parallel`, default false). EP shards MoE experts across the `tp*dp` ranks (EP size derived by vLLM), so it's a bare toggle that does NOT change the `tp*pp*dp` GPU-count invariant. Both are plain pass-throughs to `AsyncEngineArgs` (`dynamo.vllm` exposes the full vLLM arg surface via `AsyncEngineArgs.add_cli_args(..., async_args_only=False)`).
- **etcd and NATS are external.** Discovery is etcd by default; NATS is the request/event plane. `testbed.sh up etcd` and `testbed.sh up nats` are single-node conveniences only.
- **`bun dev` vs `bun run dev`**: `bun dev` is the bun shorthand and `bun run dev` is the explicit form; both invoke the `dev` script from `opencode/package.json`. `--hostname` (not `--host`) is the OpenCode flag — see `opencode/packages/opencode/src/cli/network.ts`.
- **OpenCode session ids are server-generated** and match `^ses.*` (per the SDK schema). The runner stores them as `session_id` in the trace; `directory` is the runner-chosen workspace folder name.
- **PGID-based teardown** is necessary because OpenCode's `.opencode` worker and vLLM's TP/PP shards `setsid` out of the parent. The logic is inlined in `testbed.sh` (no `_lib.sh`). `down opencode` also runs `kill_port` on `opencode.port` as a backstop.
- **`resource.ndjson` gauges are window aggregates, not point samples.** DCGM internally samples at `monitor.dcgm_update_freq_s` (default 100 ms) and `monitor_resources.py` drains via `DcgmGroupSamples.GetAllSinceLastCall(dfvc, fieldGroup)` every `interval_s` (default 1 s). Each output row's gauge field is therefore `{mean, min, max, n}` over the ~10 internal samples in that window. **Cumulative counters in `COUNTER_FIELDS`** (`PROF_PCIE_*_BYTES`, `PROF_NVLINK_*_BYTES`) stay as a single LAST-value scalar so downstream `(last_curr − last_prev) / interval_s` bandwidth math still works. The `dfvc` collection MUST be reused across drains — it carries the since-timestamp cursor inside; constructing a fresh one replays the whole DCGM ring buffer. Downstream consumers (`analyze_session_resources.py`) read `mean` out of the dict; the dict-vs-scalar branch is in `extract_metrics`.
- **Single config file invariant**: do NOT introduce a second config source (e.g. a separate `workers.env`). If a new knob is needed, add it to `testbed.yaml` and teach `config.py` + `testbed.sh` to read it.
- **Logs path is fixed**: `./logs/` (relative to `testbed.sh`'s CWD). PID files: `./logs/<component>.pid`. Stdout/stderr: `./logs/<component>.log`.

## Editing rules

- README.md is user-facing; CLAUDE.md is for future agent sessions. Keep them in sync only on user-visible commands; internal invariants belong here.
- New CLI flags go into `cli.py`.
- Schema changes to `testbed.yaml` require a matching update to `config.py` validation and the docs at the top of the yaml file.
- When something in this doc says "TBD" or "exact ... derived from vendored source", that's a contract to read the vendor folder rather than guess. Do not pin a value here without first checking the vendored source matches.

## Branch

`main` is the active branch — the scaffolding has merged. Push fixes directly to `main` unless told otherwise.

## Test suite (interface-drift detectors)

`tests/test_*_interface.py` and `tests/test_opencode_template.py` exist specifically to fail-loudly when a vendored submodule renames a flag or schema field that the testbed depends on. They grep `dynamo/` and `opencode/` source for known names; they do NOT need the components running. If one of these tests starts failing, **read the vendored source first** before patching the test — the test exists to catch real upstream drift, not to be silenced.