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
                  --router-mode {round-robin|least-loaded|kv}
                  worker discovery via NATS
                           │
                           ▼ (KV cache transfer over NIXL)
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
    testbed.yaml           # SINGLE source of truth: workspace, model, vLLM PD, dynamo, opencode
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
    # POST /session?directory=<directory>; returns session_id

async def send_message(session_id: str, prompt: str, directory: str) -> dict
    # POST /session/:id/message?directory=...; blocks until agent loop completes;
    # returns the raw JSON envelope (which carries only the FINAL assistant message — see per-task flow).

async def list_messages(session_id: str, directory: str) -> list[dict]
    # GET /session/:id/message?directory=...; returns the full message list as-is.
    # This is the canonical source of intermediate tool-loop steps.

async def stream_events(session_id: str, directory: str) -> AsyncIterator[dict]
    # GET /event SSE. Exposed for debugging only; runner does NOT consume this.
```

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

vllm:
  kv_connector: NIXL              # vLLM kv-transfer connector
  nixl_port_base: 6000            # rank N → VLLM_NIXL_SIDE_CHANNEL_PORT = base + N*100

  # Each worker entry:
  #   host: where this worker is deployed. Default 127.0.0.1.
  #         Doubles as VLLM_NIXL_SIDE_CHANNEL_HOST for that worker.
  #         Single-node only at present; multi-node SSH spawn is TBD. Keep host=127.0.0.1 until then.
  #   gpus: comma-separated GPU ids. Must satisfy len(split(',')) == tp * pp.
  prefill_workers:
    - { name: p0, host: 127.0.0.1, gpus: "0,1", tp: 2, pp: 1 }
  decode_workers:
    - { name: d0, host: 127.0.0.1, gpus: "2,3", tp: 2, pp: 1 }

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

  extra_args: ""                  # appended to every vLLM serve invocation

dynamo:
  host: 127.0.0.1
  port: 8000
  router_mode: round-robin        # round-robin | least-loaded | kv
  nats_url: nats://127.0.0.1:4222 # NATS endpoint for worker discovery (must be reachable)

opencode:
  host: 127.0.0.1
  port: 4096
  experimental_workspaces: true   # → OPENCODE_EXPERIMENTAL_WORKSPACES=true
```

There is **no `runner:` section**. Runner-side defaults (`num_samples=10`, `qps=0.5`, `seed=42`) live in `cli.py`. CLI flag > env override > yaml default.

### Worker role injection (kv_role + disaggregation_mode)

Each worker needs role-specific Dynamo/vLLM args:
- entries under `prefill_workers` → `kv_role=kv_producer`, `disaggregation_mode=prefill`
- entries under `decode_workers` → `kv_role=kv_consumer`, `disaggregation_mode=decode`

`testbed.sh` injects these at launch. **Exact arg names and JSON shape are not pinned in this doc** — they depend on the Dynamo version vendored in `dynamo/`. The launch script reads the actual flag schema from the vendored source (and `dynamo/<binary> --help`) so the testbed tracks whatever version is checked in.

### Discovery: NATS (chosen for distributed deploy)

NATS was picked over file-based discovery because file-based only works when Dynamo and all workers share a filesystem. NATS scales naturally across nodes — every worker connects to `dynamo.nats_url`, registers, and Dynamo's frontend subscribes. For local single-node use, run `nats-server` on `127.0.0.1:4222` once; for multi-node, point all `host` entries' deploys at a NATS reachable from each.

NATS is treated as an **external prerequisite**: `testbed.sh up` does not start it. `testbed.sh up nats` is provided as a single-node convenience that runs `nats-server` locally with default config.

## Lifecycle: one script, four verbs

`deploy/testbed.sh` is the only thing you need to remember. It is self-contained — no `_lib.sh`, no Makefile, no docker-compose.

```
deploy/testbed.sh up     [nats|workers|frontend|opencode|all]   # default: all (= workers + frontend + opencode)
deploy/testbed.sh down   [nats|workers|frontend|opencode|all]   # default: all
deploy/testbed.sh status
deploy/testbed.sh logs   <component>
```

All PID files and component logs are written to **`./logs/`** (created relative to wherever `testbed.sh` is invoked). PGID-based teardown logic and port-kill backstops are inlined into the script.

`up workers` reads `vllm.prefill_workers` / `vllm.decode_workers` from `testbed.yaml`. For each worker, the script:

1. validates `len(split(gpus, ',')) == tp * pp`,
2. exports `VLLM_NIXL_SIDE_CHANNEL_HOST=<worker.host>`, `VLLM_NIXL_SIDE_CHANNEL_PORT=<nixl_port_base + rank*100>`, `CUDA_VISIBLE_DEVICES=<gpus>`,
3. injects role-specific Dynamo args (kv_role / disaggregation_mode — see above),
4. passes `--max-model-len`, `--max-num-batched-tokens`, `--max-num-seqs`, `--gpu-memory-utilization`, `--kv-cache-dtype`, `--kv-transfer-config` (with `kv_connector`) plus `vllm.extra_args`,
5. spawns the worker locally. (Multi-node SSH spawn is TBD; for now keep all worker hosts at 127.0.0.1.)

`up frontend` starts `dynamo.frontend` on `dynamo.port` with `--router-mode` from yaml, pointing discovery at `dynamo.nats_url`. Assumes NATS is reachable.

`up opencode` renders `opencode/opencode.json` from the template, then runs `OPENCODE_EXPERIMENTAL_WORKSPACES=true bun dev serve --host <host> --port <port>` from inside the vendored `opencode/` directory. (TBD: exact `bun dev serve` flags and the templated variables in `opencode.json.tmpl` are derived from `opencode/`'s docs.)

`up` with no arg brings up `workers → frontend → opencode` in order; `down` reverses.

## Build (TBD)

Vendored sources are built per their own install guides:
- **`opencode/`** — bun-based. Follow `opencode/README.md`. Runtime entrypoint: `bun dev serve`.
- **`dynamo/`** — must be built with the **vLLM backend** selected. Follow `dynamo/README.md` (or its `BUILD.md` equivalent), passing whichever feature flag the vendored version uses to enable the vLLM backend.

Concrete build commands will be folded into this doc once each vendor folder is analyzed and pinned.

## Running a workload

```
.venv/bin/python -m testbed run \
  --split lite --num-samples 20 --qps 0.5 --seed 42 \
  --out results/run1
```

CLI defaults (set in `cli.py`): `num_samples=10`, `qps=0.5`, `seed=42`, `split=lite`. Override via flags or `TESTBED__*` env vars.

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

1. Compute `session_id = "session-<instance_id>-<short_uuid>"`.
2. **Pre-clone the repo**: `git clone <sample.repo> <workspace_root>/<session_id>` and `git -C <...> checkout <sample.base_commit>`. Synchronous, fail-fast — done by runner before any OpenCode call. This sidesteps the in-agent `git clone` hang.
3. POST `/session?directory=<session_id>`, then POST `/session/:id/message?directory=<session_id>` (synchronous; blocks until the agent loop finishes). RTT measured with `time.monotonic()`.
4. **Always GET `/session/:id/message?directory=<session_id>` after** — the synchronous POST response carries only the FINAL assistant message; intermediate tool-loop steps are only available via the list endpoint.
5. Write a TaskRecord with the raw message dump and basic metadata.

OpenCode behavior when `?directory=<id>` points to an **already-existing** pre-cloned dir under EXPERIMENTAL_WORKSPACES needs to be verified against the vendored `opencode/` source — the assumption here is that it accepts and uses the existing dir as-is.

## Concurrency, errors, cleanup

**Concurrency**: Poisson generates arrival timestamps; runner fires each task at its arrival via `asyncio.create_task` and gates them with a bounded semaphore. Default `max_in_flight = 16`, set in `cli.py`, overridable via `--max-in-flight`. The semaphore is acquired **after** arrival time (so it represents queueing on top of the system, not an artificial backpressure on the arrival process itself).

**Error policy**: fail-fast per task, no retry. The run as a whole never aborts on per-task failures.

| Failure point                                  | TaskRecord written?     | `success` | `rtt_s`               | `error.stage` |
|------------------------------------------------|-------------------------|-----------|-----------------------|---------------|
| `git clone` / `git checkout`                   | yes                     | false     | null                  | `clone`       |
| `POST /session` (non-2xx or timeout)           | yes                     | false     | null                  | `session`     |
| `POST /session/:id/message` (non-2xx/timeout)  | yes                     | false     | wall-clock to failure | `message`     |
| `GET /session/:id/message` (after good POST)   | yes                     | true      | from the POST         | `list`        |

`error.stage = "list"` means RTT is valid but `messages` may be empty/partial.

**Cleanup**: none. `<workspace_root>/<session_id>` directories accumulate across runs. Prune manually (`rm -rf /tmp/testbed-workspaces/session-*`) between large runs. Per-task cleanup is intentionally avoided so failing runs can be inspected.

## Output files (one run, `--out results/<dir>/`)

- `config.json` — invocation parameters (split, num_samples, qps, seed, max_in_flight, **router**, model, resolved `testbed.yaml` snapshot)
- `trace.jsonl` — one TaskRecord per line:
  ```json
  {
    "instance_id": "django__django-12345",
    "session_id": "session-django__django-12345-a1b2c3d4",
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

## Configuration overrides

`.env` (copy from `.env.example`) — secrets only:
```
OPENCODE_SERVER_PASSWORD=...
DYNAMO_API_KEY=...
HF_TOKEN=...
```

Anything in `testbed.yaml` can be overridden by a `TESTBED__<section>__<key>` env var (e.g. `TESTBED__VLLM__PREFILL__GPU_MEMORY_UTILIZATION=0.85`).

Resolution precedence: **CLI flag > `TESTBED__*` env var > `testbed.yaml` > built-in default**.

## Conventions / gotchas

- **Vendored sources are authoritative for any implementation detail.** When in doubt about a CLI flag, an env var, or a wire-format detail of OpenCode/Dynamo/vLLM, read the vendored source (`opencode/`, `dynamo/`) — do not rely on memory or external docs that may not match the pinned version.
- **OpenCode is headless-only**, launched as `OPENCODE_EXPERIMENTAL_WORKSPACES=true bun dev serve` from `opencode/`. No TUI, no shared global workspace. Every session/message call must include `?directory=<unique>`. Without the env var, concurrent agents will trample each other in a single CWD.
- **Runner pre-clones; agent does not.** SWE-bench tasks have repo+base_commit. Runner does the clone+checkout synchronously before opening the OpenCode session. The agent operates on a pre-prepared checkout, never `git clone`s itself (this has been observed to hang the agent loop).
- **System prompt lives on the user message** (`info.system`), and OpenCode types it as `string[]`. Consumers should normalize lists to `"\n\n".join(...)`.
- **Each PD worker needs a unique NIXL side-channel port.** vLLM defaults all workers to 5600 and they collide on a single host. `testbed.sh` exports `VLLM_NIXL_SIDE_CHANNEL_HOST` (per worker, from `worker.host`) and `VLLM_NIXL_SIDE_CHANNEL_PORT` (`nixl_port_base + rank*100`). Do not rely on vLLM defaults.
- **kv_role + disaggregation_mode are derived, not configured.** Workers in `prefill_workers` get producer/prefill role injection; `decode_workers` get consumer/decode. Exact flag names come from parsing the vendored Dynamo, not from this doc.
- **NATS is external.** For multi-node deploys, NATS runs as its own service. `testbed.sh up nats` exists only as a single-node convenience.
- **PGID-based teardown** is necessary because OpenCode's `.opencode` worker and vLLM's TP/PP shards `setsid` out of the parent. The logic is inlined in `testbed.sh` (no `_lib.sh`). `down opencode` also runs `kill_port` on `opencode.port` as a backstop.
- **Single config file invariant**: do NOT introduce a second config source (e.g. a separate `workers.env`). If a new knob is needed, add it to `testbed.yaml` and teach `config.py` + `testbed.sh` to read it.
- **Logs path is fixed**: `./logs/` (relative to `testbed.sh`'s CWD). PID files: `./logs/<component>.pid`. Stdout/stderr: `./logs/<component>.log`.

## Editing rules

- README.md is user-facing; CLAUDE.md is for future agent sessions. Keep them in sync only on user-visible commands; internal invariants belong here.
- New CLI flags go into `cli.py`.
- Schema changes to `testbed.yaml` require a matching update to `config.py` validation and the docs at the top of the yaml file.
- When something in this doc says "TBD" or "exact ... derived from vendored source", that's a contract to read the vendor folder rather than guess. Do not pin a value here without first checking the vendored source matches.

## Branch

Active development branch: `claude/setup-agent-scheduler-tests-Mzol4`. Push directly here unless told otherwise.