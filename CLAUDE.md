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
    # POST /session?directory=<abs_dir> with body {}.
    # Returns the SERVER-ASSIGNED session id (regex ^ses.* per the SDK schema —
    # we don't pick the id; we use it as :sessionID in subsequent calls).

async def send_message(session_id: str, prompt: str, directory: str) -> dict
    # POST /session/:id/message?directory=<abs_dir> with body
    #   {"parts":[{"type":"text","text":prompt}]}.
    # Blocks until agent loop completes; returns the raw JSON envelope
    # ({info, parts}) — only the FINAL assistant message — see per-task flow.

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

vllm:
  kv_connector: NixlConnector     # vLLM kv-transfer connector class name (matches what vLLM expects)
  nixl_port_base: 6000            # rank N → VLLM_NIXL_SIDE_CHANNEL_PORT = base + N*100
  tool_call_parser: qwen3_coder   # → --dyn-tool-call-parser on DECODE workers only ("" disables)

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
  host: 127.0.0.1                 # → --http-host on dynamo.frontend
  port: 8000                      # → --http-port on dynamo.frontend
  router_mode: round-robin        # round-robin | least-loaded | kv | random | power-of-two | direct | device-aware-weighted
  discovery_backend: etcd         # → --discovery-backend (kubernetes | etcd | file | mem)
  etcd_endpoints: http://127.0.0.1:2379   # → ETCD_ENDPOINTS env on every dynamo process
  nats_url: nats://127.0.0.1:4222         # → NATS_SERVER env (request/event plane, NOT discovery)
  request_plane: tcp              # → --request-plane (tcp | nats | http)
  event_plane: nats               # → --event-plane (nats | zmq)

opencode:
  host: 127.0.0.1
  port: 4096
  experimental_workspaces: true   # → OPENCODE_EXPERIMENTAL_WORKSPACES=true
```

There is **no `runner:` section**. Runner-side defaults (`num_samples=10`, `qps=0.5`, `seed=42`) live in `cli.py`. CLI flag > env override > yaml default.

### Worker role injection (kv_role + disaggregation_mode + tool_call_parser)

Each worker needs role-specific Dynamo/vLLM args. `testbed.sh` injects these at launch:
- `prefill_workers[]` → `--disaggregation-mode prefill` plus `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'`
- `decode_workers[]`  → `--disaggregation-mode decode` plus `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'` plus (if `vllm.tool_call_parser` is non-empty) `--dyn-tool-call-parser <name>`

The flag schema and connector class name come from the vendored Dynamo (`dynamo/components/src/dynamo/vllm/{args,backend_args}.py`). The decode-only tool-call-parser branch is enforced by `dynamo/components/src/dynamo/vllm/main.py:647-650` (`if model_type != ModelType.Prefill: runtime_config.tool_call_parser = ...`); applying the parser on prefill is a no-op but a smell.

### Discovery vs. NATS

Two separate concerns:
- **Discovery** is how the frontend learns about workers. Default `--discovery-backend etcd`; choices: `kubernetes | etcd | file | mem`. Set `ETCD_ENDPOINTS` on every dynamo process. (CLAUDE.md previously said "discovery via NATS" — that was wrong; NATS does not appear in the discovery enum.)
- **Request/Event plane** is how requests + KV events flow once workers are discovered. NATS is the default event plane and an option for the request plane. Set `NATS_SERVER` on every dynamo process.

Both etcd and NATS are treated as **external prerequisites**. `testbed.sh up etcd` and `testbed.sh up nats` exist as single-node conveniences only.

## Lifecycle: one script, four verbs

`deploy/testbed.sh` is the only thing you need to remember. It is self-contained — no `_lib.sh`, no Makefile, no docker-compose.

```
deploy/testbed.sh up     [nats|etcd|workers|frontend|opencode|all]   # default: all (= workers + frontend + opencode)
deploy/testbed.sh down   [nats|etcd|workers|frontend|opencode|all]   # default: all
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

`up frontend` starts `dynamo.frontend` with `--http-host`, `--http-port`, `--router-mode`, `--discovery-backend`, `--request-plane`, and `--event-plane` from yaml, exporting `NATS_SERVER` and `ETCD_ENDPOINTS` into the child env. Assumes NATS and etcd (or whichever discovery backend is configured) are reachable.

`up opencode` renders `opencode/opencode.json` from the template, then runs `OPENCODE_EXPERIMENTAL_WORKSPACES=true bun run dev serve --hostname <host> --port <port>` from inside the vendored `opencode/` directory. (`bun dev` is the bun shorthand for `bun run dev`; the actual flag is `--hostname`, not `--host`, per `opencode/packages/opencode/src/cli/network.ts`.)

`up` with no arg brings up `workers → frontend → opencode` in order; `down` reverses.

## Prerequisites (host install)

These must be on `PATH` before `testbed.sh up` will work. The exact commands
live in `README.md` under "Prerequisites" — do not duplicate them here, just
the rationale for each pin:

| Tool | Pin | Why |
|------|-----|-----|
| `yq` | apt | `testbed.sh` reads `testbed.yaml` via `yq` (kislyuk's Python yq, jq filters) |
| `nats-server` | apt | `testbed.sh up nats` single-node convenience (port 4222) |
| `etcd` / `etcdctl` | tarball v3.5.17 → `/usr/local/bin` | apt etcd is too old; Dynamo's default `--discovery-backend` is etcd |
| `vllm` | pip `==0.19.0` | matches Dynamo 1.1's vendored AsyncEngineArgs surface; bumping vLLM may break flag pass-through in `dynamo.vllm` |
| `nixl` | pip | KV-transfer connector class vLLM dlopen's when `--kv-transfer-config` selects `NixlConnector` |

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

1. Compute `directory = f"session-<instance_id>-<short_uuid>"` (the workspace folder name; **not** the OpenCode session id) and `abs_dir = workspace_root / directory`.
2. **Pre-clone the repo**: `git clone <sample.repo> <abs_dir>` and `git -C <abs_dir> checkout <sample.base_commit>`. Synchronous, fail-fast — done by runner before any OpenCode call. This sidesteps the in-agent `git clone` hang.
3. POST `/session?directory=<abs_dir>` with body `{}` → server returns the assigned `session_id` (matches `^ses.*`). Then POST `/session/<session_id>/message?directory=<abs_dir>` (synchronous; blocks until the agent loop finishes). RTT measured with `time.monotonic()`. The directory query value MUST be the absolute path — see Module contracts above.
4. **Always GET `/session/<session_id>/message?directory=<abs_dir>` after** — the synchronous POST response carries only the FINAL assistant message; intermediate tool-loop steps are only available via the list endpoint.
5. Write a TaskRecord with the raw message dump and basic metadata. The TaskRecord stores `directory` (the folder NAME, relative — for human readability) and `session_id` (server-assigned) as separate fields. The absolute path is recoverable as `<workspace_root>/<directory>`.

OpenCode accepts a `?directory=` that points to an already-existing pre-cloned dir under `OPENCODE_EXPERIMENTAL_WORKSPACES=true` — verified against the vendored middleware (it just `path.resolve()`s the value and uses the resulting directory as the agent's working directory).

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
- **OpenCode is headless-only**, launched as `OPENCODE_EXPERIMENTAL_WORKSPACES=true bun run dev serve --hostname <h> --port <p>` from `opencode/`. No TUI, no shared global workspace. Every session/message call must include `?directory=<absolute path>` (`InstanceMiddleware` reads it). Without the env var, concurrent agents will trample each other in a single CWD. **Always send the absolute path** — `InstanceMiddleware` calls `AppFileSystem.resolve()` (Node `path.resolve()`) so a bare folder name resolves against OpenCode's CWD (= the `opencode/` repo), not against `workspace_root`.
- **Pass `OPENCODE_CONFIG=<abs path>` when launching OpenCode.** OpenCode's per-instance config loader walks UP from the request's `?directory=` looking for `opencode.json` (`opencode/packages/opencode/src/config/paths.ts:10-21`, consumed at `config/config.ts:568-572`). Our rendered `opencode/opencode.json` is NOT on that walk path because `?directory=` lives under `/tmp/testbed-workspaces/<session>/`. Without `OPENCODE_CONFIG=<abs>` (loaded as "custom config" at `config.ts:563-566`), the rendered file is silently ignored, `cfg.provider` is empty, and `HF_TOKEN` (commonly exported for vLLM) auto-enables the `huggingface` provider via `provider.ts:160-181` (`input.env.some(item => env[item])`) — sending all inference to `https://router.huggingface.co/v1` with whatever `qwen/...` model happens to be in the `huggingface` block of `models.dev/api.json`. Belt-and-suspenders: `opencode.json.tmpl` also pins `disabled_providers: ["huggingface"]` (`provider.ts:1133` consumes this).
- **OpenCode auth is HTTP Basic** when `OPENCODE_SERVER_PASSWORD` is set on the server. Username defaults to `"opencode"` (override via `OPENCODE_SERVER_USERNAME` server-side). The client sends `Authorization: Basic <base64(user:pass)>`. There is no `x-opencode-server-password` header — do not invent one.
- **Runner pre-clones; agent does not.** SWE-bench tasks have repo+base_commit. Runner does the clone+checkout synchronously before opening the OpenCode session. The agent operates on a pre-prepared checkout, never `git clone`s itself (this has been observed to hang the agent loop).
- **System prompt lives on the user message** (`info.system`), and OpenCode types it as `string[]`. Consumers should normalize lists to `"\n\n".join(...)`.
- **Each PD worker needs a unique NIXL side-channel port.** vLLM defaults all workers to 5600 and they collide on a single host. `testbed.sh` exports `VLLM_NIXL_SIDE_CHANNEL_HOST` (per worker, from `worker.host`) and `VLLM_NIXL_SIDE_CHANNEL_PORT` (`nixl_port_base + rank*100`). Do not rely on vLLM defaults.
- **kv_role + disaggregation_mode are derived, not configured.** Workers in `prefill_workers` get `--disaggregation-mode prefill` and `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'`; `decode_workers` get the `decode`/`kv_consumer` variant.
- **etcd and NATS are external.** Discovery is etcd by default; NATS is the request/event plane. `testbed.sh up etcd` and `testbed.sh up nats` are single-node conveniences only.
- **`bun dev` vs `bun run dev`**: `bun dev` is the bun shorthand and `bun run dev` is the explicit form; both invoke the `dev` script from `opencode/package.json`. `--hostname` (not `--host`) is the OpenCode flag — see `opencode/packages/opencode/src/cli/network.ts`.
- **OpenCode session ids are server-generated** and match `^ses.*` (per the SDK schema). The runner stores them as `session_id` in the trace; `directory` is the runner-chosen workspace folder name.
- **PGID-based teardown** is necessary because OpenCode's `.opencode` worker and vLLM's TP/PP shards `setsid` out of the parent. The logic is inlined in `testbed.sh` (no `_lib.sh`). `down opencode` also runs `kill_port` on `opencode.port` as a backstop.
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