# testbed

Drives [SWE-bench](https://www.swebench.com/) tasks through an [OpenCode](https://github.com/anomalyco/opencode) headless agent against a [Dynamo](https://github.com/ai-dynamo/dynamo) OpenAI-compatible frontend whose backend is a vLLM PD-disaggregated worker pool. Used to measure router and scheduling decisions under realistic coding-agent workloads.

```
SWE-bench sample → runner.py (Poisson or sequential) → OpenCode server (:4096)
                     → Dynamo frontend (:8000/v1) → vLLM prefill+decode workers (KV via NIXL)
```

This README is the **usage manual**. For architecture, module contracts, and gotchas see `CLAUDE.md`.

---

## 1. Prerequisites (once per host)

System packages and binaries expected on `PATH` before `deploy/testbed.sh up`.

```bash
# yq — testbed.sh reads testbed.yaml via yq. nats-server — single-node
# convenience for `deploy/testbed.sh up nats`.
sudo apt install -y yq nats-server

# vLLM pinned to 0.19.0 (matches the AsyncEngineArgs surface vendored Dynamo
# 1.1 was built against); NIXL is the KV-transfer connector vLLM loads when
# --kv-transfer-config selects NixlConnector.
uv pip install vllm==0.19.0
uv pip install nixl

# etcd (Dynamo's default --discovery-backend); apt's is too old.
ETCD_VER=v3.5.17
curl -fsSL https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz -o /tmp/etcd.tar.gz
tar -xzf /tmp/etcd.tar.gz -C /tmp
sudo mv /tmp/etcd-${ETCD_VER}-linux-amd64/etcd /tmp/etcd-${ETCD_VER}-linux-amd64/etcdctl /usr/local/bin/
rm -rf /tmp/etcd.tar.gz /tmp/etcd-${ETCD_VER}-linux-amd64

# Sanity check — every line should print a version.
yq --version && nats-server --version && etcd --version && python -c "import vllm, nixl; print(vllm.__version__)"
```

Vendored submodules (`opencode/`, `dynamo/`) are built per their own install guides — see CLAUDE.md.

---

## 2. Setup

```bash
# Python deps for the testbed itself.
uv pip install -e ".[dev]"

# Submodules + testbed-owned patches (idempotent; see §7).
git submodule update --init opencode dynamo
scripts/apply_opencode_patches.sh        # profiling hooks (OPENCODE_PROFILE)
scripts/apply_dynamo_patches.sh          # per-request scheduling-delay logging

# Config. Both targets are gitignored.
cp .env.example .env                      # secrets only (passwords, HF_TOKEN, ...)
cp deploy/testbed.yaml.example deploy/testbed.yaml
$EDITOR deploy/testbed.yaml               # workspace, model, vLLM PD, dynamo, opencode
```

Configuration resolves as **CLI flag > `TESTBED__*` env var > `deploy/testbed.yaml` > built-in default**. Any yaml key is overridable, e.g. `TESTBED__DYNAMO__ROUTER_MODE=kv`.

---

## 3. Lifecycle (`deploy/testbed.sh`)

```bash
deploy/testbed.sh up   [nats|etcd|workers|frontend|opencode|monitor|scrape_metrics|all]
deploy/testbed.sh down [nats|etcd|workers|frontend|opencode|monitor|scrape_metrics|all]
deploy/testbed.sh status
deploy/testbed.sh logs <component>        # tail -F logs/<component>.log
```

- `up`/`down` with no target = `all` = **workers + frontend + opencode** (in order / reverse).
- `monitor` and `scrape_metrics` are **opt-in** — never part of `all`, bring up/down separately.
- etcd + NATS are external prerequisites; `up etcd` / `up nats` are single-node conveniences.
- All PID + log files live under `./logs/`.

```bash
# Typical bring-up
deploy/testbed.sh up nats
deploy/testbed.sh up etcd
deploy/testbed.sh up            # workers → frontend → opencode
deploy/testbed.sh status
```

Smoke-test slices without a full run:
```bash
scripts/curl_smoke.sh routes      # OpenCode OpenAPI paths
scripts/curl_smoke.sh dynamo      # one Dynamo /v1/chat/completions
scripts/curl_smoke.sh opencode    # full session+message round trip
scripts/curl_smoke.sh swebench    # a real SWE-bench prompt
```

---

## 4. Running a workload

```bash
.venv/bin/python -m testbed run \
  --split lite --num-samples 20 --qps 0.5 --seed 42 \
  --out results/run1
```

`run` flags (defaults in parentheses):

| flag | default | meaning |
|------|---------|---------|
| `--split` | `lite` | SWE-bench split: `lite` \| `verified` \| `full` |
| `--num-samples` | `10` | number of tasks (deterministic pick from `(split, seed, n)`) |
| `--qps` | `0.5` | Poisson arrival rate (ignored with `--sequential`) |
| `--seed` | `42` | sampling + arrival seed |
| `--max-in-flight` | `16` | bounded concurrency (ignored with `--sequential`) |
| `--task-timeout-s` | `300` | per-task cap on the agent loop; `<=0` disables |
| `--reset-workspace` | off | deterministic per-instance workspace dir, reset to `base_commit` each task |
| `--sequential` | off | strictly one request at a time; bypasses Poisson |
| `--router` | `""` | label recorded in `config.json` only (does NOT change routing) |
| `--out` | required | output directory |

Progress prints per task to **stderr** (`[done/total] instance_id ok/FAIL rtt=.. elapsed=..`).

**Router sweep** (actual routing is set when the frontend starts, not by `--router`):
```bash
for r in round-robin least-loaded kv; do
  TESTBED__DYNAMO__ROUTER_MODE=$r deploy/testbed.sh down frontend
  TESTBED__DYNAMO__ROUTER_MODE=$r deploy/testbed.sh up   frontend
  .venv/bin/python -m testbed run --num-samples 20 --router $r --out results/$r
done
```

### Output files (`--out results/<dir>/`)
- `config.json` — resolved invocation params + `testbed.yaml` snapshot
- `trace.jsonl` — one TaskRecord per line (`instance_id`, `session_id`, `rtt_s`, `success`, `error`, raw `messages`)
- `summary.json` — `rtt_s` p50/p95, `success_rate`, `count`

---

## 5. Reproducibility

Agent-loop output can diverge run-to-run; the knobs that remove the controllable sources:

```bash
# Cleanest per-task reproducibility: one request at a time + byte-stable workspace.
.venv/bin/python -m testbed run \
  --split lite --num-samples 20 --seed 42 \
  --sequential --reset-workspace \
  --out results/repro1
```

- `--sequential` removes concurrent-batching and timing-induced ordering variance.
- `--reset-workspace` makes the workspace path + state byte-stable (opencode embeds cwd in its system prompt).
- Greedy decoding + seeds + `repetition_penalty: 1.0` are pinned in `testbed.yaml` (see CLAUDE.md "Reproducibility").

Verify N runs reached the same result (see §6 `compare_traces.py`):
```bash
for i in 1 2 3; do
  .venv/bin/python -m testbed run --split lite --num-samples 20 --seed 42 \
    --sequential --reset-workspace --out results/repro$i
done
scripts/compare_traces.py --traces results/repro{1,2,3}/trace.jsonl \
  --output results/repro_cmp --figures
echo "exit=$?"   # 3 ⇒ at least one session's final answer diverged
```

---

## 6. Analysis

All analyzers are standalone (`scripts/*.py`), write CSVs (+ optional `--figures` PDFs) under `--output`, and print a summary. Pass `--profile`/`--logs`/`--trace` etc. depending on the source.

| script | input | what it answers |
|--------|-------|-----------------|
| `compare_traces.py` | N × `trace.jsonl` | reproducibility across runs: per-session turns/tokens/**final answer/code diff**, status REPRODUCIBLE / TRAJ_DIFF_SAME_ANSWER / ANSWER_DIFF. exit 3 on any answer divergence. `--figures` |
| `analyze_trace_parallelism.py` | `trace.jsonl` | parallel tool calls + sub-agent (`task`) spawns per step |
| `analyze_frontend_log.py` | `logs/frontend.log` | per-request e2e / TTFT / ITL / ISL-OSL distributions + figures |
| `analyze_profiles.py` | profile NDJSON dir | paper figures: session E2E, per-tool time, turn decomposition |
| `analyze_idle_time.py` | profile NDJSON | multi-turn wall split: busy vs bootstrap / **inter-turn idle** / teardown |
| `analyze_subagent_time.py` | profile NDJSON | per-turn share of time spent in `task` sub-agents (union of intervals) |
| `analyze_tool_time.py` | profile NDJSON | per-turn share of time in non-`task` tools, broken down by tool name |
| `analyze_session_resources.py` | `logs/resource.ndjson` | GPU/CPU stats per session window; window-aggregate `{mean,min,max}`; PCIe/NVLink as per-window deltas; prefill/decode role split |
| `analyze_vllm_metrics.py` | `logs/vllm_metrics.ndjson` | vLLM `/metrics` stats: gauges, counter rates, histogram percentiles, per worker/role |
| `analyze_worker_scheduling.py` | `logs/vllm-*.log` | per-request prefill/decode **scheduling delay** (needs dynamo patch, §7) |
| `analyze_request_wait.py` | `frontend.log` + `logs/` | queue-wait as a **fraction of e2e**, joined by request_id; tail concentration; `--figures` |

Collectors (run alongside a workload, opt-in):
```bash
deploy/testbed.sh up scrape_metrics   # vLLM /metrics → logs/vllm_metrics.ndjson  (no sudo)
sudo DCGM_PY=/path/to/venv/python deploy/testbed.sh up monitor   # DCGM GPU + psutil → logs/resource.ndjson
```
`monitor` needs `sudo` and `monitor.dcgm_py` set in `testbed.yaml` (read from yaml so sudo's env-strip doesn't lose it).

OpenCode profiling is ENV-gated: launch opencode with `OPENCODE_PROFILE=1` (per-session NDJSON lands in `<workspace_root>/profiles/`). Aggregate with `scripts/aggregate_profiles.sh <workspace_root>`.

Utilities: `jsonl_to_json.py` (NDJSON→JSON), `trim_idle_tail.py`, `sse_chunk_timing.py` (single-stream chunk timing), `view_trace.sh`.

---

## 7. Vendored-submodule patches (`deploy/patches/`)

Submodules stay pinned; testbed-owned changes are patches applied via prefix-routed scripts (so the two submodules never cross-apply):

```bash
scripts/apply_opencode_patches.sh [--check|--revert]   # opencode-*.patch
scripts/apply_dynamo_patches.sh   [--check|--revert]   # dynamo-*.patch
```

- `opencode-profile.patch` — ENV-gated step-event profiler (`OPENCODE_PROFILE=1`).
- `dynamo-scheduling-log.patch` — per-request scheduler queue-wait logged to the worker log (`SCHED_DELAY ...`), read by `analyze_worker_scheduling.py` / `analyze_request_wait.py`. **Python-only — no cargo rebuild**, just restart workers after applying.

---

## 8. Layout

```
src/testbed/      cli, runner, config, poisson, swebench, opencode client
deploy/           testbed.yaml.example, testbed.sh, opencode.json.tmpl, patches/
scripts/          analyzers (analyze_*.py, compare_traces.py), collectors, apply_*_patches.sh, curl_smoke.sh
opencode/         vendored opencode (submodule)
dynamo/           vendored dynamo (submodule)
tests/            pytest, no network (mocks only)
logs/             PID + component logs (created at runtime)
```

---

## 9. Tests

```bash
pytest                       # no network; mocks only
pytest tests/test_config.py  # a single module
```

`tests/test_*_interface.py` and `test_opencode_template.py` are drift detectors — they fail when a vendored submodule renames a flag/field the testbed depends on. If one fails, read the vendored source before patching the test.
