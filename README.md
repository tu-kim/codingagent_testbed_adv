# testbed

Drives [SWE-bench](https://www.swebench.com/) tasks through an [OpenCode](https://github.com/anomalyco/opencode) headless agent against a [Dynamo](https://github.com/ai-dynamo/dynamo) OpenAI-compatible frontend whose backend is a vLLM PD-disaggregated worker pool. Used to measure router and scheduling decisions under realistic coding-agent workloads.

See `CLAUDE.md` for the full architecture, contracts, and gotchas.

## Quickstart

```bash
# 1. Install Python deps.
uv pip install -e ".[dev]"

# 2. Configure (copy and edit).
cp .env.example .env
$EDITOR deploy/testbed.yaml

# 3. Bring the stack up. Submodule build (opencode/, dynamo/) is a prerequisite — see CLAUDE.md.
deploy/testbed.sh up        # workers + frontend + opencode (NATS/etcd are external)

# 4. Run a workload.
.venv/bin/python -m testbed run \
  --split lite --num-samples 20 --qps 0.5 --seed 42 \
  --out results/run1

# 5. Tear down.
deploy/testbed.sh down
```

Smoke-test individual components:

```bash
scripts/curl_smoke.sh routes      # OpenCode OpenAPI paths
scripts/curl_smoke.sh dynamo      # Dynamo /v1/chat/completions
scripts/curl_smoke.sh opencode    # full session+message round trip
scripts/curl_smoke.sh swebench    # SWE-bench prompt round trip
```

Configuration overrides resolve as **CLI flag > `TESTBED__*` env var > `deploy/testbed.yaml` > built-in default**.

## Layout

```
src/testbed/      # python: cli, runner, config, poisson, swebench, opencode client
deploy/           # testbed.yaml, testbed.sh, opencode.json.tmpl
scripts/          # curl_smoke.sh
opencode/         # vendored opencode source (submodule)
dynamo/           # vendored dynamo source (submodule)
tests/            # pytest, no network
```
