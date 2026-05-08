# testbed

Drives [SWE-bench](https://www.swebench.com/) tasks through an [OpenCode](https://github.com/anomalyco/opencode) headless agent against a [Dynamo](https://github.com/ai-dynamo/dynamo) OpenAI-compatible frontend whose backend is a vLLM PD-disaggregated worker pool. Used to measure router and scheduling decisions under realistic coding-agent workloads.

See `CLAUDE.md` for the full architecture, contracts, and gotchas.

## Prerequisites

System packages and binaries the testbed shell glue and the vendored stacks
expect to find on `PATH`. Run once per host before `deploy/testbed.sh up`.

```bash
# yq — testbed.sh reads testbed.yaml via yq. nats-server — single-node
# convenience target for `deploy/testbed.sh up nats`.
sudo apt install -y yq nats-server

# vLLM and NIXL into the active venv. vLLM is pinned to 0.19.0 to match the
# AsyncEngineArgs surface vendored Dynamo (1.1) was built against; NIXL is
# the KV-transfer connector vLLM loads when --kv-transfer-config selects
# NixlConnector.
uv pip install vllm==0.19.0
uv pip install nixl

# etcd binary install (Dynamo's default --discovery-backend). The apt package
# is too old on most distros; install upstream tarball into /usr/local/bin.
ETCD_VER=v3.5.17
curl -fsSL https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz -o /tmp/etcd.tar.gz
tar -xzf /tmp/etcd.tar.gz -C /tmp
sudo mv /tmp/etcd-${ETCD_VER}-linux-amd64/etcd     /usr/local/bin/
sudo mv /tmp/etcd-${ETCD_VER}-linux-amd64/etcdctl  /usr/local/bin/
rm -rf /tmp/etcd.tar.gz /tmp/etcd-${ETCD_VER}-linux-amd64

# Sanity check — every line should print a version, not "command not found".
yq --version && nats-server --version && etcd --version && python -c "import vllm, nixl; print(vllm.__version__)"
```

The vendored submodules (`opencode/`, `dynamo/`) are built per their own
install guides; see CLAUDE.md.

## Quickstart

```bash
# 1. Install Python deps for the testbed itself.
uv pip install -e ".[dev]"

# 2. Configure (copy and edit).
cp .env.example .env
$EDITOR deploy/testbed.yaml

# 3. Bring up the dependencies first (NATS + etcd), then the stack.
deploy/testbed.sh up nats     # single-node convenience for nats-server
deploy/testbed.sh up etcd     # single-node convenience for etcd
deploy/testbed.sh up          # workers → frontend → opencode

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
