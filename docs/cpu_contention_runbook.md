# CPU-contention stress runbook (mock LLM)

Goal: show that at very high `--max-in-flight`, the CPU side (opencode
scaffold + tool subprocesses) becomes the bottleneck. Against a real GPU
backend this is masked — raising in-flight first inflates KV-cache
pressure and TTFT, so requests queue GPU-side before CPU contention can
appear. `scripts/mock_llm_server.py` removes the GPU from the loop: an
OpenAI-compatible responder with configurable (near-zero) latency, so
concurrency lands directly on tool execution and scaffold work.

## Setup

```bash
# 1. profile patch (per-turn tool walls + post_stream_overhead)
scripts/apply_opencode_patches.sh

# 2. start the mock (from repo root so logs/ pid+ndjson land in ./logs/)
python scripts/mock_llm_server.py --port 8000 \
    --tool-turns 4 --tool-cmd 'ls' --ttft-ms 0 --itl-ms 0

# 3. point opencode at the mock instead of dynamo, then restart opencode
TESTBED__DYNAMO__HOST=127.0.0.1 TESTBED__DYNAMO__PORT=8000 \
    deploy/testbed.sh up opencode
# (or edit deploy/testbed.yaml dynamo.host/port; the rendered
#  opencode.json {{DYNAMO_BASE_URL}} is the only touchpoint)

# 4. resource sampler — tracks every *.pid in logs/, including mock_llm.pid
sudo bash deploy/testbed.sh up monitor    # needs monitor.dcgm_py in yaml
# GPU-less host: monitor requires DCGM today; if unavailable, capture CPU
# with a psutil one-liner loop or run on the GPU host anyway (GPUs idle).

# 5. runner sweep (workers/frontend NOT needed — mock replaces them)
OPENCODE_PROFILE=1  # must be set before `up opencode` (step 3)
for mif in 8 16 32 64 128; do
  .venv/bin/python -m testbed run --workload terminalbench \
      --num-samples 30 --qps 4 --max-in-flight $mif \
      --out results/cpu-contention-mif$mif
done
```

Notes:
- The mock ignores sampling params (`seed`, `temperature`) — fine.
- `--tool-cmd` controls how much CPU each tool turn burns. `ls` ≈ pure
  scaffold/subprocess-spawn cost; heavier commands (e.g.
  `python3 -c 'sum(range(10**7))'`) emulate compute-bound sandboxes.
- The mock always answers every session with the same N-tool-turn
  script, so per-task work is uniform — differences across `mif` come
  from contention, not workload variance.

## Run matrix

| axis | values |
|---|---|
| `--max-in-flight` | 8, 16, 32, 64, 128 |
| mock latency (ttft, itl) | (0, 0) · (200ms, 10ms) · (1000ms, 30ms) |
| fixed | `--qps 4`, `--num-samples 30`, terminalbench (tool-heavy) |

The latency axis re-introduces "LLM think time": at (0,0) in-flight
tasks stack up on tools almost continuously (max CPU pressure); the
realistic tier shows whether contention survives when LLM gaps give the
CPU breathing room.

## What to measure

| signal | source | contention signature |
|---|---|---|
| tool wall p50/p95 per mif | profile NDJSON `tool.end.duration_s` (`scripts/analyze_profiles.py`) | inflates with mif while injected mock latency is flat |
| injected vs observed LLM wall | mock `logs/mock_llm.ndjson` (`wall_s`, `injected_*`) vs profile `llm.end.duration_s` | client-side gap growing ⇒ scaffold/event-loop contention, not server |
| host CPU + load | `logs/resource.ndjson` `host.cpu_util_pct`, `host.load_1min` vs `n_cores` | `load_1min > n_cores`, cpu_util pegged |
| per-tree CPU | `resource.ndjson` `processes[]` (opencode tree, mock tree, runner tree) | opencode tree plateaus at core budget while demand rises |
| scaffold overhead | profile `turn.end.post_overhead_s`, `llm.end.post_stream_overhead_s` | grows with mif (bun event-loop/DB contention) |
| end-to-end | `trace.jsonl` `rtt_s` | superlinear growth vs mif once CPU saturates |

Null result: tool walls and post-overheads flat across mif ⇒ CPU
contention NOT demonstrated at these levels — raise qps/mif or use a
heavier `--tool-cmd` before concluding.

## Relation to goal 1 (small-turn offloading)

The same profile runs feed `scripts/analyze_turn_scheduling.py` on REAL
(dynamo) runs: per preceding-tool stats of output-tokens vs scheduler
queue-share identify which tools bracket small-LLM turns — the CPU
offloading candidates. The mock experiment quantifies the CPU-side
headroom/ceiling those offloaded turns would compete for.
