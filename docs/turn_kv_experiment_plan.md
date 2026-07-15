# Turn-transition KV loss → CPU-GPU unified scheduling: experiment plan

Story under test (each step measured before the next is argued):

1. vLLM manages KV by ref count; when turn *i* finishes, its blocks drop
   to the free-LRU queue (cached-but-free, NOT immediately erased).
2. Under high workload / high max-in-flight / long tool latency, other
   requests allocate KV while the session is away → the still-active
   session's blocks are displaced before turn *i+1* arrives. LRU is
   liveness-blind: a mid-loop session's KV can be evicted before a
   finished session's.
3. Hit ratio collapses with away time and displaced traffic → re-entry
   pays scheduler queue wait + re-prefill.
4. KVBM can preserve evicted KV in host/disk tiers — but only when the
   host tier meaningfully exceeds the GPU KV pool (undersized = thrash),
   and onboarding is ON-DEMAND at scheduling time (no prefetch; verified
   against dynamo source), so the host→device transfer rides inside TTFT.
5. For turns with tiny new-compute and tiny output ("small turns",
   identified per preceding tool), that full GPU toll (queue + onboard +
   re-prefill) buys almost no decode work → process them on the CPU with
   the host-resident KV instead.
6. All of it points to a CPU-GPU unified, turn-aware scheduler.

Onboard timing caveat: this dynamo tag records NO per-request transfer
duration (aggregate counters only) — onboard cost is inferred as
`prefill_compute_ms(kvbm) − prefill_compute_ms(baseline)` A/B.

---

## E0 — Characterize turns in isolation (`--sequential`)

Purpose: LLM-time distribution + token structure per turn WITHOUT any
eviction/concurrency interference → identify the small-turn population
(CPU-offload candidates) by (prev_tool, cur_tool).

```bash
OPENCODE_PROFILE=1  # set before `up opencode`
.venv/bin/python -m testbed run --workload terminalbench \
    --sequential --reset-workspace --num-samples 30 --out results/e0-seq
python3 scripts/analyze_turn_scheduling.py \
    --profiles <workspace_root>/profiles --logs logs/ --out results/e0-seq/turns
```

Data: `turn_sched.csv` (`prev_tools`, `cur_tools`, `input_tokens`,
`cache_read` → new-compute = input − cache-portion, `output_tokens`,
`llm_wall_s`), `by_tool.csv` small-share per tool.

## E1 — Miss under load (the problem exists)

Purpose: show hit-ratio collapse vs away time AND vs displaced traffic,
as a function of load. Sequential run from E0 = control (hit stays high).

```bash
for qps in 1 2 4; do
  .venv/bin/python -m testbed run --workload terminalbench \
      --qps $qps --max-in-flight 32 --num-samples 30 --out results/e1-q$qps
  python3 scripts/analyze_turn_scheduling.py \
      --profiles <ws>/profiles --logs logs/ --out results/e1-q$qps/turns
done
```

Data: `away_cache.csv` per run; `turn_sched.csv` columns `away_s`,
`away_displaced_tokens`, `cache_hit_ratio` + the two pearson r lines
(displaced-tokens r is the causal one — LRU displaces by allocation
pressure, not elapsed time). Watch `vllm:kv_cache_usage_perc` in
`vllm_metrics.ndjson` — the mechanism only operates near-full.
Liveness-blindness (story 2): compare hit of non-final turns (session
still active) — they miss anyway; per-turn rows support this directly.

## E2 — Who pays: small turns (cost/benefit asymmetry)

Purpose: per preceding-tool, output tokens are small but queue-share and
re-prefill are large → worst cost/benefit → offload candidates.

Same E1 runs; data: `by_tool.csv` (output_tokens p50 vs
prefill/decode_queue p50, queue_share small vs large conditional),
`turn_sched.csv` re-prefill = `input_tokens` (non-cached).

## E3 — KVBM: preservation and its price

Purpose: (a) undersized host tier = thrash (offload high, hit ≈ 0) —
negative control, already measured (20G: host 0.9%, disk 0%);
(b) adequately-sized tier (host ≫ GPU KV pool, disk ≥ host) recovers
hits; (c) the onboard cost appears in TTFT.

```bash
# baseline (kvbm off) and treatment (kvbm on), SAME seed/qps/samples:
TESTBED__VLLM__KVBM__ENABLED=false deploy/testbed.sh up workers ; run → results/e3-base
TESTBED__VLLM__KVBM__ENABLED=true  ... cpu_cache_gb ≫ GPU-KV-GB ... ; run → results/e3-kvbm
python3 scripts/compare_prefill_compute.py \
    --baseline-frontend results/e3-base/logs/frontend.log --baseline-logs results/e3-base/logs \
    --kvbm-frontend results/e3-kvbm/logs/frontend.log     --kvbm-logs results/e3-kvbm/logs \
    --out results/e3-cmp
```

Data: `kvbm_host/disk_cache_hit_rate`, `kvbm_offload_blocks_d2h`,
`kvbm_onboard_blocks_h2d` (host hit), `kvbm_onboard_blocks_d2d`
(**disk→device despite the name** — tag misnomer; no device-hit counter
exists, GPU-resident hits are served by vLLM's own prefix cache) — all
in `vllm_metrics.ndjson` via the kvbm scrape targets. Onboard cost =
`compare_prefill_compute.csv` delta rows.

## E4 — Break-even: CPU path for small turns

Purpose: given measured queue_ms + re-prefill + output per turn, at what
CPU throughput does host-KV CPU processing beat the GPU path?

```bash
# knobs from a CPU microbench on the target host (llama.cpp / torch-cpu etc.)
python3 scripts/cpu_offload_breakeven.py \
    --turn-sched results/e1-q4/turns/turn_sched.csv \
    --cpu-prefill-tps <measured> --cpu-decode-tps <measured> \
    --host-kv-read-gbps <measured> --kv-bytes-per-token <model-specific> \
    --out results/e4-be
```

Data: `breakeven_turns.csv` (per-turn gpu_ms vs cpu_ms, winner),
`breakeven_by_tool.csv` (cpu-win-rate per preceding tool; break-even
cpu_decode_tps p50 over small turns = "how good must the CPU be").
Link to scenario 2: the mock-LLM CPU-contention runbook
(`docs/cpu_contention_runbook.md`) bounds how much CPU headroom that
offloaded path would actually have at high in-flight.

## Arc

E0 (who is small) → E1 (misses happen, traffic-driven) → E2 (small turns
pay the most per token) → E3 (KVBM preserves KV but bills TTFT, and only
when sized right) → E4 (CPU threshold where offload wins) → conclusion:
turn-aware CPU-GPU unified scheduling.

Per-experiment metrics/plots: to be specified by the operator (TBD).
