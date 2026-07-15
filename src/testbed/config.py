"""Single source of truth loader for deploy/testbed.yaml + TESTBED__* env overrides."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "deploy" / "testbed.yaml"

_ENV_PREFIX = "TESTBED__"
_ENV_RE = re.compile(rf"^{re.escape(_ENV_PREFIX)}([A-Z0-9_]+(?:__[A-Z0-9_]+)+)$")


RouterMode = Literal[
    "round-robin",
    "least-loaded",
    "kv",
    "random",
    "power-of-two",
    "direct",
    "device-aware-weighted",
]
DiscoveryBackend = Literal["kubernetes", "etcd", "file", "mem"]
RequestPlane = Literal["tcp", "nats"]
EventPlane = Literal["nats", "zmq"]
KVCacheDtype = Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
# Tool-call parser names registered in dynamo's Rust core. Mirrored from the
# authoritative `map.insert("<name>", ...)` registrations in
# dynamo/lib/parsers/src/tool_calling/parsers.rs (canonical truth at runtime is
# `dynamo._core.get_tool_parser_names()`). Only the underscore-form names are
# listed; hyphenated aliases (minimax-m3, deepseek-v4, ...) are accepted by
# dynamo but intentionally omitted here. tests/test_dynamo_interface.py fails
# loudly if this drifts from parsers.rs.
ToolCallParser = Literal[
    "deepseek_v3",
    "deepseek_v3_1",
    "deepseek_v3_2",
    "deepseek_v4",
    "deepseekv4",
    "default",
    "gemma4",
    "glm47",
    "harmony",
    "hermes",
    "jamba",
    "kimi_k2",
    "llama3_json",
    "minimax_m2",
    "minimax_m3",
    "minimax_m3_nom",
    "mistral",
    "nemotron_deci",
    "nemotron_nano",
    "phi4",
    "pythonic",
    "qwen25",
    "qwen3_coder",
]

# Reasoning parser names registered in dynamo's Rust core. Mirrored from the
# `map.insert("<name>", ...)` registrations in
# dynamo/lib/parsers/src/reasoning/mod.rs (canonical truth at runtime is
# `dynamo._core.get_reasoning_parser_names()`). Underscore-form names only;
# hyphenated aliases (minimax-m3, deepseek-v4, gemma-4) are accepted by dynamo
# but omitted here. tests/test_dynamo_interface.py fails loudly on drift.
ReasoningParser = Literal[
    "basic",
    "deepseek_r1",
    "deepseek_v3",
    "deepseek_v3_1",
    "deepseek_v3_2",
    "deepseek_v4",
    "deepseekv4",
    "gemma4",
    "glm45",
    "gpt_oss",
    "granite",
    "kimi",
    "kimi_k25",
    "minimax_append_think",
    "minimax_m3",
    "mistral",
    "nemotron3",
    "nemotron_deci",
    "nemotron_nano",
    "nemotron_v3",
    "qwen3",
    "step3",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MonitorCfg(_Strict):
    # Python interpreter with DCGM bindings installed. Required by
    # `up monitor`; reading from yaml (instead of $DCGM_PY) sidesteps
    # sudo's env-stripping. Empty/null means "not configured".
    dcgm_py: str = ""
    # Optional dir containing dcgm_fields.py, prepended to sys.path by
    # monitor_resources.py when the bindings aren't pip-installed.
    # Same yaml-over-env motivation as dcgm_py.
    dcgm_bindings_path: str = ""
    interval_s: float = 1.0
    # DCGM's internal sampling period -- DCGM buffers at this rate and we
    # drain every `interval_s`, so each output row aggregates
    # `interval_s / dcgm_update_freq_s` samples per field per GPU into
    # {mean,min,max,n}. Default 0.1s = 10Hz, capped at 1s by DCGM
    # perfworks (windows >1s round short compute bursts to 0).
    dcgm_update_freq_s: float = 0.1
    output: str = "logs/resource.ndjson"
    pids_from: str = "logs/"
    # vLLM /metrics scraper (separate component from the DCGM/psutil
    # sampler above). Keys live under `monitor:` for naming consistency
    # so all background-sampling knobs share a section; env override
    # uses TESTBED__MONITOR__SCRAPE_INTERVAL_S etc.
    scrape_interval_s: float = 1.0
    scrape_output: str = "logs/vllm_metrics.ndjson"
    # Exact-name allowlist of vLLM Prometheus metric names to keep.
    # null/empty = use scrape_vllm_metrics.py's DEFAULT_METRIC_NAMES
    # (~10 KV-cache/queue/throughput metrics). Provide a list to
    # override -- e.g. to add latency histograms back, or to capture
    # a narrower subset. `TESTBED__MONITOR__VLLM_METRIC_NAMES` env
    # override is NOT supported (lists don't round-trip cleanly
    # through scalar env vars); edit the yaml.
    vllm_metric_names: list[str] | None = None


class ModelCfg(_Strict):
    name: str
    served_name: str
    # Sampling overrides applied to every primary opencode agent (build /
    # plan / general / title / summary / compaction). Default temperature=0
    # + top_p=1.0 = greedy decoding, the reproducible baseline. Without
    # these, opencode falls back to ProviderTransform.temperature(model)
    # which is 0.55 for any qwen model -- making runs non-reproducible.
    temperature: float = 0.0
    top_p: float = 1.0
    # Per-request seed (OpenAI-compat body "seed" field). With temp=0
    # greedy decoding the seed only affects tie-breaks in the sampler,
    # but it's free determinism so we pin it.
    seed: int = 42


class WorkerCfg(_Strict):
    name: str
    host: str = "127.0.0.1"
    gpus: str
    tp: int
    pp: int
    dp: int = 1          # --data-parallel-size (1 = off; pairs with ep for MoE)
    ep: bool = False     # --enable-expert-parallel (MoE expert sharding; EP size = tp*dp)

    @field_validator("gpus")
    @classmethod
    def _gpus_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("gpus must be a non-empty comma-separated list")
        return v

    def gpu_count(self) -> int:
        return len([g for g in self.gpus.split(",") if g.strip()])

    def validate_against_parallelism(self) -> None:
        # Total ranks = tp * pp * dp. Expert parallelism (ep) shards MoE
        # experts across the tp*dp ranks; it does NOT add ranks, so it
        # doesn't enter the GPU-count equation.
        expected = self.tp * self.pp * self.dp
        if self.gpu_count() != expected:
            raise ValueError(
                f"worker {self.name!r}: gpus has {self.gpu_count()} entries "
                f"but tp*pp*dp = {expected} (gpus={self.gpus!r}, "
                f"tp={self.tp}, pp={self.pp}, dp={self.dp})"
            )


class VLLMRoleCfg(_Strict):
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    gpu_memory_utilization: float
    kv_cache_dtype: KVCacheDtype = "auto"


class KvbmCfg(_Strict):
    """Dynamo KVBM (KV Block Manager): tiers KV cache to host memory (G2)
    and disk (G3) beyond the GPU pool.

    Enabled by selecting vLLM connector class "DynamoConnector" via
    --kv-transfer-config (module path kvbm.vllm_integration.connector,
    kv_role kv_both — dynamo/examples/backends/vllm/launch/agg_kvbm.sh);
    tier sizes ride env vars DYN_KVBM_CPU_CACHE_GB / DYN_KVBM_DISK_CACHE_GB
    (dynamo/lib/runtime/src/config/environment_names.rs:180-269).

    Testbed scope: AGG (colocation) workers only. The disagg shape nests
    DynamoConnector+NixlConnector under vLLM's PdConnector on the PREFILL
    worker (disagg_kvbm.sh) with kv_role=kv_both on both sides — a different
    role contract from our kv_producer/kv_consumer wiring, so it is rejected
    here until that path is validated on this tag.

    Prereq: the kvbm python extension (lib/bindings/kvbm, separate CUDA
    wheel) must be importable on the worker host; dynamo's consolidator
    degrades with only a warning when it is missing (main.py:593-599), but
    the vLLM connector itself then fails — build the wheel first.
    """

    enabled: bool = False
    # Host-tier size in GB. REQUIRED > 0 when enabled: without a CPU cache
    # KVBM has no tier to offload into. Vendor guidance: must meaningfully
    # exceed the GPU KV pool or offload churn DEGRADES performance
    # (docs/components/kvbm/kvbm-guide.md:272).
    cpu_cache_gb: float = 0.0
    # Disk-tier size in GB. 0 = no disk tier. Disk-only (cpu=0, disk>0)
    # is experimental upstream and rejected here.
    disk_cache_gb: float = 0.0
    # KVBM prometheus endpoint (kvbm_host_cache_hit_rate,
    # kvbm_offload_blocks_d2h, kvbm_onboard_blocks_h2d, ...). Per-worker
    # port = base + rank; <= 0 disables DYN_KVBM_METRICS.
    metrics_port_base: int = 6880
    # Leader coordination ZMQ ports; must be unique per co-located KVBM
    # worker (same collision class as NIXL side channels). Per-worker
    # pub = pub_base + rank, ack = ack_base + rank.
    leader_zmq_pub_port_base: int = 56001
    leader_zmq_ack_port_base: int = 56101

    @model_validator(mode="after")
    def _validate_kvbm(self) -> "KvbmCfg":
        if self.enabled:
            if self.cpu_cache_gb <= 0:
                raise ValueError(
                    "kvbm.enabled requires cpu_cache_gb > 0 (disk-only "
                    "tiering is experimental upstream and unsupported here)"
                )
            if self.disk_cache_gb < 0:
                raise ValueError("kvbm.disk_cache_gb must be >= 0")
        return self


class VLLMCfg(_Strict):
    kv_connector: str = "NixlConnector"
    nixl_port_base: int = 6000
    # Each vllm worker can expose its DCGM-style dynamo system status
    # server (/metrics + /health) at host:port. Set positive to enable;
    # per-worker port = system_port_base + rank. -1 disables.
    system_port_base: int = 21000
    # Empty string => do not pass --dyn-tool-call-parser (decode worker runs
    # without server-side parsing; agent will receive raw text). Per
    # dynamo/components/src/dynamo/vllm/main.py:723-724 this is applied to
    # decode workers only; prefill workers always skip.
    tool_call_parser: ToolCallParser | Literal[""] = "qwen3_coder"
    # Empty string => do not pass --dyn-reasoning-parser. Decode-only, same
    # branch as tool_call_parser (main.py:724). Needed for models that emit
    # in-band reasoning blocks the frontend must strip (e.g. MiniMax M3's
    # <mm:think>...</mm:think> => reasoning_parser: minimax_m3). Default off
    # because most models (e.g. qwen3-coder) emit no separate reasoning span.
    reasoning_parser: ReasoningParser | Literal[""] = ""
    # Tri-state vLLM toggles forwarded as paired flags
    # (--enable-prefix-caching / --no-enable-prefix-caching). None →
    # don't pass either flag (vLLM v1 default = True for prefix
    # caching). True / False → pass the corresponding flag explicitly.
    # Matters for SWE-bench reproducibility because prefix caching
    # leaks information between samples sharing prompt prefixes.
    # Defaults stay None so vLLM throughput optimizations remain on.
    # Prefix caching / chunked prefill / continuous batching introduce
    # only epsilon-level FP variance which seldom flips greedy argmax;
    # flip to False explicitly if output reproducibility breaks.
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    # Engine-level vLLM seed (--seed N) is the cheap-determinism win
    # that pins scheduler/sampler RNG across runs. enforce_eager and
    # disable_custom_all_reduce default to None (vLLM defaults) because
    # they hurt throughput; flip to True if seed alone doesn't suffice.
    seed: int = 42
    enforce_eager: bool | None = None
    disable_custom_all_reduce: bool | None = None
    # vLLM's `--override-generation-config '<json>'` merges into the
    # model's `generation_config.json` defaults BEFORE per-request
    # SamplingParams are built. Closes the reproducibility hole that
    # --seed alone misses: Qwen ships generation_config.json with
    # repetition_penalty=1.05, which tilts logits even under greedy
    # decoding (penalty applied before argmax). Default pins greedy +
    # neutral so vLLM behaves as a pure-argmax server for any field
    # the client doesn't override. Set to None / {} to skip the flag.
    override_generation_config: dict[str, Any] | None = Field(
        default_factory=lambda: {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "repetition_penalty": 1.0,
        }
    )
    # Two mutually exclusive deployment topologies:
    #   PD disaggregation: prefill_workers + decode_workers (+ prefill/decode
    #     role sections). Workers get --disaggregation-mode prefill|decode and
    #     a NixlConnector --kv-transfer-config; KV flows prefill -> decode.
    #   PD colocation (aggregated): agg_workers (+ agg role section). One
    #     worker type does both phases: --disaggregation-mode agg, NO
    #     --kv-transfer-config, NO NIXL side-channel env (KV stays in-engine;
    #     dynamo/components/src/dynamo/vllm/backend_args.py:378-379 resolves
    #     the omitted flag to AGGREGATED, and args.py:213-223 only requires
    #     kv-transfer-config for prefill). Aggregated workers register as
    #     component "backend" with needs=[] (args.py:183-185,
    #     worker_factory.py:563-569) so the frontend needs no changes and
    #     kv router-mode still works (agg publishes KV events —
    #     args.py:338-343 disables them for DECODE only).
    # Mixing both in one namespace is rejected: an agg worker and a PD pair
    # would register under the same component with ambiguous readiness.
    prefill_workers: list[WorkerCfg] = Field(default_factory=list)
    decode_workers: list[WorkerCfg] = Field(default_factory=list)
    agg_workers: list[WorkerCfg] = Field(default_factory=list)
    prefill: VLLMRoleCfg | None = None
    decode: VLLMRoleCfg | None = None
    agg: VLLMRoleCfg | None = None
    # KVBM host/disk KV tiering — agg-only (see KvbmCfg docstring).
    kvbm: KvbmCfg = Field(default_factory=KvbmCfg)
    extra_args: str = ""

    @field_validator("prefill_workers", "decode_workers", "agg_workers")
    @classmethod
    def _validate_workers(cls, workers: list[WorkerCfg]) -> list[WorkerCfg]:
        for w in workers:
            w.validate_against_parallelism()
        return workers

    @model_validator(mode="after")
    def _validate_topology(self) -> "VLLMCfg":
        disagg = bool(self.prefill_workers or self.decode_workers)
        agg = bool(self.agg_workers)
        if agg and disagg:
            raise ValueError(
                "agg_workers is mutually exclusive with prefill_workers/"
                "decode_workers: aggregated and disaggregated workers would "
                "register under the same dynamo component with ambiguous "
                "readiness. Configure one topology per deployment."
            )
        if not agg and not disagg:
            raise ValueError(
                "no vllm workers configured: set prefill_workers+decode_workers "
                "(PD disaggregation) or agg_workers (PD colocation)"
            )
        if disagg:
            # A decode worker's WorkerSet needs a Prefill peer to become
            # ready (dynamo model.rs readiness gating) — half a pair never
            # serves, so reject it at config time.
            if not (self.prefill_workers and self.decode_workers):
                raise ValueError(
                    "PD disaggregation needs BOTH prefill_workers and "
                    "decode_workers (a lone decode pool never becomes ready; "
                    "a lone prefill pool serves nothing). For a single-pool "
                    "setup use agg_workers."
                )
            if self.prefill is None or self.decode is None:
                raise ValueError(
                    "prefill_workers/decode_workers require the matching "
                    "'prefill:' and 'decode:' role sections"
                )
        if agg and self.agg is None:
            raise ValueError("agg_workers requires the 'agg:' role section")
        if self.kvbm.enabled and disagg:
            raise ValueError(
                "vllm.kvbm is currently supported for agg (colocation) "
                "workers only: the disagg shape needs PdConnector-nested "
                "DynamoConnector+NixlConnector with kv_role=kv_both on the "
                "prefill worker (dynamo disagg_kvbm.sh), which conflicts "
                "with our kv_producer/kv_consumer wiring and is not "
                "validated on this tag. Use agg_workers or disable kvbm."
            )
        return self


class DynamoCfg(_Strict):
    host: str = "127.0.0.1"
    port: int = 8000
    router_mode: RouterMode = "round-robin"
    discovery_backend: DiscoveryBackend = "etcd"
    etcd_endpoints: str = "http://127.0.0.1:2379"
    nats_url: str = "nats://127.0.0.1:4222"
    request_plane: RequestPlane = "tcp"
    event_plane: EventPlane = "nats"


class OpenCodeCfg(_Strict):
    host: str = "127.0.0.1"
    port: int = 4096
    experimental_workspaces: bool = True


class TestbedCfg(_Strict):
    workspace_root: str = "/tmp/testbed-workspaces"
    model: ModelCfg
    vllm: VLLMCfg
    dynamo: DynamoCfg = Field(default_factory=DynamoCfg)
    opencode: OpenCodeCfg = Field(default_factory=OpenCodeCfg)
    monitor: MonitorCfg = Field(default_factory=MonitorCfg)


def _walk_set(root: dict[str, Any], path: list[str], value: Any) -> None:
    """Set root[path[0]][path[1]]... = value, creating intermediate dicts as needed."""
    cur = root
    for key in path[:-1]:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    cur[path[-1]] = value


def _apply_env_overrides(data: dict[str, Any], environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Apply TESTBED__SECTION__KEY... env overrides into the config dict.

    Values are parsed as YAML scalars so "0.85" becomes float, "true" becomes bool.
    """
    env = environ if environ is not None else os.environ
    for var, raw in env.items():
        m = _ENV_RE.match(var)
        if not m:
            continue
        path = [seg.lower() for seg in m.group(1).split("__")]
        # `yaml.safe_load("")` returns None, which breaks fields whose schema
        # accepts the literal empty string as an opt-out (e.g.
        # vllm.tool_call_parser: ToolCallParser | Literal[""]). Preserve the
        # raw "" so a `TESTBED__FOO__BAR=` env var disables the field cleanly.
        if raw == "":
            parsed: Any = ""
        else:
            try:
                parsed = yaml.safe_load(raw)
            except yaml.YAMLError:
                parsed = raw
        _walk_set(data, path, parsed)
    return data


def load(path: Path | str | None = None, *, environ: dict[str, str] | None = None) -> TestbedCfg:
    """Load testbed.yaml and apply TESTBED__* env overrides."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with cfg_path.open("r") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{cfg_path}: top-level YAML must be a mapping")
    _apply_env_overrides(data, environ=environ)
    return TestbedCfg(**data)


def resolved_snapshot(cfg: TestbedCfg) -> dict[str, Any]:
    """Return the fully-resolved config as a JSON-serializable dict."""
    return cfg.model_dump(mode="json")
