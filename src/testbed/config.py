"""Single source of truth loader for deploy/testbed.yaml + TESTBED__* env overrides."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
RequestPlane = Literal["tcp", "nats", "http"]
EventPlane = Literal["nats", "zmq"]
KVCacheDtype = Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
# Tool-call parser names registered in dynamo's Rust core. Sourced from
# dynamo/docs/agents/tool-calling.md:38-56 -- canonical truth at runtime is
# `dynamo._core.get_tool_parser_names()`. tests/test_dynamo_interface.py
# fails loudly if upstream renames/removes one.
ToolCallParser = Literal[
    "deepseek_v3",
    "deepseek_v3_1",
    "deepseek_v3_2",
    "default",
    "glm47",
    "harmony",
    "hermes",
    "jamba",
    "kimi_k2",
    "llama3_json",
    "minimax_m2",
    "mistral",
    "nemotron_deci",
    "nemotron_nano",
    "phi4",
    "pythonic",
    "qwen3_coder",
]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MonitorCfg(_Strict):
    enabled: bool = True
    interval_s: float = 1.0
    output: str = "logs/resource.ndjson"
    pids_from: str = "logs/"


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


class WorkerCfg(_Strict):
    name: str
    host: str = "127.0.0.1"
    gpus: str
    tp: int
    pp: int

    @field_validator("gpus")
    @classmethod
    def _gpus_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("gpus must be a non-empty comma-separated list")
        return v

    def gpu_count(self) -> int:
        return len([g for g in self.gpus.split(",") if g.strip()])

    def validate_against_parallelism(self) -> None:
        expected = self.tp * self.pp
        if self.gpu_count() != expected:
            raise ValueError(
                f"worker {self.name!r}: gpus has {self.gpu_count()} entries "
                f"but tp*pp = {expected} (gpus={self.gpus!r}, tp={self.tp}, pp={self.pp})"
            )


class VLLMRoleCfg(_Strict):
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    gpu_memory_utilization: float
    kv_cache_dtype: KVCacheDtype = "auto"


class VLLMCfg(_Strict):
    kv_connector: str = "NixlConnector"
    nixl_port_base: int = 6000
    # Each vllm worker can expose its DCGM-style dynamo system status
    # server (/metrics + /health) at host:port. Set positive to enable;
    # per-worker port = system_port_base + rank. -1 disables.
    system_port_base: int = 21000
    # Empty string => do not pass --dyn-tool-call-parser (decode worker runs
    # without server-side parsing; agent will receive raw text). Per
    # dynamo/components/src/dynamo/vllm/main.py:647-650 this is applied to
    # decode workers only; prefill workers always skip.
    tool_call_parser: ToolCallParser | Literal[""] = "qwen3_coder"
    # Tri-state vLLM toggles forwarded as paired flags
    # (--enable-prefix-caching / --no-enable-prefix-caching). None →
    # don't pass either flag (vLLM v1 default = True for prefix
    # caching). True / False → pass the corresponding flag explicitly.
    # Matters for SWE-bench reproducibility because prefix caching
    # leaks information between samples sharing prompt prefixes.
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    prefill_workers: list[WorkerCfg]
    decode_workers: list[WorkerCfg]
    prefill: VLLMRoleCfg
    decode: VLLMRoleCfg
    extra_args: str = ""

    @field_validator("prefill_workers", "decode_workers")
    @classmethod
    def _validate_workers(cls, workers: list[WorkerCfg]) -> list[WorkerCfg]:
        for w in workers:
            w.validate_against_parallelism()
        return workers


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
