from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from testbed import config as cfg_mod


_BASE_YAML = """
workspace_root: /tmp/x
model:
  name: test-model
  served_name: local
vllm:
  prefill_workers:
    - { name: p0, gpus: "0,1", tp: 2, pp: 1 }
  decode_workers:
    - { name: d0, gpus: "2,3", tp: 2, pp: 1 }
  prefill: { max_model_len: 1024, max_num_batched_tokens: 1024, max_num_seqs: 4, gpu_memory_utilization: 0.9 }
  decode:  { max_model_len: 1024, max_num_batched_tokens: 1024, max_num_seqs: 4, gpu_memory_utilization: 0.9 }
"""


def _write(tmp: Path, body: str) -> Path:
    p = tmp / "testbed.yaml"
    p.write_text(body)
    return p


def test_load_defaults(tmp_path: Path):
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ={})
    assert cfg.dynamo.router_mode == "round-robin"
    assert cfg.dynamo.discovery_backend == "etcd"
    assert cfg.opencode.experimental_workspaces is True
    assert cfg.vllm.kv_connector == "NixlConnector"


def test_env_override_strings_coerce_to_typed_values(tmp_path: Path):
    env = {
        "TESTBED__DYNAMO__ROUTER_MODE": "kv",
        "TESTBED__VLLM__PREFILL__GPU_MEMORY_UTILIZATION": "0.85",
        "TESTBED__OPENCODE__PORT": "5000",
        "TESTBED__OPENCODE__EXPERIMENTAL_WORKSPACES": "false",
    }
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ=env)
    assert cfg.dynamo.router_mode == "kv"
    assert cfg.vllm.prefill.gpu_memory_utilization == 0.85
    assert cfg.opencode.port == 5000
    assert cfg.opencode.experimental_workspaces is False


def test_gpu_count_validator_rejects_mismatch(tmp_path: Path):
    bad = yaml.safe_load(_BASE_YAML)
    bad["vllm"]["prefill_workers"][0]["gpus"] = "0"  # 1 gpu, tp*pp=2
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        cfg_mod.load(p, environ={})


def test_router_mode_enum_rejects_unknown(tmp_path: Path):
    env = {"TESTBED__DYNAMO__ROUTER_MODE": "junk"}
    with pytest.raises(ValidationError):
        cfg_mod.load(_write(tmp_path, _BASE_YAML), environ=env)


def test_resolved_snapshot_is_jsonable(tmp_path: Path):
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ={})
    snap = cfg_mod.resolved_snapshot(cfg)
    assert snap["dynamo"]["router_mode"] == "round-robin"
    import json
    json.dumps(snap)  # must not raise


def test_tool_call_parser_default_is_qwen3_coder(tmp_path: Path):
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ={})
    assert cfg.vllm.tool_call_parser == "qwen3_coder"


def test_tool_call_parser_rejects_unknown_name(tmp_path: Path):
    env = {"TESTBED__VLLM__TOOL_CALL_PARSER": "not-a-real-parser"}
    with pytest.raises(ValidationError):
        cfg_mod.load(_write(tmp_path, _BASE_YAML), environ=env)


def test_tool_call_parser_empty_env_string_disables_cleanly(tmp_path: Path):
    """`TESTBED__VLLM__TOOL_CALL_PARSER=` (empty) must opt out without
    triggering pydantic validation. yaml.safe_load("") returns None, which
    would fail ToolCallParser | Literal[""] -- _apply_env_overrides therefore
    preserves the raw "" specifically for fields that accept empty-string as
    a disable sentinel."""
    env = {"TESTBED__VLLM__TOOL_CALL_PARSER": ""}
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ=env)
    assert cfg.vllm.tool_call_parser == ""


def test_tool_call_parser_yaml_empty_string_disables_cleanly(tmp_path: Path):
    body = _BASE_YAML + '  tool_call_parser: ""\n'
    cfg = cfg_mod.load(_write(tmp_path, body), environ={})
    assert cfg.vllm.tool_call_parser == ""


def test_override_generation_config_default_neutralizes_qwen_defaults(tmp_path: Path):
    """Default must pin pure-argmax greedy: temperature=0, top_p=1, top_k=-1,
    AND repetition_penalty=1.0. The last one is the load-bearing field --
    Qwen's generation_config.json ships repetition_penalty=1.05 which tilts
    logits BEFORE argmax, breaking 'greedy' reproducibility under --seed."""
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ={})
    assert cfg.vllm.override_generation_config == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 1.0,
    }


def test_override_generation_config_yaml_null_opts_out(tmp_path: Path):
    body = _BASE_YAML + "  override_generation_config: null\n"
    cfg = cfg_mod.load(_write(tmp_path, body), environ={})
    assert cfg.vllm.override_generation_config is None


def test_override_generation_config_per_key_env_override_merges_into_yaml(tmp_path: Path):
    """`_apply_env_overrides` + `_walk_set` mutate `data["vllm"]["override_..."]`
    BEFORE pydantic instantiation, so per-key env overrides merge into the
    yaml-loaded dict. (When the yaml omits the field entirely, `_walk_set`
    creates a fresh dict containing ONLY the env-overridden keys -- pydantic's
    default_factory is bypassed because the key is present in `data`.)"""
    body = _BASE_YAML + (
        "  override_generation_config:\n"
        "    temperature: 0.0\n"
        "    top_p: 1.0\n"
        "    top_k: -1\n"
        "    repetition_penalty: 1.0\n"
    )
    env = {"TESTBED__VLLM__OVERRIDE_GENERATION_CONFIG__REPETITION_PENALTY": "0.95"}
    cfg = cfg_mod.load(_write(tmp_path, body), environ=env)
    assert cfg.vllm.override_generation_config == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "repetition_penalty": 0.95,
    }


def test_override_generation_config_yaml_value_replaces_default(tmp_path: Path):
    body = _BASE_YAML + (
        "  override_generation_config:\n"
        "    temperature: 0.0\n"
        "    top_k: -1\n"
    )
    cfg = cfg_mod.load(_write(tmp_path, body), environ={})
    # yaml-provided dict replaces the default wholesale (no merge).
    assert cfg.vllm.override_generation_config == {"temperature": 0.0, "top_k": -1}


def test_monitor_dcgm_update_freq_default_is_100ms(tmp_path: Path):
    """`monitor.dcgm_update_freq_s` is the DCGM internal sampling period.
    Default 0.1s lets each 1s drain aggregate ~10 samples per field per
    GPU into {mean,min,max,n}, replacing the old GetLatest point sample."""
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ={})
    assert cfg.monitor.dcgm_update_freq_s == 0.1


def test_monitor_dcgm_update_freq_yaml_override(tmp_path: Path):
    """Yaml `monitor.dcgm_update_freq_s` is honored. Used to dial DCGM
    sampling cadence independently of the NDJSON drain `interval_s`."""
    body = _BASE_YAML + "\nmonitor:\n  dcgm_update_freq_s: 0.05\n"
    cfg = cfg_mod.load(_write(tmp_path, body), environ={})
    assert cfg.monitor.dcgm_update_freq_s == 0.05


def test_monitor_dcgm_update_freq_env_override(tmp_path: Path):
    """`TESTBED__MONITOR__DCGM_UPDATE_FREQ_S` coerces from string."""
    env = {"TESTBED__MONITOR__DCGM_UPDATE_FREQ_S": "0.2"}
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ=env)
    assert cfg.monitor.dcgm_update_freq_s == 0.2


def test_override_generation_config_env_without_yaml_bypasses_default_factory(tmp_path: Path):
    """Footgun documented in testbed.sh: when yaml OMITS the field entirely,
    a per-key env override creates a fresh dict containing ONLY the env-set
    key. pydantic's default_factory is bypassed because `data["vllm"]
    ["override_generation_config"]` is now present in the input dict, so
    `temperature`/`top_p`/`top_k` are LOST. Per-key env overrides therefore
    assume the yaml-supplied baseline; users who want a different repetition
    penalty without losing the other fields must either keep the yaml block
    intact or pass the whole JSON via TESTBED__VLLM__OVERRIDE_GENERATION_CONFIG."""
    env = {"TESTBED__VLLM__OVERRIDE_GENERATION_CONFIG__REPETITION_PENALTY": "1.0"}
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ=env)
    # NOT the four-key default -- just the single key.
    assert cfg.vllm.override_generation_config == {"repetition_penalty": 1.0}
