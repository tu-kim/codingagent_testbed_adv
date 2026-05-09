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
