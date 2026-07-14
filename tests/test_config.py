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


_AGG_YAML = """
workspace_root: /tmp/x
model:
  name: test-model
  served_name: local
vllm:
  agg_workers:
    - { name: a0, gpus: "0,1", tp: 2, pp: 1 }
  agg: { max_model_len: 1024, max_num_batched_tokens: 1024, max_num_seqs: 4, gpu_memory_utilization: 0.9 }
"""


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


def test_worker_dp_ep_default_off(tmp_path: Path):
    # Workers omitting dp/ep get dp=1, ep=False; tp*pp*dp == gpu_count holds.
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ={})
    w = cfg.vllm.prefill_workers[0]
    assert w.dp == 1
    assert w.ep is False


def test_worker_dp_widens_gpu_count_requirement(tmp_path: Path):
    # dp=2 with tp=2 requires 4 gpus; 2 gpus must now be rejected.
    bad = yaml.safe_load(_BASE_YAML)
    bad["vllm"]["prefill_workers"][0]["dp"] = 2  # tp*pp*dp = 4, gpus="0,1" = 2
    p = tmp_path / "bad_dp.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        cfg_mod.load(p, environ={})


def test_worker_dp_ep_accepted_when_gpu_count_matches(tmp_path: Path):
    good = yaml.safe_load(_BASE_YAML)
    good["vllm"]["prefill_workers"][0].update(gpus="0,1,2,3", dp=2, ep=True)
    p = tmp_path / "good_dp.yaml"
    p.write_text(yaml.safe_dump(good))
    cfg = cfg_mod.load(p, environ={})
    w = cfg.vllm.prefill_workers[0]
    assert w.dp == 2 and w.ep is True


def test_worker_ep_does_not_change_gpu_count(tmp_path: Path):
    # ep:true is a bare toggle — it must NOT alter the tp*pp*dp invariant,
    # so tp=2,pp=1,dp=1 still needs exactly 2 gpus with ep enabled.
    cfg = yaml.safe_load(_BASE_YAML)
    cfg["vllm"]["prefill_workers"][0]["ep"] = True  # gpus="0,1" still == tp*pp*dp=2
    p = tmp_path / "ep.yaml"
    p.write_text(yaml.safe_dump(cfg))
    loaded = cfg_mod.load(p, environ={})  # must not raise
    assert loaded.vllm.prefill_workers[0].ep is True


# ---------------------------------------------------------------------------
# PD topology: disagg (prefill_workers+decode_workers) vs agg (agg_workers)
# are mutually exclusive; exactly one must be fully configured.
# ---------------------------------------------------------------------------

def test_disagg_topology_still_validates(tmp_path: Path):
    """Backward compat: the pre-existing full prefill+decode shape (as used
    by _BASE_YAML throughout this file) must keep validating unchanged."""
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ={})
    assert len(cfg.vllm.prefill_workers) == 1
    assert len(cfg.vllm.decode_workers) == 1
    assert cfg.vllm.agg_workers == []
    assert cfg.vllm.agg is None


def test_agg_topology_validates_with_no_prefill_decode_keys(tmp_path: Path):
    """Pure PD-colocation config: agg_workers + agg section, and the yaml
    contains no prefill/decode keys at all (not even empty ones)."""
    cfg = cfg_mod.load(_write(tmp_path, _AGG_YAML), environ={})
    assert len(cfg.vllm.agg_workers) == 1
    assert cfg.vllm.agg is not None
    assert cfg.vllm.agg.max_model_len == 1024
    assert cfg.vllm.prefill_workers == []
    assert cfg.vllm.decode_workers == []
    assert cfg.vllm.prefill is None
    assert cfg.vllm.decode is None


def test_agg_workers_mutually_exclusive_with_prefill_workers(tmp_path: Path):
    mixed = yaml.safe_load(_AGG_YAML)
    mixed["vllm"]["prefill_workers"] = [
        {"name": "p0", "gpus": "2,3", "tp": 2, "pp": 1}
    ]
    p = tmp_path / "mixed.yaml"
    p.write_text(yaml.safe_dump(mixed))
    with pytest.raises(ValidationError, match="mutually exclusive"):
        cfg_mod.load(p, environ={})


def test_agg_workers_mutually_exclusive_with_decode_workers(tmp_path: Path):
    mixed = yaml.safe_load(_AGG_YAML)
    mixed["vllm"]["decode_workers"] = [
        {"name": "d0", "gpus": "2,3", "tp": 2, "pp": 1}
    ]
    p = tmp_path / "mixed2.yaml"
    p.write_text(yaml.safe_dump(mixed))
    with pytest.raises(ValidationError, match="mutually exclusive"):
        cfg_mod.load(p, environ={})


def test_no_workers_at_all_rejected(tmp_path: Path):
    body = """
workspace_root: /tmp/x
model:
  name: test-model
  served_name: local
vllm: {}
"""
    with pytest.raises(ValidationError, match="no vllm workers configured"):
        cfg_mod.load(_write(tmp_path, body), environ={})


def test_prefill_workers_without_decode_workers_rejected(tmp_path: Path):
    half = yaml.safe_load(_BASE_YAML)
    del half["vllm"]["decode_workers"]
    del half["vllm"]["decode"]
    p = tmp_path / "half_prefill.yaml"
    p.write_text(yaml.safe_dump(half))
    with pytest.raises(ValidationError, match="BOTH prefill_workers and decode_workers"):
        cfg_mod.load(p, environ={})


def test_decode_workers_without_prefill_workers_rejected(tmp_path: Path):
    half = yaml.safe_load(_BASE_YAML)
    del half["vllm"]["prefill_workers"]
    del half["vllm"]["prefill"]
    p = tmp_path / "half_decode.yaml"
    p.write_text(yaml.safe_dump(half))
    with pytest.raises(ValidationError, match="BOTH prefill_workers and decode_workers"):
        cfg_mod.load(p, environ={})


def test_disagg_workers_present_but_prefill_section_missing_rejected(tmp_path: Path):
    """Both worker lists present, but the 'prefill:' role section is dropped
    -- must be rejected distinctly from the half-pair case."""
    bad = yaml.safe_load(_BASE_YAML)
    del bad["vllm"]["prefill"]
    p = tmp_path / "no_prefill_section.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError, match="prefill:.*decode:"):
        cfg_mod.load(p, environ={})


def test_disagg_workers_present_but_decode_section_missing_rejected(tmp_path: Path):
    bad = yaml.safe_load(_BASE_YAML)
    del bad["vllm"]["decode"]
    p = tmp_path / "no_decode_section.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError, match="prefill:.*decode:"):
        cfg_mod.load(p, environ={})


def test_agg_workers_without_agg_section_rejected(tmp_path: Path):
    bad = yaml.safe_load(_AGG_YAML)
    del bad["vllm"]["agg"]
    p = tmp_path / "no_agg_section.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError, match="agg_workers requires the 'agg:' role section"):
        cfg_mod.load(p, environ={})


def test_agg_worker_gpu_count_mismatch_rejected(tmp_path: Path):
    """The per-worker field_validator (_validate_workers) now also runs over
    agg_workers, so a tp*pp*dp mismatch on an agg worker is still caught."""
    bad = yaml.safe_load(_AGG_YAML)
    bad["vllm"]["agg_workers"][0]["gpus"] = "0"  # 1 gpu, tp*pp=2
    p = tmp_path / "bad_agg_gpus.yaml"
    p.write_text(yaml.safe_dump(bad))
    with pytest.raises(ValidationError):
        cfg_mod.load(p, environ={})


def test_agg_max_model_len_env_override(tmp_path: Path):
    """TESTBED__VLLM__AGG__MAX_MODEL_LEN reaches the agg role section, same
    override mechanism already exercised for VLLM__PREFILL__* fields."""
    env = {"TESTBED__VLLM__AGG__MAX_MODEL_LEN": "8192"}
    cfg = cfg_mod.load(_write(tmp_path, _AGG_YAML), environ=env)
    assert cfg.vllm.agg.max_model_len == 8192


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


def test_monitor_dcgm_py_default_is_empty(tmp_path: Path):
    """`monitor.dcgm_py` defaults to empty string -- testbed.sh treats
    empty as 'not configured' and refuses to bring up monitor. Lives in
    yaml (not $DCGM_PY env) so `sudo testbed.sh up monitor` works
    without sudo preserving user env."""
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ={})
    assert cfg.monitor.dcgm_py == ""
    assert cfg.monitor.dcgm_bindings_path == ""


def test_monitor_dcgm_py_yaml_value(tmp_path: Path):
    body = _BASE_YAML + (
        "\nmonitor:\n"
        "  dcgm_py: /opt/venv/bin/python\n"
        "  dcgm_bindings_path: /usr/local/dcgm/bindings/python3\n"
    )
    cfg = cfg_mod.load(_write(tmp_path, body), environ={})
    assert cfg.monitor.dcgm_py == "/opt/venv/bin/python"
    assert cfg.monitor.dcgm_bindings_path == "/usr/local/dcgm/bindings/python3"


def test_monitor_dcgm_py_env_override(tmp_path: Path):
    """`TESTBED__MONITOR__DCGM_PY` overrides yaml. Useful when the user
    wants to swap interpreters without editing yaml."""
    env = {
        "TESTBED__MONITOR__DCGM_PY": "/tmp/py",
        "TESTBED__MONITOR__DCGM_BINDINGS_PATH": "/tmp/bindings",
    }
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ=env)
    assert cfg.monitor.dcgm_py == "/tmp/py"
    assert cfg.monitor.dcgm_bindings_path == "/tmp/bindings"


def test_monitor_vllm_metric_names_default_is_none(tmp_path: Path):
    """null/absent means 'use scrape_vllm_metrics.py's DEFAULT_METRIC_NAMES'.
    Passing the empty string through the script CLI also triggers the
    fallback, so this and an empty list both express the same intent;
    we just default to None for cleanliness."""
    cfg = cfg_mod.load(_write(tmp_path, _BASE_YAML), environ={})
    assert cfg.monitor.vllm_metric_names is None


def test_monitor_vllm_metric_names_yaml_list(tmp_path: Path):
    """User can pin an explicit allowlist via yaml. The list is the
    source of truth; env-var override is intentionally not supported
    (lists don't round-trip through scalar TESTBED__* vars)."""
    body = _BASE_YAML + (
        "\nmonitor:\n"
        "  vllm_metric_names:\n"
        "    - vllm:gpu_cache_usage_perc\n"
        "    - vllm:num_preemptions_total\n"
        "    - vllm:num_requests_running\n"
    )
    cfg = cfg_mod.load(_write(tmp_path, body), environ={})
    assert cfg.monitor.vllm_metric_names == [
        "vllm:gpu_cache_usage_perc",
        "vllm:num_preemptions_total",
        "vllm:num_requests_running",
    ]


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
