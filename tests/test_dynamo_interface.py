"""Pin the CLI surface our deploy/testbed.sh assumes against the vendored
Dynamo source. If a flag/env-var rename happens upstream, these tests fail
*before* a deploy goes wrong."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DYNAMO = _REPO_ROOT / "dynamo"
_FRONTEND_ARGS = _DYNAMO / "components" / "src" / "dynamo" / "frontend" / "frontend_args.py"
_FRONTEND_MAIN = _DYNAMO / "components" / "src" / "dynamo" / "frontend" / "__main__.py"
_VLLM_BACKEND_ARGS = _DYNAMO / "components" / "src" / "dynamo" / "vllm" / "backend_args.py"
_VLLM_MAIN = _DYNAMO / "components" / "src" / "dynamo" / "vllm" / "__main__.py"
_VLLM_CONSTANTS = _DYNAMO / "components" / "src" / "dynamo" / "vllm" / "constants.py"
_VLLM_MAIN_PY = _DYNAMO / "components" / "src" / "dynamo" / "vllm" / "main.py"
_RUNTIME_ARGS = (
    _DYNAMO / "components" / "src" / "dynamo" / "common" / "configuration" / "groups" / "runtime_args.py"
)
_TOOL_PARSER_SRC = (
    _DYNAMO / "lib" / "parsers" / "src" / "tool_calling" / "parsers.rs"
)
_REASONING_PARSER_SRC = (
    _DYNAMO / "lib" / "parsers" / "src" / "reasoning" / "mod.rs"
)
_RUNTIME_ENV_NAMES = (
    _DYNAMO / "lib" / "runtime" / "src" / "config" / "environment_names.rs"
)
_TESTBED_SH = _REPO_ROOT / "deploy" / "testbed.sh"


pytestmark = pytest.mark.skipif(
    not _DYNAMO.exists(),
    reason="vendored dynamo/ submodule not present",
)


# ---------- Frontend ----------

def test_dynamo_frontend_module_entrypoint_exists():
    """`python -m dynamo.frontend` resolves to this file."""
    assert _FRONTEND_MAIN.exists(), _FRONTEND_MAIN


@pytest.mark.parametrize(
    "flag",
    [
        "--http-host",
        "--http-port",
        "--router-mode",
        "--discovery-backend",
        "--request-plane",
        "--event-plane",
    ],
)
def test_frontend_flag_is_declared_upstream(flag: str):
    src = _FRONTEND_ARGS.read_text()
    assert f'flag_name="{flag}"' in src, (
        f"Dynamo frontend no longer declares {flag!r}. "
        f"deploy/testbed.sh's up_frontend() must change."
    )


@pytest.mark.parametrize(
    "flag",
    [
        "--http-host",
        "--http-port",
        "--router-mode",
        "--discovery-backend",
        "--request-plane",
        "--event-plane",
    ],
)
def test_testbed_sh_passes_each_required_frontend_flag(flag: str):
    sh = _TESTBED_SH.read_text()
    assert flag in sh, f"deploy/testbed.sh no longer passes {flag} to dynamo.frontend"


def test_frontend_router_mode_choices_match_config_enum():
    """testbed.config.RouterMode is the source of truth Python-side; it must
    be a subset of the choices Dynamo's frontend actually accepts."""
    src = _FRONTEND_ARGS.read_text()
    block = re.search(
        r'flag_name="--router-mode".*?choices=\[(.*?)\]',
        src,
        re.DOTALL,
    )
    assert block, "could not find --router-mode choices block in dynamo frontend_args.py"
    upstream = {m.group(1) for m in re.finditer(r'"([^"]+)"', block.group(1))}

    from testbed.config import RouterMode  # type: ignore[attr-defined]
    import typing

    ours = set(typing.get_args(RouterMode))
    missing = ours - upstream
    assert not missing, (
        f"testbed.config.RouterMode contains values Dynamo doesn't accept: "
        f"{sorted(missing)}; upstream={sorted(upstream)}"
    )


def test_frontend_discovery_backend_choices_match_config_enum():
    src = _FRONTEND_ARGS.read_text()
    block = re.search(
        r'flag_name="--discovery-backend".*?choices=\[(.*?)\]',
        src,
        re.DOTALL,
    )
    assert block, "could not find --discovery-backend choices block"
    upstream = {m.group(1) for m in re.finditer(r'"([^"]+)"', block.group(1))}

    from testbed.config import DiscoveryBackend
    import typing

    ours = set(typing.get_args(DiscoveryBackend))
    assert ours == upstream, (
        f"testbed.config.DiscoveryBackend ({sorted(ours)}) != "
        f"Dynamo's --discovery-backend choices ({sorted(upstream)})"
    )


def test_frontend_request_plane_and_event_plane_choices_match():
    src = _FRONTEND_ARGS.read_text()
    rp = re.search(r'flag_name="--request-plane".*?choices=\[(.*?)\]', src, re.DOTALL)
    ep = re.search(r'flag_name="--event-plane".*?choices=\[(.*?)\]', src, re.DOTALL)
    assert rp and ep
    upstream_rp = {m.group(1) for m in re.finditer(r'"([^"]+)"', rp.group(1))}
    upstream_ep = {m.group(1) for m in re.finditer(r'"([^"]+)"', ep.group(1))}

    from testbed.config import RequestPlane, EventPlane
    import typing

    assert set(typing.get_args(RequestPlane)) == upstream_rp
    assert set(typing.get_args(EventPlane)) == upstream_ep


# ---------- vLLM worker ----------

def test_dynamo_vllm_module_entrypoint_exists():
    assert _VLLM_MAIN.exists(), _VLLM_MAIN


def test_vllm_disaggregation_mode_flag_exists():
    src = _VLLM_BACKEND_ARGS.read_text()
    assert 'flag_name="--disaggregation-mode"' in src


def test_vllm_disaggregation_mode_supports_prefill_and_decode():
    src = _VLLM_CONSTANTS.read_text()
    # The DisaggregationMode enum must include the values testbed.sh passes:
    # `prefill` and `decode`. Don't pin the exhaustive set — Dynamo may add
    # new modes.
    assert re.search(r'PREFILL\s*=\s*"prefill"', src)
    assert re.search(r'DECODE\s*=\s*"decode"', src)


@pytest.mark.parametrize("flag", ["--model", "--served-model-name", "--kv-transfer-config"])
def test_testbed_sh_passes_required_vllm_flags(flag: str):
    """vLLM-native flags we depend on. (We don't grep the upstream vLLM source
    here — it's a dependency, not vendored — but we DO verify our shell uses
    the canonical names.)"""
    sh = _TESTBED_SH.read_text()
    assert flag in sh


def test_testbed_sh_kv_transfer_config_sets_correct_kv_role_per_role():
    """Prefill workers must register as kv_producer; decode as kv_consumer.
    These are the role names vLLM's NixlConnector expects."""
    sh = _TESTBED_SH.read_text()
    # Look for both literals near the disaggregation switch.
    assert "kv_producer" in sh
    assert "kv_consumer" in sh
    # And the role-mapping branch must associate prefill→producer.
    prefill_block = re.search(
        r'if\s*\[\[\s*"\$role"\s*==\s*"prefill"\s*\]\]\s*;\s*then(.*?)else(.*?)fi',
        sh,
        re.DOTALL,
    )
    assert prefill_block
    assert "kv_producer" in prefill_block.group(1)
    assert "kv_consumer" in prefill_block.group(2)


# ---------- Env vars ----------

def test_nats_server_env_var_name_is_canonical():
    """testbed.sh exports NATS_SERVER. Confirm Dynamo still reads that name."""
    src = _RUNTIME_ENV_NAMES.read_text()
    assert 'pub const NATS_SERVER: &str = "NATS_SERVER";' in src


def test_etcd_endpoints_env_var_name_is_canonical():
    """The frontend's --discovery-backend help text references ETCD_ENDPOINTS,
    which is the canonical etcd client env var (consumed by the etcd Rust SDK)."""
    src = _FRONTEND_ARGS.read_text()
    assert "ETCD_ENDPOINTS" in src


@pytest.mark.parametrize("env_var", ["NATS_SERVER", "ETCD_ENDPOINTS"])
def test_testbed_sh_exports_each_required_env(env_var: str):
    sh = _TESTBED_SH.read_text()
    # Either a `KEY=VALUE` pair (passed to spawn's env array) or a direct export.
    assert re.search(rf'\b{env_var}=', sh), (
        f"deploy/testbed.sh no longer exports {env_var}; Dynamo will fall back "
        f"to localhost defaults which is wrong for multi-node."
    )


# ---------- NIXL side-channel ports (single-host PD invariant) ----------

def test_testbed_sh_exports_unique_nixl_ports_via_rank_offset():
    """vLLM defaults all PD workers to NIXL port 5600, which collides on a
    single host. CLAUDE.md commits us to (nixl_port_base + rank*100). Verify."""
    sh = _TESTBED_SH.read_text()
    assert "VLLM_NIXL_SIDE_CHANNEL_HOST=" in sh
    assert "VLLM_NIXL_SIDE_CHANNEL_PORT=" in sh
    # The rank offset must use rank * 100 (matches the CLAUDE.md contract).
    assert re.search(r"nixl_base\s*\+\s*rank\s*\*\s*100", sh) or re.search(
        r"nixl_port_base\s*\+\s*rank\s*\*\s*100", sh
    )


# ---------- Tool-call parser (decode-only invariant) ----------

def test_dyn_tool_call_parser_flag_declared_upstream():
    """Dynamo bypasses vLLM's --enable-auto-tool-choice and runs its own
    Rust-side parsers selected via --dyn-tool-call-parser. If the flag name
    drifts, decode workers will silently lose tool-call parsing and the
    frontend logs 'tool choice auto specified but no tool parser configured'."""
    src = _RUNTIME_ARGS.read_text()
    assert 'flag_name="--dyn-tool-call-parser"' in src, (
        "dynamo runtime_args.py no longer declares --dyn-tool-call-parser; "
        "deploy/testbed.sh's spawn_worker (decode branch) is broken."
    )


def test_testbed_sh_passes_dyn_tool_call_parser_to_decode_only():
    """Per dynamo/components/src/dynamo/vllm/main.py:722-724, the parser is
    skipped on Prefill workers. testbed.sh must inject --dyn-tool-call-parser
    only when role=decode -- otherwise we either pass it on prefill (harmless
    but lies in the trace) or pass it always (noise). We encode this as
    "flag present in file" + "flag absent from the prefill branch", which is
    robust against nested `fi` shapes (a non-greedy `(.*?)fi` capture would
    stop at the first inner `fi` and miss flags placed deeper in the block)."""
    sh = _TESTBED_SH.read_text()
    assert "--dyn-tool-call-parser" in sh, (
        "deploy/testbed.sh stopped passing --dyn-tool-call-parser; the "
        "decode worker will run without server-side parsing and the agent "
        "loop receives raw text instead of structured tool_calls."
    )
    # Same shape used by test_testbed_sh_kv_transfer_config_sets_correct_kv_role_per_role:
    # the prefill/else/fi block is the only role-conditional block in spawn_worker.
    prefill_block = re.search(
        r'if\s*\[\[\s*"\$role"\s*==\s*"prefill"\s*\]\]\s*;\s*then(.*?)else(.*?)fi',
        sh,
        re.DOTALL,
    )
    assert prefill_block, "spawn_worker no longer has the prefill/else/fi role branch"
    assert "--dyn-tool-call-parser" not in prefill_block.group(1), (
        "--dyn-tool-call-parser leaked into the prefill branch — Dynamo "
        "ignores it there (main.py:722-724 only sets tool_call_parser when "
        "worker_type != WorkerType.Prefill), so its presence is a smell."
    )


def test_main_py_skips_tool_parser_on_prefill():
    """The decode-only invariant we encode in testbed.sh comes directly from
    dynamo/components/src/dynamo/vllm/main.py. If upstream removes the
    `model_type != ModelType.Prefill` guard (e.g. starts applying the parser
    everywhere, or moves the gate elsewhere), our shell branch becomes either
    redundant or wrong -- catch the drift here."""
    src = _VLLM_MAIN_PY.read_text()
    assert re.search(
        r"worker_type\s*!=\s*WorkerType\.Prefill\s*:\s*\n?\s*runtime_config\.tool_call_parser",
        src,
    ), (
        "dynamo/components/src/dynamo/vllm/main.py no longer gates "
        "tool_call_parser on `worker_type != WorkerType.Prefill`. "
        "deploy/testbed.sh's decode-only branch may now be wrong; re-read "
        "the vendored source before editing the test."
    )


def test_dyn_reasoning_parser_flag_declared_upstream():
    """Dynamo strips in-band reasoning (e.g. MiniMax M3 <mm:think>) via its
    Rust-side reasoning parsers selected with --dyn-reasoning-parser. If the
    flag name drifts, decode workers leak reasoning text into agent content."""
    src = _RUNTIME_ARGS.read_text()
    assert 'flag_name="--dyn-reasoning-parser"' in src, (
        "dynamo runtime_args.py no longer declares --dyn-reasoning-parser; "
        "deploy/testbed.sh's spawn_worker (decode branch) is broken."
    )


def test_testbed_sh_passes_dyn_reasoning_parser_to_decode_only():
    """Same decode-only invariant as the tool-call parser (main.py:722-724
    sets reasoning_parser on the `worker_type != WorkerType.Prefill` branch).
    testbed.sh must inject --dyn-reasoning-parser only on decode workers."""
    sh = _TESTBED_SH.read_text()
    assert "--dyn-reasoning-parser" in sh, (
        "deploy/testbed.sh stopped passing --dyn-reasoning-parser; decode "
        "workers will leak <think>/<mm:think> reasoning into agent content."
    )
    prefill_block = re.search(
        r'if\s*\[\[\s*"\$role"\s*==\s*"prefill"\s*\]\]\s*;\s*then(.*?)else(.*?)fi',
        sh,
        re.DOTALL,
    )
    assert prefill_block, "spawn_worker no longer has the prefill/else/fi role branch"
    assert "--dyn-reasoning-parser" not in prefill_block.group(1), (
        "--dyn-reasoning-parser leaked into the prefill branch — Dynamo "
        "only sets reasoning_parser when worker_type != WorkerType.Prefill."
    )


def test_config_reasoning_parser_literal_matches_dynamo():
    """testbed.config.ReasoningParser mirrors the `map.insert("<name>", ...)`
    registrations in dynamo/lib/parsers/src/reasoning/mod.rs (canonical runtime
    truth is dynamo._core.get_reasoning_parser_names()). Drift means our
    validator rejects reasoning parser names Dynamo accepts (or vice versa).
    Underscore-form names only; hyphenated aliases are not mirrored."""
    src = _REASONING_PARSER_SRC.read_text()
    upstream = set(re.findall(r'map\.insert\(\s*"([a-z][a-z0-9_]*)"', src))
    assert "minimax_m3" in upstream, (
        f"reasoning/mod.rs no longer registers minimax_m3; "
        f"current registered names: {sorted(upstream)}"
    )

    from testbed.config import ReasoningParser
    import typing

    ours = set(typing.get_args(ReasoningParser))
    missing_locally = upstream - ours
    extra_locally = ours - upstream
    assert not missing_locally and not extra_locally, (
        f"testbed.config.ReasoningParser drifted from dynamo reasoning/mod.rs.\n"
        f"  upstream-only: {sorted(missing_locally)}\n"
        f"  ours-only:     {sorted(extra_locally)}\n"
        f"Update src/testbed/config.py:ReasoningParser to match."
    )


def test_config_tool_call_parser_literal_matches_dynamo_doc_table():
    """testbed.config.ToolCallParser is mirrored from the authoritative
    `map.insert("<name>", ...)` registrations in
    dynamo/lib/parsers/src/tool_calling/parsers.rs (canonical runtime truth is
    dynamo._core.get_tool_parser_names(), which needs the built binary; the Rust
    map is the offline source -- the docs table proved stale, e.g. omitting
    minimax_m3 that the runtime accepts). Drift means our pydantic validator
    rejects parser names Dynamo actually accepts (or vice versa). Only
    underscore-form names are compared; hyphenated aliases (minimax-m3, ...) are
    intentionally not mirrored."""
    src = _TOOL_PARSER_SRC.read_text()
    upstream = set(re.findall(r'map\.insert\(\s*"([a-z][a-z0-9_]*)"', src))
    assert "qwen3_coder" in upstream, (
        f"parsers.rs no longer registers qwen3_coder; "
        f"current registered names: {sorted(upstream)}"
    )

    from testbed.config import ToolCallParser
    import typing

    ours = set(typing.get_args(ToolCallParser))
    missing_locally = upstream - ours
    extra_locally = ours - upstream
    assert not missing_locally and not extra_locally, (
        f"testbed.config.ToolCallParser drifted from dynamo parsers.rs.\n"
        f"  upstream-only: {sorted(missing_locally)}\n"
        f"  ours-only:     {sorted(extra_locally)}\n"
        f"Update src/testbed/config.py:ToolCallParser to match."
    )
