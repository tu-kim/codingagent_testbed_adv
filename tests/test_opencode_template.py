"""Verify deploy/opencode.json.tmpl renders to a structure that matches the
schema expected by the vendored opencode/ submodule. No network."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_TMPL = _REPO_ROOT / "deploy" / "opencode.json.tmpl"
_OPENCODE_PROVIDER_TS = (
    _REPO_ROOT / "opencode" / "packages" / "opencode" / "src" / "config" / "provider.ts"
)
_OPENCODE_CONFIG_TS = (
    _REPO_ROOT / "opencode" / "packages" / "opencode" / "src" / "config" / "config.ts"
)


def _render(
    *,
    dynamo_base_url: str = "http://127.0.0.1:8000/v1",
    served_name: str = "local",
    model_name: str = "qwen3-coder-30b-a3b",
    provider_id: str = "dynamo",
    temperature: str = "0.0",
    top_p: str = "1.0",
    seed: str = "42",
) -> dict:
    """Mirror testbed.sh's sed substitution and parse the result."""
    text = _TMPL.read_text()
    text = text.replace("{{DYNAMO_BASE_URL}}", dynamo_base_url)
    text = text.replace("{{MODEL_SERVED_NAME}}", served_name)
    text = text.replace("{{MODEL_NAME}}", model_name)
    text = text.replace("{{PROVIDER_ID}}", provider_id)
    text = text.replace("{{TEMPERATURE}}", temperature)
    text = text.replace("{{TOP_P}}", top_p)
    text = text.replace("{{SEED}}", seed)
    return json.loads(text)


def test_template_has_no_unsubstituted_placeholders():
    """Catch the case where someone adds {{FOO}} to the template but forgets to
    teach testbed.sh to substitute it — the resulting file would crash OpenCode."""
    raw = _TMPL.read_text()
    leftover = re.findall(r"\{\{[A-Z_]+\}\}", raw)
    # Each placeholder must be substituted in deploy/testbed.sh.
    sh = (_REPO_ROOT / "deploy" / "testbed.sh").read_text()
    for ph in leftover:
        assert ph in sh, f"template uses {ph!r} but deploy/testbed.sh has no sed for it"


def test_rendered_template_is_valid_json():
    cfg = _render()
    assert isinstance(cfg, dict)


def test_rendered_template_has_provider_block_with_required_fields():
    cfg = _render(provider_id="dynamo", served_name="local", model_name="qwen3-coder-30b-a3b")
    assert "provider" in cfg
    assert "dynamo" in cfg["provider"]
    block = cfg["provider"]["dynamo"]
    # The four fields the OpenCode provider Info schema actually consumes:
    assert block["name"] == "dynamo"
    assert block["npm"] == "@ai-sdk/openai-compatible"
    assert block["options"]["baseURL"] == "http://127.0.0.1:8000/v1"
    assert block["options"]["apiKey"] == "unused"
    # The model registry uses the served-name as the key (= the id sent
    # upstream to Dynamo) and `name` as a display label.
    assert "local" in block["models"]
    assert block["models"]["local"]["name"] == "qwen3-coder-30b-a3b"


def test_testbed_sh_uses_yq_fallback_for_sampling():
    """yq returns the literal string 'null' for missing keys, so without
    a `// <default>` fallback the rendered opencode.json gets
    "temperature": null and the override is silently lost (opencode falls
    back to ProviderTransform default 0.55 for qwen). This regression is
    invisible in normal smoke testing; pin the fallback explicitly."""
    sh = (_REPO_ROOT / "deploy" / "testbed.sh").read_text()
    assert ".model.temperature // 0" in sh, (
        "deploy/testbed.sh dropped the yq fallback for "
        ".model.temperature -- if user's local testbed.yaml predates "
        "the sampling fields, rendered opencode.json will have "
        '"temperature": null and override is silently lost.'
    )
    assert ".model.top_p // 1" in sh, "same applies to .model.top_p fallback"


def test_rendered_template_overrides_sampling_on_all_primary_agents():
    """Without an explicit per-agent override, opencode falls back to
    ProviderTransform.temperature(model) which returns 0.55 for qwen --
    making runs non-reproducible. We pin temperature=0 / top_p=1.0 on
    every primary agent so experiment runs are greedy-decoded."""
    cfg = _render(temperature="0.0", top_p="1.0")
    assert "agent" in cfg
    expected_agents = ("build", "plan", "general", "title", "summary", "compaction")
    for name in expected_agents:
        assert name in cfg["agent"], f"agent.{name} missing from rendered template"
        a = cfg["agent"][name]
        assert a["temperature"] == 0.0, f"agent.{name}.temperature != 0"
        assert a["top_p"] == 1.0, f"agent.{name}.top_p != 1"


def test_rendered_template_pins_seed_as_flat_option():
    """Agent options must be FLAT (just `{seed: N}`). opencode's
    ProviderTransform.providerOptions(model, options) wraps the dict
    under the provider key automatically (`{ [providerID]: options }`,
    see provider/transform.ts:1186), so pre-nesting under the
    provider name produces a double-wrap that makes vLLM reject the
    request body with 400 (seen in live testing 2026-05-22)."""
    cfg = _render(seed="42")
    for name in ("build", "plan", "general", "title", "summary", "compaction"):
        opts = cfg["agent"][name].get("options", {})
        assert opts == {"seed": 42,
                        "nvext": {"extra_fields": ["timing"]}}, (
            f"agent.{name}.options must be flat (no provider key wrap; "
            f"ProviderTransform handles wrapping) with the seed and the "
            f"nvext timing opt-in, got {opts!r}"
        )


def test_rendered_template_opts_into_nvext_timing():
    """Since dynamo v1.3.0-minimax-m3-dev.1, nvext.timing on the response
    is PER-REQUEST OPT-IN: the frontend attaches it only when the request
    body carries nvext.extra_fields containing "timing"
    (dynamo/lib/llm/src/protocols/openai/nvext.rs:215-234,297). Without
    this every profile llm.end.dynamo is null and elapsed-based analyses
    degrade to client-bracket approximations."""
    cfg = _render()
    for name in ("build", "plan", "general", "title", "summary", "compaction"):
        nv = cfg["agent"][name].get("options", {}).get("nvext")
        assert nv == {"extra_fields": ["timing"]}, (
            f"agent.{name}.options.nvext must opt into timing, got {nv!r}"
        )


def test_rendered_template_model_options_cover_unknown_agents():
    """seed + nvext must ALSO live on the provider MODEL options.
    Agent-level options only cover the six agents named in the template;
    task-tool SUBAGENT sessions (and any future agent name) run with an
    agent whose options default to {}, so their requests would carry
    neither seed nor the nvext timing opt-in — observed 2026-08-03 as
    whole sessions with llm.end.dynamo null. llm.ts merges
    input.model.options into EVERY request regardless of agent
    (session/llm.ts:141; config model options land on Provider.Model via
    provider.ts:1242), so the model block is the catch-all."""
    cfg = _render(seed="42")
    models = cfg["provider"]["dynamo"]["models"]
    (model_cfg,) = models.values()
    opts = model_cfg.get("options", {})
    assert opts == {"seed": 42, "nvext": {"extra_fields": ["timing"]}}, (
        f"model-level options must carry seed + nvext opt-in "
        f"(subagent coverage), got {opts!r}"
    )


def test_rendered_template_seed_value_substitutes():
    cfg = _render(seed="7")
    assert cfg["agent"]["build"]["options"]["seed"] == 7
    cfg = _render(seed="12345")
    assert cfg["agent"]["title"]["options"]["seed"] == 12345


def test_rendered_template_accepts_nonzero_sampling():
    """Substitution must accept arbitrary numeric values; opencode's
    schema validates floats. Pin the render path, not a specific value."""
    cfg = _render(temperature="0.5", top_p="0.9")
    assert cfg["agent"]["build"]["temperature"] == 0.5
    assert cfg["agent"]["build"]["top_p"] == 0.9


def test_rendered_template_permission_is_catchall_allow():
    """Headless hang-prevention invariant (sibling to OPENCODE_CLIENT=server
    for the question tool). opencode's permission evaluate() defaults to "ask"
    when no rule matches; the asking tools block on Deferred.await forever with
    no human approver. A write to /tmp (outside the workspace) does NOT trigger
    the `edit` permission first -- write.ts calls assertExternalDirectoryEffect
    BEFORE the edit ask, which asks `external_directory` (only Truncate.GLOB is
    allowed by default, not arbitrary /tmp). So allowing just edit/bash/webfetch
    is insufficient and still hangs. The fix is a catch-all `{"*": "allow"}`
    that covers every gate (edit, bash, webfetch, external_directory, doom_loop,
    ...). opencode's Wildcard `*` compiles to `.*` with the dotall flag, so the
    `*` pattern matches deep external paths with slashes too."""
    cfg = _render()
    perm = cfg.get("permission")
    assert isinstance(perm, dict), "rendered template missing top-level 'permission' block"
    assert perm.get("*") == "allow", (
        f"permission must be catch-all {{'*': 'allow'}} to prevent headless "
        f"approval hangs (incl. external_directory for out-of-tree writes), "
        f"got {perm!r}"
    )


def test_rendered_template_pins_model_to_provider_slash_served_name():
    cfg = _render(provider_id="dynamo", served_name="local")
    # OpenCode's top-level `model` field is `provider/model` per the schema
    # comment in opencode/packages/opencode/src/config/config.ts.
    assert cfg["model"] == "dynamo/local"


@pytest.mark.skipif(not _OPENCODE_PROVIDER_TS.exists(), reason="vendored opencode/ not present")
def test_provider_npm_field_name_matches_vendored_schema():
    """If OpenCode renames `provider.<id>.npm` to something else, our render breaks."""
    src = _OPENCODE_PROVIDER_TS.read_text()
    # The Info struct must declare an `npm:` field. We're being deliberately
    # loose to tolerate Schema.optional(...) wrappings.
    assert re.search(r"\bnpm:\s*Schema\.", src), (
        "opencode/packages/opencode/src/config/provider.ts no longer declares "
        "an `npm` field on the provider Info struct — opencode.json.tmpl is "
        "now broken."
    )


@pytest.mark.skipif(not _OPENCODE_CONFIG_TS.exists(), reason="vendored opencode/ not present")
def test_top_level_model_field_exists_in_vendored_schema():
    src = _OPENCODE_CONFIG_TS.read_text()
    assert re.search(r"\bmodel:\s*Schema\.optional\(ConfigModelID\)", src), (
        "opencode/packages/opencode/src/config/config.ts no longer declares a "
        "top-level `model` field — opencode.json.tmpl needs to change."
    )


def test_rendered_template_disables_huggingface_provider():
    """Without this, OpenCode auto-enables the huggingface provider whenever
    HF_TOKEN is exported (provider.ts ~line 162 `input.env.some(item => env[item])`)
    and silently routes inference to https://router.huggingface.co/v1, bypassing
    the configured local Dynamo. The disabled_providers list is consumed at
    provider.ts:1133 (`new Set(cfg.disabled_providers ?? [])`)."""
    cfg = _render()
    assert cfg.get("disabled_providers") == ["huggingface"], (
        "opencode.json.tmpl must keep huggingface in disabled_providers; "
        "removing it lets HF_TOKEN auto-enable the HF provider and override "
        "the dynamo provider at request time."
    )


@pytest.mark.skipif(not _OPENCODE_CONFIG_TS.exists(), reason="vendored opencode/ not present")
def test_disabled_providers_field_exists_in_vendored_schema():
    src = _OPENCODE_CONFIG_TS.read_text()
    assert re.search(r"\bdisabled_providers:\s*Schema\.", src), (
        "opencode/packages/opencode/src/config/config.ts no longer declares "
        "`disabled_providers` — the safety belt in opencode.json.tmpl is now a no-op."
    )


def test_testbed_sh_passes_opencode_config_env_to_spawn():
    """OpenCode walks up from the per-request ?directory= looking for
    opencode.json (paths.ts:10-21). Our rendered file lives at
    $OPENCODE_DIR/opencode.json, which is NOT on that walk path -- so without
    OPENCODE_CONFIG=<abs> in the spawn env, the rendered config is ignored
    and HF_TOKEN auto-enables the huggingface provider, sending inference to
    router.huggingface.co. Guard against silent regression."""
    sh = (_REPO_ROOT / "deploy" / "testbed.sh").read_text()
    assert "OPENCODE_CONFIG=" in sh, (
        "deploy/testbed.sh stopped passing OPENCODE_CONFIG to the OpenCode "
        "spawn -- per-request config discovery will silently miss "
        "opencode/opencode.json and fall back to HF_TOKEN auto-enable."
    )


def test_substitution_is_idempotent_across_runs(tmp_path: Path):
    """testbed.sh runs `sed -e ... opencode.json.tmpl > opencode/opencode.json`
    on every `up opencode`. The output must be byte-identical for the same inputs."""
    a = subprocess.run(
        [
            "sed",
            "-e", "s|{{DYNAMO_BASE_URL}}|http://127.0.0.1:8000/v1|g",
            "-e", "s|{{MODEL_SERVED_NAME}}|local|g",
            "-e", "s|{{MODEL_NAME}}|qwen3-coder-30b-a3b|g",
            "-e", "s|{{PROVIDER_ID}}|dynamo|g",
            "-e", "s|{{TEMPERATURE}}|0.0|g",
            "-e", "s|{{TOP_P}}|1.0|g",
            "-e", "s|{{SEED}}|42|g",
            str(_TMPL),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    b = subprocess.run(
        [
            "sed",
            "-e", "s|{{DYNAMO_BASE_URL}}|http://127.0.0.1:8000/v1|g",
            "-e", "s|{{MODEL_SERVED_NAME}}|local|g",
            "-e", "s|{{MODEL_NAME}}|qwen3-coder-30b-a3b|g",
            "-e", "s|{{PROVIDER_ID}}|dynamo|g",
            "-e", "s|{{TEMPERATURE}}|0.0|g",
            "-e", "s|{{TOP_P}}|1.0|g",
            "-e", "s|{{SEED}}|42|g",
            str(_TMPL),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    assert a == b
    # And every substitution actually fired (no '{{...}}' left over).
    assert "{{" not in a
