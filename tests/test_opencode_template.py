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
) -> dict:
    """Mirror testbed.sh's sed substitution and parse the result."""
    text = _TMPL.read_text()
    text = text.replace("{{DYNAMO_BASE_URL}}", dynamo_base_url)
    text = text.replace("{{MODEL_SERVED_NAME}}", served_name)
    text = text.replace("{{MODEL_NAME}}", model_name)
    text = text.replace("{{PROVIDER_ID}}", provider_id)
    text = text.replace("{{TEMPERATURE}}", temperature)
    text = text.replace("{{TOP_P}}", top_p)
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


def test_rendered_template_accepts_nonzero_sampling():
    """Substitution must accept arbitrary numeric values; opencode's
    schema validates floats. Pin the render path, not a specific value."""
    cfg = _render(temperature="0.5", top_p="0.9")
    assert cfg["agent"]["build"]["temperature"] == 0.5
    assert cfg["agent"]["build"]["top_p"] == 0.9


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
            str(_TMPL),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    assert a == b
    # And every substitution actually fired (no '{{...}}' left over).
    assert "{{" not in a
