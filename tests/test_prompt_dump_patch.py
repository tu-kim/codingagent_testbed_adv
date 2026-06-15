"""Pin the symbols in dynamo/components/src/dynamo/vllm/handlers.py that the
testbed-owned prompt-dump patch depends on.

Two guard layers:

1. Vendored-source grep tests (require dynamo/ submodule; skip when absent):
   handlers.py must keep the four symbols our patch hooks onto. If upstream
   renames or removes any of them the patch will fail to apply or silently
   misbehave -- these tests catch the drift before a deploy.

2. Patch-file-only tests (never skip; only the .patch file is required):
   The patch itself must contain the env-gate identifiers and NDJSON record
   keys our analysis tooling reads from prompt-<pid>.jsonl. If someone edits
   the patch to rename DYN_PROMPT_DUMP_DIR or the record key "num_prompt_tokens"
   the downstream scripts break silently without these assertions.

The combine-apply test (git apply --check with both dynamo patches) verifies
that dynamo-scheduling-log.patch and dynamo-prompt-dump.patch are mutually
non-conflicting at the pinned submodule commit. It is guarded by both submodule
presence AND git being on PATH; it is NOT skipped merely because the patches
already touch different line ranges -- upstream could rewrite handlers.py in a
way that makes them conflict, and we want to know early.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DYNAMO = _REPO_ROOT / "dynamo"
_HANDLERS = _DYNAMO / "components" / "src" / "dynamo" / "vllm" / "handlers.py"
_PATCH = _REPO_ROOT / "deploy" / "patches" / "dynamo-prompt-dump.patch"
_SCHEDULING_PATCH = _REPO_ROOT / "deploy" / "patches" / "dynamo-scheduling-log.patch"

_SUBMODULE_PRESENT = _HANDLERS.exists()
_GIT_ON_PATH = shutil.which("git") is not None


# ---------------------------------------------------------------------------
# Vendored-source drift tests (require dynamo/ submodule)
# ---------------------------------------------------------------------------

_needs_submodule = pytest.mark.skipif(
    not _SUBMODULE_PRESENT,
    reason="vendored dynamo/ submodule not present",
)


@pytest.fixture(scope="module")
def handlers_src() -> str:
    return _HANDLERS.read_text()


@_needs_submodule
def test_handlers_file_exists():
    """Baseline: the submodule is checked out at a commit that still has
    handlers.py at the expected path. If the file moves upstream the other
    tests would produce misleading 'symbol absent' failures."""
    assert _HANDLERS.exists(), _HANDLERS


@_needs_submodule
def test_token_ids_key_present_in_request_dict(handlers_src):
    """The patch reads prompt tokens via request['token_ids'] (primary path in
    _build_prompt) and request.get('token_ids', ...) (the input-length helper
    near line 378). Both forms must survive upstream refactors for the dump to
    see non-empty token_ids on the hot path.

    We assert that EITHER form appears, not a specific line number, so an
    upstream refactor that merges the two into one form doesn't false-positive.
    """
    has_subscript = 'request["token_ids"]' in handlers_src
    has_get = 'request.get("token_ids"' in handlers_src
    assert has_subscript or has_get, (
        "handlers.py no longer reads token_ids from the request dict via "
        'request["token_ids"] or request.get("token_ids", ...). '
        "The prompt dump will receive empty/None token_ids on every request. "
        "Re-read handlers.py and update _dump_engine_prompt accordingly."
    )


@_needs_submodule
def test_engine_client_generate_has_at_least_three_call_sites(handlers_src):
    """The patch inserts self._dump_engine_prompt() immediately before each
    self.engine_client.generate() call. There must be at least 3 call sites
    (BaseWorkerHandler streaming path, DecodeWorkerHandler, PrefillWorkerHandler)
    for all three workers to emit dump records.

    If upstream merges paths or restructures the class hierarchy the patch will
    apply to fewer sites than intended -- the assertion on count catches that
    without pinning exact line numbers."""
    count = len(re.findall(r"\bengine_client\.generate\(", handlers_src))
    assert count >= 3, (
        f"handlers.py has only {count} engine_client.generate() call site(s); "
        "expected >= 3 (BaseWorkerHandler streaming path, DecodeWorkerHandler, "
        "PrefillWorkerHandler). The patch precedes each site; fewer sites means "
        "some workers will emit no dump records. Re-read the class hierarchy."
    )


@_needs_submodule
def test_tokens_prompt_class_used_for_prompt_construction(handlers_src):
    """The patch reconstructs prompt text by extracting prompt_token_ids from
    the TokensPrompt instance and calling tok.decode() on them. If upstream
    replaces TokensPrompt with a different wrapper class the patch's
    isinstance(prompt, dict) branch may fall through on every request.

    Presence of 'TokensPrompt(' in handlers.py confirms the class is still
    the live prompt-construction abstraction at this pinned commit."""
    assert "TokensPrompt(" in handlers_src, (
        "handlers.py no longer constructs TokensPrompt instances. "
        "The prompt dump's dict-introspection branch assumes "
        "prompt.prompt_token_ids is accessible as prompt_token_ids= kwarg. "
        "If vLLM renamed or replaced TokensPrompt, update _dump_engine_prompt "
        "to match the new prompt wrapper API."
    )


@_needs_submodule
def test_tokenizer_accessed_via_getattr_on_engine_client(handlers_src):
    """The patch detokenizes via getattr(self.engine_client, 'tokenizer', None)
    -- the same indirection already used in handlers.py near line 1718 to avoid
    AttributeError on engine backends that don't expose a tokenizer.

    This test confirms the pattern exists in the vendored source so we know the
    attribute name ('tokenizer') and the safe-access idiom are stable at the
    pinned commit. If upstream renames the attribute to 'get_tokenizer()' or
    moves it to a subobject, the dump silently falls back to
    decode_error='no tokenizer on engine_client' on every request."""
    assert 'getattr(self.engine_client, "tokenizer"' in handlers_src, (
        'handlers.py no longer accesses self.engine_client.tokenizer via '
        'getattr(self.engine_client, "tokenizer", None). '
        "The prompt dump's detokenize path mirrors this access pattern to stay "
        "safe on engine backends that don't expose a tokenizer. If upstream "
        "renamed or removed the attribute, update _dump_engine_prompt and "
        "confirm the tokenizer API under dynamo/components/src/dynamo/vllm/."
    )


# ---------------------------------------------------------------------------
# Patch-file-only tests (never skip -- only the .patch file is required)
# ---------------------------------------------------------------------------

_needs_patch = pytest.mark.skipif(
    not _PATCH.exists(),
    reason="deploy/patches/dynamo-prompt-dump.patch not present",
)


@pytest.fixture(scope="module")
def patch_text() -> str:
    return _PATCH.read_text()


@_needs_patch
def test_patch_targets_handlers_py(patch_text):
    """Sanity: the patch modifies the correct file."""
    assert "dynamo/vllm/handlers.py" in patch_text, (
        "dynamo-prompt-dump.patch no longer targets handlers.py -- "
        "the patch path may have been edited incorrectly."
    )


@_needs_patch
def test_patch_declares_env_gate(patch_text):
    """DYN_PROMPT_DUMP is the master on/off switch. If it's renamed in the
    patch, testbed.sh and CLAUDE.md references become stale."""
    assert "DYN_PROMPT_DUMP" in patch_text


@_needs_patch
def test_patch_declares_dump_dir_env(patch_text):
    """DYN_PROMPT_DUMP_DIR is where analysis scripts expect the NDJSON files.
    Renaming it here means 'ls /tmp/dynamo-prompt-dump/' produces nothing and
    the analysis appears to have run cleanly with zero prompts."""
    assert "DYN_PROMPT_DUMP_DIR" in patch_text


@_needs_patch
def test_patch_declares_text_and_tokens_env_flags(patch_text):
    """DYN_PROMPT_DUMP_TEXT (default on) and DYN_PROMPT_DUMP_TOKENS (default
    off) control fidelity. Both must be declared for the env-based toggle to
    work without source edits."""
    assert "DYN_PROMPT_DUMP_TEXT" in patch_text
    assert "DYN_PROMPT_DUMP_TOKENS" in patch_text


@_needs_patch
def test_patch_adds_dump_engine_prompt_method(patch_text):
    """_dump_engine_prompt is the single instrumentation method; all three
    call sites in the patch use this name. A rename here would require
    updating all three hunk sites in the patch."""
    assert "_dump_engine_prompt" in patch_text


@_needs_patch
def test_patch_adds_three_call_sites(patch_text):
    """The patch must inject self._dump_engine_prompt() at three engine call
    sites (two in the base handler, one each in Decode/Prefill). Fewer than 3
    means a worker type silently emits no dump records."""
    count = patch_text.count("self._dump_engine_prompt(")
    assert count >= 3, (
        f"dynamo-prompt-dump.patch contains only {count} "
        "self._dump_engine_prompt() call site(s); expected >= 3 "
        "(BaseWorkerHandler streaming, DecodeWorkerHandler, PrefillWorkerHandler)."
    )


@_needs_patch
def test_patch_tags_role_prefill_and_decode(patch_text):
    """Each call site passes a role string so PD-disaggregated analysis can
    filter to prefill-only (the canonical templated prompt). Both role literals
    must appear in the patch."""
    assert '"prefill"' in patch_text, (
        'dynamo-prompt-dump.patch no longer passes role="prefill" to '
        "_dump_engine_prompt; PrefillWorkerHandler dump records will be "
        "untagged and indistinguishable from decode records."
    )
    assert '"decode"' in patch_text, (
        'dynamo-prompt-dump.patch no longer passes role="decode" to '
        "_dump_engine_prompt; decode worker dump records will be untagged."
    )


@_needs_patch
def test_patch_ndjson_record_contains_request_id(patch_text):
    """'request_id' must appear in the NDJSON record dict so downstream
    scripts can join prompt dumps with trace.jsonl by instance/request."""
    assert '"request_id"' in patch_text


@_needs_patch
def test_patch_ndjson_record_contains_num_prompt_tokens(patch_text):
    """'num_prompt_tokens' lets quick histogram analysis skip full NDJSON parse
    for token-count distributions. If removed, analysis scripts that do
    `rec['num_prompt_tokens']` will KeyError on existing dump files."""
    assert '"num_prompt_tokens"' in patch_text


@_needs_patch
def test_patch_uses_per_pid_filename(patch_text):
    """Per-PID NDJSON files (prompt-<pid>.jsonl) prevent concurrent
    prefill/decode workers from interleaving writes and corrupting NDJSON.
    The pid suffix is pinned by analysis scripts via glob('prompt-*.jsonl')."""
    assert "os.getpid()" in patch_text, (
        "dynamo-prompt-dump.patch no longer uses os.getpid() for per-worker "
        "file naming. Concurrent workers on the same host will write to the "
        "same file and interleave NDJSON records, corrupting the dump."
    )


@_needs_patch
def test_patch_exceptions_never_raise(patch_text):
    """Instrumentation must never break the agent loop. The outer try/except
    in _dump_engine_prompt must be present so a serialisation error or full
    disk doesn't abort an active generate() call."""
    # The patch comment + code both use the 'instrumentation must never break'
    # idiom; assert the outer except block is present in the patch diff lines.
    add_lines = "\n".join(
        line[1:] for line in patch_text.splitlines() if line.startswith("+")
    )
    assert "except Exception" in add_lines, (
        "dynamo-prompt-dump.patch's _dump_engine_prompt no longer wraps its "
        "body in a bare 'except Exception'. A serialisation error will now "
        "propagate and abort the vLLM generate() call."
    )


# ---------------------------------------------------------------------------
# Combined-apply test (require both submodule AND git on PATH)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _SUBMODULE_PRESENT,
    reason="vendored dynamo/ submodule not present",
)
@pytest.mark.skipif(
    not _GIT_ON_PATH,
    reason="git not found on PATH",
)
@pytest.mark.skipif(
    not _PATCH.exists() or not _SCHEDULING_PATCH.exists(),
    reason="one or both dynamo patches missing from deploy/patches/",
)
def test_both_dynamo_patches_apply_cleanly_together():
    """dynamo-scheduling-log.patch and dynamo-prompt-dump.patch both modify
    handlers.py. Verify they are mutually non-conflicting on the pinned
    submodule commit by asking git to dry-run both in sequence.

    This catches the case where an upstream handlers.py rewrite makes the
    two patches produce overlapping hunks even though each applies cleanly on
    its own. `git apply --check` is purely a dry-run: it changes nothing in
    the working tree."""
    result = subprocess.run(
        [
            "git",
            "apply",
            "--check",
            str(_SCHEDULING_PATCH.resolve()),
            str(_PATCH.resolve()),
        ],
        cwd=str(_DYNAMO),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "dynamo-scheduling-log.patch and dynamo-prompt-dump.patch conflict "
        f"when applied together on the pinned submodule commit.\n"
        f"stdout: {result.stdout}\n"
        f"stderr: {result.stderr}\n"
        "Rebase one or both patches against the current handlers.py."
    )
