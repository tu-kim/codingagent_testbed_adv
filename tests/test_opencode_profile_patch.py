"""Pin the testbed-owned opencode profile patch's contract.

The patch lives at deploy/patches/opencode-profile.patch and is applied
on top of the vendored opencode/ submodule by
scripts/apply_opencode_patches.sh. These tests fail loudly if the patch
loses any of the hooks our analysis tooling depends on (Profile.llm
methods, the `case "finish":` hook, dynamo nvext extraction, etc.).

No opencode/ submodule needed -- we only read the patch file itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_PATCH = _REPO_ROOT / "deploy" / "patches" / "opencode-profile.patch"


pytestmark = pytest.mark.skipif(
    not _PATCH.exists(),
    reason="opencode-profile.patch not present (deploy/patches/)",
)


@pytest.fixture(scope="module")
def patch_text() -> str:
    return _PATCH.read_text()


# ---------- profile.ts (the new file added by the patch) ----------


def test_patch_creates_profile_module(patch_text):
    """profile.ts is added as a new file via `@@ -0,0 ...`."""
    assert "+++ b/packages/opencode/src/profile/profile.ts" in patch_text


def test_session_state_type_declares_stream_finish_map(patch_text):
    """Profile.llm.streamFinish stores per-step ms in this map; if the
    field declaration drops, the patch fails to compile."""
    assert "streamFinishByStep: Map<number, number>" in patch_text


def test_session_state_initializes_stream_finish_map(patch_text):
    """getOrInit must populate streamFinishByStep with an empty Map.
    A missing init means `streamEnd_s` calculation degenerates silently
    (always falls back to firstTool/lastText/stepEnd)."""
    assert "streamFinishByStep: new Map()" in patch_text


def test_sessions_cleared_on_query_end(patch_text):
    """The sessions Map must be drained at query.end to avoid leaking
    file descriptors + per-step maps across long-running parents."""
    assert "sessions.delete(sessionID)" in patch_text


def test_extract_dynamo_timing_helper_present(patch_text):
    """The helper that pulls nvext.timing out of AI SDK's providerMetadata.
    Keep its name pinned so the processor.ts hook keeps wiring through."""
    assert "function extractDynamoTiming" in patch_text


def test_stream_finish_method_present(patch_text):
    """Profile.llm.streamFinish emits the `llm.stream-finish` event with
    dynamo timing in-band -- the centerpiece of moving off log scraping."""
    assert "streamFinish(" in patch_text
    assert '"llm.stream-finish"' in patch_text


def test_llm_end_emits_dynamo_timing(patch_text):
    """`llm.end` payload carries `dynamo: extractDynamoTiming(...)` so a
    single profile NDJSON is sufficient for analysis."""
    assert "dynamo: extractDynamoTiming(info.providerMetadata)" in patch_text


def test_llm_end_emits_post_stream_overhead(patch_text):
    """Decomposition: step_duration_s - stream_end_s = framework
    finalization (snapshot + DB writes). Surface it explicitly."""
    assert "post_stream_overhead_s" in patch_text


# ---------- processor.ts hooks ----------


def test_processor_hooks_start_step(patch_text):
    assert "Profile.llm.start(ctx.sessionID)" in patch_text


def test_processor_hooks_text_end(patch_text):
    assert "Profile.llm.streamingEnd(ctx.sessionID)" in patch_text


def test_processor_hooks_finish_step_with_provider_metadata(patch_text):
    """finish-step hook MUST pass providerMetadata so dynamo nvext lands
    in `llm.end.dynamo`. Regression here = silent loss of server timing."""
    assert "Profile.llm.end(ctx.sessionID, {" in patch_text
    assert "providerMetadata: value.providerMetadata" in patch_text


def test_processor_hooks_finish_event(patch_text):
    """The `case "finish":` hook is what unlocks accurate stream_end_s
    and dynamo nvext extraction (providerMetadata only arrives at the
    AI SDK `finish` event for some providers)."""
    assert "case \"finish\":" in patch_text
    fin_idx = patch_text.find("case \"finish\":")
    # Find the next add-line marker right after to confirm the streamFinish
    # call is inserted under this case, not under some unrelated branch.
    tail = patch_text[fin_idx : fin_idx + 400]
    assert "Profile.llm.streamFinish(ctx.sessionID" in tail


def test_processor_hooks_tool_wrappers(patch_text):
    """Both builtin and MCP tool execute wrappers call Profile.tool.start
    -- needed for the firstToolStartByStep heuristic to keep working
    even on the legacy duration_s path. Patterns use 'in' (no exact
    arg-list match) so a future arg reorder doesn't break the test."""
    assert 'Profile.tool.start(ctx.sessionID,' in patch_text
    assert '"builtin", args)' in patch_text
    assert '"mcp", args)' in patch_text


def test_processor_hooks_tool_end_on_both_paths(patch_text):
    """tool.end is fired by Effect.onExit so failures/dies still produce
    a `tool.end` with ok:false. Pin both builtin and MCP paths."""
    builtin = patch_text.find("Profile.tool.end(ctx.sessionID, callID, {")
    mcp = patch_text.find("Profile.tool.end(ctx.sessionID, opts.toolCallId, {")
    assert builtin != -1, "builtin tool.end hook missing"
    assert mcp != -1, "MCP tool.end hook missing"
    # Each path must wrap in Effect.onExit (covers success + defects)
    assert patch_text.count("Effect.onExit") >= 2


# ---------- prompt.ts hooks (query + turn brackets) ----------


def test_prompt_hooks_query_start(patch_text):
    """Outermost bracket around the agent loop; carries directory + initial
    snapshot. Without it, summary.json's duration computation is broken."""
    assert "Profile.query.start(sessionID," in patch_text


def test_prompt_hooks_query_end(patch_text):
    """Closes the outermost bracket and triggers sessions.delete cleanup."""
    assert "Profile.query.end(sessionID," in patch_text


def test_prompt_hooks_turn_start_carries_messages_snapshot(patch_text):
    """Profile.turn.start snapshots the exact messages going to the LLM
    (subject to OPENCODE_PROFILE_MESSAGES level). Loss of this hook
    means full prompt replay becomes impossible."""
    assert "Profile.turn.start(sessionID, step," in patch_text
    # The snapshot wires through system + messages + model fields
    assert "messages: messagesForLLM" in patch_text


def test_prompt_hooks_turn_end(patch_text):
    assert "Profile.turn.end(sessionID, step)" in patch_text


# ---------- env-gating + sentinel values ----------


def test_env_gate_pinned(patch_text):
    """Every Profile.* method short-circuits when OPENCODE_PROFILE is
    unset/falsy. Without this gate, prod runs would write NDJSON files
    every session."""
    assert "OPENCODE_PROFILE" in patch_text
    assert "envEnabled" in patch_text


def test_snapshot_levels_pinned(patch_text):
    """OPENCODE_PROFILE_MESSAGES values consumed by summarizeMessages /
    summarizeSystem. CLAUDE.md documents them; pin the literal set."""
    assert "OPENCODE_PROFILE_MESSAGES" in patch_text
    for level in ('"full"', '"head"', '"count"'):
        assert level in patch_text
