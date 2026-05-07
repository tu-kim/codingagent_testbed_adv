"""Pin the OpenCode HTTP/CLI surface our testbed code assumes against the
vendored opencode/ submodule. A rename or schema change upstream fails these
tests at lint-time, before a real run hangs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[1]
_OPENCODE = _REPO_ROOT / "opencode"
_PKG = _OPENCODE / "packages" / "opencode"
_INSTANCE_MIDDLEWARE = _PKG / "src" / "server" / "routes" / "instance" / "middleware.ts"
_SESSION_GROUP = _PKG / "src" / "server" / "routes" / "instance" / "httpapi" / "groups" / "session.ts"
_AUTH_TS = _PKG / "src" / "server" / "auth.ts"
_SERVE_TS = _PKG / "src" / "cli" / "cmd" / "serve.ts"
_NETWORK_TS = _PKG / "src" / "cli" / "network.ts"
_MESSAGE_V2 = _PKG / "src" / "session" / "message-v2.ts"


pytestmark = pytest.mark.skipif(
    not _OPENCODE.exists(),
    reason="vendored opencode/ submodule not present",
)


# ---------- Routes the Python client posts/gets to ----------

def test_session_paths_root_is_session():
    src = _SESSION_GROUP.read_text()
    assert re.search(r'^const root = "/session"\s*$', src, re.MULTILINE), (
        "OpenCode's instance session group changed root path; "
        "src/testbed/opencode.py base URL routes are now wrong."
    )


def test_post_session_create_endpoint_exists():
    """`OpenCodeClient.create_session` posts to /session."""
    src = _SESSION_GROUP.read_text()
    assert "HttpApiEndpoint.post(\"create\", SessionPaths.create" in src
    assert "create: root," in src


def test_post_session_message_is_the_v1_prompt_endpoint():
    """`OpenCodeClient.send_message` posts to /session/:id/message — this maps
    to the v1 `prompt` operation. The v2 prompt route at /api/session/:id/prompt
    is intentionally NOT used by the testbed."""
    src = _SESSION_GROUP.read_text()
    assert "prompt: `${root}/:sessionID/message`" in src
    assert 'HttpApiEndpoint.post("prompt", SessionPaths.prompt' in src


def test_get_session_message_returns_array_of_with_parts():
    """`list_messages` expects an array, not the v2 paginated `{items, cursor}`."""
    src = _SESSION_GROUP.read_text()
    assert (
        'HttpApiEndpoint.get("messages", SessionPaths.messages' in src
    )
    # The success type must be a Schema.Array of MessageV2.WithParts.
    assert re.search(
        r'success:\s*described\(\s*Schema\.Array\(MessageV2\.WithParts\)',
        src,
    ), "GET /session/:id/message no longer returns a flat array — list_messages will break"


def test_message_v2_with_parts_has_info_and_parts_fields():
    """The Python client returns the response body raw; downstream code reads
    `.info` and `.parts`."""
    src = _MESSAGE_V2.read_text()
    block = re.search(
        r"export const WithParts = Schema\.Struct\(\{(.*?)\}\)",
        src,
        re.DOTALL,
    )
    assert block, "WithParts struct moved or was renamed"
    body = block.group(1)
    assert re.search(r"\binfo\s*:", body)
    assert re.search(r"\bparts\s*:", body)


def test_post_prompt_payload_requires_parts_array():
    """The Python client sends `{"parts":[{"type":"text","text":...}]}`. The
    upstream PromptPayload is PromptInput minus sessionID; PromptInput.parts
    is REQUIRED."""
    prompt_ts = _PKG / "src" / "session" / "prompt.ts"
    src = prompt_ts.read_text()
    block = re.search(
        r"export const PromptInput = Schema\.Struct\(\{(.*?)\}\)\.pipe\(",
        src,
        re.DOTALL,
    )
    assert block, "PromptInput struct moved or was renamed"
    fields = block.group(1)
    # `parts:` (without `Schema.optional(...)` immediately after) means required.
    parts_line = re.search(r"\bparts\s*:\s*([^,\n]+)", fields)
    assert parts_line, "PromptInput no longer has a `parts` field"
    assert "Schema.optional" not in parts_line.group(1), (
        "PromptInput.parts became optional — runner.send_message body might "
        "need adjusting (or at least a docs update)."
    )


# ---------- ?directory= contract ----------

def test_instance_middleware_accepts_directory_query_or_header():
    src = _INSTANCE_MIDDLEWARE.read_text()
    # Both must be honored — testbed code uses the query, but we check both
    # so a future maintainer can switch without breaking the assumption.
    assert 'c.req.query("directory")' in src
    assert 'c.req.header("x-opencode-directory")' in src


def test_instance_middleware_resolves_directory_to_absolute_path():
    """This is the bug we hit: relative `?directory=foo` is path.resolve()'d
    against OpenCode's CWD, not against any workspace root. The runner must
    therefore send absolute paths."""
    src = _INSTANCE_MIDDLEWARE.read_text()
    assert "AppFileSystem.resolve" in src

    fs_ts = _OPENCODE / "packages" / "core" / "src" / "filesystem.ts"
    if fs_ts.exists():
        fs_src = fs_ts.read_text()
        # AppFileSystem.resolve calls Node's path.resolve (imported as `pathResolve`).
        assert "pathResolve(windowsPath(p))" in fs_src, (
            "AppFileSystem.resolve no longer uses path.resolve — the "
            "absolute-path requirement may have changed"
        )


# ---------- Auth ----------

def test_server_auth_uses_basic_with_opencode_username_default():
    src = _AUTH_TS.read_text()
    # Username default must be "opencode" — that's what OpenCodeClient hardcodes.
    assert re.search(
        r'EffectConfig\.string\("OPENCODE_SERVER_USERNAME"\)\.pipe\('
        r'EffectConfig\.withDefault\("opencode"\)\)',
        src,
    ), "Auth username default is no longer 'opencode'; OpenCodeClient init needs updating"
    # Header builder must produce a Basic-auth value.
    assert re.search(r'`Basic \$\{Buffer\.from\(`\$\{username\}:\$\{password\}`\)', src)


def test_server_auth_does_not_recognize_x_opencode_server_password_header():
    """The legacy header name our old client used does NOT exist server-side.
    Calling code must NOT regress to it."""
    middleware = _PKG / "src" / "server" / "middleware.ts"
    if middleware.exists():
        assert "x-opencode-server-password" not in middleware.read_text()
    # Also check the auth file itself.
    assert "x-opencode-server-password" not in _AUTH_TS.read_text()


# ---------- Launch ----------

def test_serve_command_accepts_hostname_and_port():
    """testbed.sh launches OpenCode with `--hostname` and `--port`. Confirm
    those flag names are still wired up via withNetworkOptions."""
    src = _NETWORK_TS.read_text()
    # Yargs option keys
    assert re.search(r"^\s*hostname:\s*\{", src, re.MULTILINE)
    assert re.search(r"^\s*port:\s*\{", src, re.MULTILINE)


def test_serve_command_is_named_serve():
    src = _SERVE_TS.read_text()
    assert re.search(r'command:\s*"serve"', src)


def test_dev_script_resolves_to_packages_opencode_index():
    """testbed.sh runs `bun run dev serve --hostname X --port Y` from `opencode/`
    (the repo root). bun's `run` forwards trailing argv to the script. The dev
    script in the root package.json must therefore boot the opencode CLI."""
    pkg = _OPENCODE / "package.json"
    src = pkg.read_text()
    # Script must invoke `packages/opencode/...src/index.ts`.
    assert re.search(
        r'"dev"\s*:\s*"[^"]*packages/opencode[^"]*src/index\.ts',
        src,
    )


def test_experimental_workspaces_env_var_is_used():
    """testbed.sh exports OPENCODE_EXPERIMENTAL_WORKSPACES=true. Confirm the
    flag name is still consumed somewhere in opencode."""
    workspace_ts = _PKG / "src" / "control-plane" / "workspace.ts"
    if workspace_ts.exists():
        assert "OPENCODE_EXPERIMENTAL_WORKSPACES" in workspace_ts.read_text()
