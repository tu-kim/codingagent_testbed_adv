"""Async HTTP client for the OpenCode headless server.

Every instance route accepts a `?directory=<dir>` query parameter that
`InstanceMiddleware` (opencode/packages/opencode/src/server/routes/instance/middleware.ts)
uses to resolve the working directory for that request. We always send it.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .config import OpenCodeCfg


class OpenCodeClient:
    def __init__(
        self,
        cfg: OpenCodeCfg,
        *,
        password: str | None = None,
        username: str = "opencode",
    ) -> None:
        # OpenCode authenticates via HTTP Basic when OPENCODE_SERVER_PASSWORD is
        # set on the server. The username defaults to "opencode" (see
        # opencode/packages/opencode/src/server/auth.ts:19).
        auth = httpx.BasicAuth(username, password) if password else None
        # POST /session/:id/message blocks until the agent loop completes,
        # which can take many minutes. Keep read timeout open-ended.
        self._client = httpx.AsyncClient(
            base_url=f"http://{cfg.host}:{cfg.port}",
            timeout=httpx.Timeout(connect=10.0, read=None, write=30.0, pool=None),
            auth=auth,
        )

    async def __aenter__(self) -> "OpenCodeClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def create_session(self, directory: str) -> str:
        """POST /session?directory=<dir>. Returns server-assigned session id (matches ^ses.*)."""
        resp = await self._client.post(
            "/session",
            params={"directory": directory},
            json={},
        )
        resp.raise_for_status()
        return resp.json()["id"]

    async def send_message(self, session_id: str, prompt: str, directory: str) -> dict[str, Any]:
        """POST /session/:id/message. Blocks until the agent loop finishes.

        Returns the raw JSON envelope ({info, parts}) — the FINAL assistant message only.
        Use list_messages() to get the full tool-loop history.
        """
        resp = await self._client.post(
            f"/session/{session_id}/message",
            params={"directory": directory},
            json={"parts": [{"type": "text", "text": prompt}]},
        )
        resp.raise_for_status()
        return resp.json()

    async def list_messages(self, session_id: str, directory: str) -> list[dict[str, Any]]:
        """GET /session/:id/message — the canonical source of intermediate tool-loop steps."""
        resp = await self._client.get(
            f"/session/{session_id}/message",
            params={"directory": directory},
        )
        resp.raise_for_status()
        return resp.json()

    async def stream_events(self, directory: str) -> AsyncIterator[dict[str, Any]]:
        """GET /event SSE. Exposed for debugging only; runner does not consume it."""
        async with self._client.stream(
            "GET",
            "/event",
            params={"directory": directory},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload:
                    continue
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue


def normalize_system_prompt(info: dict[str, Any]) -> str:
    """Join `info.system` (a `string[]` on stored messages) with double newlines.

    Per CLAUDE.md, OpenCode types the system prompt as `string[]` on the
    user message envelope; consumers should join them.
    """
    raw = info.get("system", [])
    if isinstance(raw, str):
        return raw
    return "\n\n".join(str(s) for s in raw)
