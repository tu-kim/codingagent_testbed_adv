from __future__ import annotations

import json

import httpx
import pytest
import respx

from testbed.config import OpenCodeCfg
from testbed.opencode import OpenCodeClient, normalize_system_prompt


_CFG = OpenCodeCfg(host="127.0.0.1", port=4096)
_BASE = "http://127.0.0.1:4096"


@pytest.fixture
def cfg() -> OpenCodeCfg:
    return _CFG


@respx.mock(assert_all_called=True)
async def test_create_session_posts_to_session_with_directory_query(respx_mock, cfg):
    route = respx_mock.post(f"{_BASE}/session", params={"directory": "wkspace"}).mock(
        return_value=httpx.Response(200, json={"id": "ses_123"})
    )
    async with OpenCodeClient(cfg) as client:
        sid = await client.create_session(directory="wkspace")
    assert sid == "ses_123"
    assert route.called
    # POST body must be present (server expects optional body — we send {}).
    req = route.calls[0].request
    assert req.method == "POST"
    assert json.loads(req.content or b"{}") == {}


@respx.mock(assert_all_called=True)
async def test_send_message_payload_shape(respx_mock, cfg):
    route = respx_mock.post(
        f"{_BASE}/session/ses_123/message",
        params={"directory": "wkspace"},
    ).mock(return_value=httpx.Response(200, json={"info": {"id": "msg_1"}, "parts": []}))
    async with OpenCodeClient(cfg) as client:
        env = await client.send_message("ses_123", "hello", directory="wkspace")
    assert env == {"info": {"id": "msg_1"}, "parts": []}
    body = json.loads(route.calls[0].request.content)
    assert body == {"parts": [{"type": "text", "text": "hello"}]}


@respx.mock(assert_all_called=True)
async def test_list_messages_returns_response_unchanged(respx_mock, cfg):
    payload = [{"info": {"id": "m1"}, "parts": []}, {"info": {"id": "m2"}, "parts": []}]
    respx_mock.get(
        f"{_BASE}/session/ses_123/message",
        params={"directory": "wkspace"},
    ).mock(return_value=httpx.Response(200, json=payload))
    async with OpenCodeClient(cfg) as client:
        out = await client.list_messages("ses_123", directory="wkspace")
    assert out == payload


@respx.mock(assert_all_called=True)
async def test_basic_auth_header_with_default_username(respx_mock, cfg):
    """OpenCode authenticates via HTTP Basic; username defaults to 'opencode'."""
    import base64

    route = respx_mock.post(f"{_BASE}/session", params={"directory": "wkspace"}).mock(
        return_value=httpx.Response(200, json={"id": "ses_ok"})
    )
    async with OpenCodeClient(cfg, password="hunter2") as client:
        await client.create_session(directory="wkspace")
    headers = route.calls[0].request.headers
    expected = "Basic " + base64.b64encode(b"opencode:hunter2").decode()
    assert headers.get("authorization") == expected
    # The legacy header must NOT be sent — it has no effect server-side.
    assert "x-opencode-server-password" not in {k.lower() for k in headers.keys()}


@respx.mock(assert_all_called=True)
async def test_basic_auth_header_with_custom_username(respx_mock, cfg):
    import base64

    route = respx_mock.post(f"{_BASE}/session", params={"directory": "wkspace"}).mock(
        return_value=httpx.Response(200, json={"id": "ses_ok"})
    )
    async with OpenCodeClient(cfg, password="hunter2", username="alice") as client:
        await client.create_session(directory="wkspace")
    headers = route.calls[0].request.headers
    expected = "Basic " + base64.b64encode(b"alice:hunter2").decode()
    assert headers.get("authorization") == expected


@respx.mock(assert_all_called=True)
async def test_no_auth_header_when_password_unset(respx_mock, cfg):
    route = respx_mock.post(f"{_BASE}/session", params={"directory": "wkspace"}).mock(
        return_value=httpx.Response(200, json={"id": "ses_ok"})
    )
    async with OpenCodeClient(cfg) as client:
        await client.create_session(directory="wkspace")
    headers = route.calls[0].request.headers
    # httpx normalizes header keys to lowercase.
    assert "authorization" not in {k.lower() for k in headers.keys()}


@respx.mock(assert_all_called=True)
async def test_directory_query_passes_absolute_path_unchanged(respx_mock, cfg):
    """OpenCode resolves `?directory=` via path.resolve() against its CWD; the
    runner sends an absolute path so the middleware finds the pre-cloned dir."""
    abs_path = "/tmp/testbed-workspaces/session-django__django-1-abc12345"
    route = respx_mock.post(f"{_BASE}/session", params={"directory": abs_path}).mock(
        return_value=httpx.Response(200, json={"id": "ses_abs"})
    )
    async with OpenCodeClient(cfg) as client:
        sid = await client.create_session(directory=abs_path)
    assert sid == "ses_abs"
    sent = route.calls[0].request.url.params.get("directory")
    assert sent == abs_path


def test_normalize_system_prompt_joins_string_array():
    info = {"system": ["one", "two", "three"]}
    assert normalize_system_prompt(info) == "one\n\ntwo\n\nthree"


def test_normalize_system_prompt_passthrough_string():
    info = {"system": "alpha"}
    assert normalize_system_prompt(info) == "alpha"


def test_normalize_system_prompt_missing():
    assert normalize_system_prompt({}) == ""
