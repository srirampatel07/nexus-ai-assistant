"""Local llama.cpp provider tests (local Qwen3-8B integration).

All HTTP mocked via httpx.MockTransport — the suite never needs a running
server. Local mode = openai-compatible provider, local base URL, exact
server model id, empty API key (no Bearer header sent).
"""

import httpx
import pytest

from app.brain.provider import (
    ChatMessage,
    OpenAICompatibleProvider,
    create_provider,
)
from app.core.config import Settings
from app.core.exceptions import ProviderUnavailable
from app.tools.schemas import LlmTool

LOCAL_BASE = "http://127.0.0.1:8080/v1"
LOCAL_MODEL = "Qwen3-8B-Q4_K_M"


def _local_settings(**over):
    kwargs = dict(
        ai_provider="openai-compatible",
        ai_model=LOCAL_MODEL,
        ai_api_key="",
        ai_base_url=LOCAL_BASE,
    )
    kwargs.update(over)
    return Settings(**kwargs)


def _patched(handler, **provider_kw):
    params = dict(base_url=LOCAL_BASE, model=LOCAL_MODEL, api_key="")
    params.update(provider_kw)
    provider = OpenAICompatibleProvider(**params)
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    class FakeClient:
        def __init__(self, *a, **k):
            self._client = real_client(transport=transport)

        async def __aenter__(self):
            return self._client

        async def __aexit__(self, *a):
            await self._client.aclose()

    return provider, FakeClient


def _messages():
    return [ChatMessage(role="user", content="hello local")]


def test_local_settings_build_compatible_provider_without_key():
    provider = create_provider(_local_settings())
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._base_url == LOCAL_BASE
    assert provider._model == LOCAL_MODEL
    assert "Authorization" not in provider._headers()


@pytest.mark.asyncio
async def test_local_base_url_and_model_used(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        import json

        seen.update(json.loads(request.read().decode()))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "hi from qwen"}}]}
        )

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    resp = await provider.send_message(_messages())
    assert resp.content == "hi from qwen"
    assert seen["url"] == f"{LOCAL_BASE}/chat/completions"
    assert seen["model"] == LOCAL_MODEL
    assert seen["auth"] is None  # empty key: no credential sent


@pytest.mark.asyncio
async def test_local_health_check_hits_models_endpoint(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"data": [{"id": LOCAL_MODEL}]})

    provider = OpenAICompatibleProvider(
        base_url=LOCAL_BASE, model=LOCAL_MODEL, api_key=""
    )
    real_client = httpx.AsyncClient
    transport = httpx.MockTransport(handler)

    class FakeClient:
        def __init__(self, *a, **k):
            self._client = real_client(transport=transport)

        async def __aenter__(self):
            return self._client

        async def __aexit__(self, *a):
            await self._client.aclose()

        async def get(self, url, headers=None):
            seen["url"] = url
            return await self._client.get(url, headers=headers)

        async def post(self, *a, **k):
            raise AssertionError("health check must not POST")

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    assert await provider.health_check() is True
    assert seen["url"] == f"{LOCAL_BASE}/models"


@pytest.mark.asyncio
async def test_local_server_unavailable(monkeypatch):
    class Refused:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("connection refused")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Refused)
    provider, _ = _patched(lambda request: None)
    with pytest.raises(ProviderUnavailable, match="[Uu]nreachable"):
        await provider.send_message(_messages())


@pytest.mark.asyncio
async def test_local_timeout(monkeypatch):
    class Slow:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ReadTimeout("too slow")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Slow)
    provider, _ = _patched(lambda request: None)
    with pytest.raises(ProviderUnavailable, match="[Tt]imed out"):
        await provider.send_message(_messages())


@pytest.mark.asyncio
async def test_local_malformed_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": "no choices here"})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    with pytest.raises(ProviderUnavailable, match="[Mm]alformed"):
        await provider.send_message(_messages())


@pytest.mark.asyncio
async def test_local_tools_unsupported_falls_back_to_plain(monkeypatch):
    """llama.cpp may not support tool calls: fall back honestly, never fake."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                400, json={"error": {"message": "tool_choice not supported"}}
            )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "plain answer"}}]}
        )

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    resp = await provider.send_message_with_tools(
        _messages(),
        [LlmTool(name="system_info", description="x",
                 parameters={"type": "object", "properties": {}})],
    )
    assert resp.tool_calls == []  # nothing fabricated
    assert resp.content == "plain answer"
