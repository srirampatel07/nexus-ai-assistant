"""OpenAI-compatible transport tests (mocked HTTP, no servers, no inference).

These exercise the shared transport also used by the NVIDIA provider:
exact endpoint/model use, Bearer handling, health checks, error mapping,
and honest tool-unsupported fallback. All HTTP is mocked via
httpx.MockTransport — the suite never needs a running server.
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

MOCK_BASE = "https://mock.test/v1"
MOCK_MODEL = "mock-model-1"


def _mock_settings(**over):
    kwargs = dict(
        ai_provider="openai-compatible",
        ai_model=MOCK_MODEL,
        ai_api_key="",
        ai_base_url=MOCK_BASE,
    )
    kwargs.update(over)
    return Settings(**kwargs)


def _patched(handler, **provider_kw):
    params = dict(base_url=MOCK_BASE, model=MOCK_MODEL, api_key="")
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
    return [ChatMessage(role="user", content="hello mock")]


def test_transport_settings_build_compatible_provider_without_key():
    provider = create_provider(_mock_settings())
    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider._base_url == MOCK_BASE
    assert provider._model == MOCK_MODEL
    assert "Authorization" not in provider._headers()


@pytest.mark.asyncio
async def test_transport_base_url_and_model_used(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        import json

        seen.update(json.loads(request.read().decode()))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "hi from mock"}}]}
        )

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    resp = await provider.send_message(_messages())
    assert resp.content == "hi from mock"
    assert seen["url"] == f"{MOCK_BASE}/chat/completions"
    assert seen["model"] == MOCK_MODEL
    assert seen["auth"] is None  # empty key: no credential sent


@pytest.mark.asyncio
async def test_transport_health_check_hits_models_endpoint(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"data": [{"id": MOCK_MODEL}]})

    provider = OpenAICompatibleProvider(
        base_url=MOCK_BASE, model=MOCK_MODEL, api_key=""
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
    assert seen["url"] == f"{MOCK_BASE}/models"


@pytest.mark.asyncio
async def test_transport_server_unavailable(monkeypatch):
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
async def test_transport_timeout(monkeypatch):
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
async def test_transport_malformed_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": "no choices here"})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    with pytest.raises(ProviderUnavailable, match="[Mm]alformed"):
        await provider.send_message(_messages())


@pytest.mark.asyncio
async def test_transport_tools_unsupported_falls_back_to_plain(monkeypatch):
    """Endpoints without tool support fall back honestly, never fake."""
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
