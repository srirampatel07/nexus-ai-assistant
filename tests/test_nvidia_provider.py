"""NVIDIA Nemotron provider tests (cloud integration).

All HTTP mocked — never touches the live NVIDIA endpoint, never consumes
quota, never needs real credentials. Covers the exact reference values
(base URL, model, Bearer auth, reasoning params), tool calling, streaming
compatibility, error categorization, health_detail, and factory wiring.
"""

import httpx
import pytest

from app.brain.provider import (
    NVIDIA_DEFAULT_BASE_URL,
    NVIDIA_DEFAULT_MODEL,
    ChatMessage,
    NvidiaNemotronProvider,
    create_provider,
    resolve_nvidia_settings,
)
from app.core.config import Settings
from app.core.exceptions import ConfigError, ProviderAuthError, ProviderUnavailable
from app.tools.schemas import LlmTool


def _nvidia_settings(**over):
    base = dict(
        ai_provider="nvidia",
        ai_model="nexus-default",
        ai_api_key="",
        ai_base_url="",
        ai_timeout_seconds=30.0,
        nvidia_api_key="test-nvapi-key",
        nvidia_base_url=NVIDIA_DEFAULT_BASE_URL,
        nvidia_model=NVIDIA_DEFAULT_MODEL,
    )
    base.update(over)
    return Settings(**base)


def _provider(**kw):
    params = dict(
        base_url=NVIDIA_DEFAULT_BASE_URL,
        model=NVIDIA_DEFAULT_MODEL,
        api_key="test-nvapi-key",
        timeout=30.0,
    )
    params.update(kw)
    return NvidiaNemotronProvider(**params)


def _messages():
    return [ChatMessage(role="user", content="hello nvidia")]


def _tools():
    return [
        LlmTool(
            name="system_info",
            description="Get system info.",
            parameters={"type": "object", "properties": {}},
        )
    ]


def _patched(handler, **provider_kw):
    provider = _provider(**provider_kw)
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
            return await self._client.get(url, headers=headers)

    return provider, FakeClient


class _Capture:
    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def __call__(self, *a, **k):
        capture = self

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                capture.requests.append({"url": url, "headers": headers, "json": json})
                return capture._responses.pop(0)

            async def get(self, url, headers=None):
                capture.requests.append({"url": url, "headers": headers})
                return capture._responses.pop(0)

        return FakeClient()


# -- factory / resolution ----------------------------------------------------

def test_factory_creates_nvidia_with_official_defaults():
    p = create_provider(_nvidia_settings())
    assert isinstance(p, NvidiaNemotronProvider)
    assert p.name == "nvidia"
    assert p._base_url == NVIDIA_DEFAULT_BASE_URL
    assert p._model == NVIDIA_DEFAULT_MODEL
    assert p._base_url == "https://integrate.api.nvidia.com/v1"
    assert p._model == "nvidia/nemotron-3.5-lightning-30b-a3b"


def test_factory_accepts_aliases():
    for alias in ("nvidia", "NVIDIA", "nvidia-nemotron", "nemotron"):
        p = create_provider(_nvidia_settings(ai_provider=alias))
        assert isinstance(p, NvidiaNemotronProvider)


def test_factory_prefers_nvidia_key_over_ai_key():
    s = _nvidia_settings(nvidia_api_key="nv-key", ai_api_key="ai-key")
    _, _, key = resolve_nvidia_settings(s)
    assert key == "nv-key"


def test_factory_falls_back_to_ai_key():
    s = _nvidia_settings(nvidia_api_key="", ai_api_key="ai-fallback")
    _, _, key = resolve_nvidia_settings(s)
    assert key == "ai-fallback"


def test_factory_ai_base_url_overrides_default():
    s = _nvidia_settings(ai_base_url="https://custom.example/v1")
    base, _, _ = resolve_nvidia_settings(s)
    assert base == "https://custom.example/v1"


def test_factory_ignores_loopback_base_url_leftover():
    s = _nvidia_settings(ai_base_url="http://127.0.0.1:8080/v1")
    base, _, _ = resolve_nvidia_settings(s)
    assert base == NVIDIA_DEFAULT_BASE_URL


def test_factory_ignores_local_gguf_model_leftover():
    local_path = (
        r"C:\Users\shash\.cache\huggingface\hub\snapshots\abc\Qwen3-8B-Q4_K_M.gguf"
    )
    s = _nvidia_settings(ai_model=local_path)
    _, model, _ = resolve_nvidia_settings(s)
    assert model == NVIDIA_DEFAULT_MODEL


def test_factory_ai_model_overrides_default():
    s = _nvidia_settings(ai_model="custom-model")
    _, model, _ = resolve_nvidia_settings(s)
    assert model == "custom-model"


def test_factory_placeholder_model_uses_nvidia_default():
    s = _nvidia_settings(ai_model="nexus-default")
    _, model, _ = resolve_nvidia_settings(s)
    assert model == NVIDIA_DEFAULT_MODEL


def test_missing_key_raises_auth_error():
    with pytest.raises(ProviderAuthError, match="NVIDIA_API_KEY"):
        _provider(api_key="")
    with pytest.raises(ProviderAuthError):
        create_provider(_nvidia_settings(nvidia_api_key="", ai_api_key=""))


def test_unknown_provider_still_rejected():
    with pytest.raises(ConfigError):
        create_provider(Settings(ai_provider="nope"))


def test_safe_dump_redacts_nvidia_key():
    s = _nvidia_settings(nvidia_api_key="super-secret-nv")
    dumped = s.model_dump_safe()
    assert dumped["nvidia_api_key"] == "***"
    assert "super-secret-nv" not in str(dumped)


# -- request shape (reference params) ----------------------------------------

@pytest.mark.asyncio
async def test_reference_params_sent(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.read().decode()))
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    resp = await provider.send_message(_messages())
    assert resp.content == "hi"
    assert resp.provider == "nvidia"
    assert resp.model == NVIDIA_DEFAULT_MODEL
    assert seen["url"] == f"{NVIDIA_DEFAULT_BASE_URL}/chat/completions"
    assert seen["model"] == NVIDIA_DEFAULT_MODEL
    assert seen["auth"] == "Bearer test-nvapi-key"
    assert seen["temperature"] == 1.0
    assert seen["top_p"] == 0.95
    assert seen["max_tokens"] == 16384
    # SDK extra_body contents are flattened top-level (raw HTTP has no
    # SDK helper): a literal "extra_body" key is rejected by NVIDIA
    # (HTTP 400 Unsupported parameter(s): extra_body).
    assert "extra_body" not in seen
    assert seen["chat_template_kwargs"] == {"enable_thinking": True}
    assert seen["reasoning_budget"] == 16384


@pytest.mark.asyncio
async def test_no_extra_body_wrapper_in_payload(monkeypatch):
    """Regression for live HTTP 400: extra_body must never appear literally."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.read().decode())
        assert "extra_body" not in body, f"literal extra_body rejected: {sorted(body)}"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    assert (await provider.send_message(_messages())).content == "ok"


@pytest.mark.asyncio
async def test_reasoning_params_merged_top_level(monkeypatch):
    """Supported custom params sit alongside model/messages, not nested."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.read().decode())
        for key in ("model", "messages", "temperature", "top_p",
                    "max_tokens", "chat_template_kwargs", "reasoning_budget"):
            assert key in body, f"missing top-level {key}: {sorted(body)}"
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        assert body["reasoning_budget"] == 16384
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    await provider.send_message(_messages())


@pytest.mark.asyncio
async def test_reasoning_content_tolerated(monkeypatch):
    """Thinking models may return reasoning_content alongside content."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "final answer",
                            "reasoning_content": "internal chain of thought",
                        }
                    }
                ],
                "usage": {"total_tokens": 5},
            },
        )

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    resp = await provider.send_message(_messages())
    assert resp.content == "final answer"
    assert resp.usage == {"total_tokens": 5}


@pytest.mark.asyncio
async def test_stream_message_yields_completed_reply(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "streamed"}}]})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    chunks = [c async for c in provider.stream_message(_messages())]
    assert chunks == ["streamed"]


# -- tool calling ------------------------------------------------------------

@pytest.mark.asyncio
async def test_tool_params_and_parsing(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.read().decode()))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "system_info",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    resp = await provider.send_message_with_tools(_messages(), _tools())
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "system_info"
    # Same corrected payload logic on tool calls: flattened reasoning
    # params, no literal wrapper.
    assert "extra_body" not in seen
    assert seen["temperature"] == 1.0
    assert seen["top_p"] == 0.95
    assert seen["max_tokens"] == 16384
    assert seen["chat_template_kwargs"] == {"enable_thinking": True}
    assert seen["reasoning_budget"] == 16384
    tools = seen["tools"]
    assert tools[0]["function"]["name"] == "system_info"
    assert seen["tool_choice"] == "auto"


@pytest.mark.asyncio
async def test_tool_request_has_no_extra_body_wrapper(monkeypatch):
    """Tool-call requests use the same corrected payload logic."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.read().decode())
        assert "extra_body" not in body
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        assert body["reasoning_budget"] == 16384
        assert "tools" in body and "tool_choice" in body
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    resp = await provider.send_message_with_tools(_messages(), _tools())
    assert resp.content == "ok"


@pytest.mark.asyncio
async def test_request_body_contains_no_key_material(monkeypatch):
    """Bearer key travels in headers only; JSON body must not contain it."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.read().decode())
        assert "test-nvapi-key" not in json.dumps(body)
        assert request.headers.get("authorization") == "Bearer test-nvapi-key"
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    await provider.send_message(_messages())
    await provider.send_message_with_tools(_messages(), _tools())


def test_base_openai_compatible_provider_sends_no_reasoning_params():
    """Generic transport path unchanged: no NVIDIA extras leaked."""
    from app.brain.provider import OpenAICompatibleProvider

    assert OpenAICompatibleProvider(
        base_url="https://example.com/v1", model="m", api_key="k"
    )._extra_payload() == {}


@pytest.mark.asyncio
async def test_plain_text_has_no_fabricated_calls(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "just text"}}]})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    resp = await provider.send_message_with_tools(_messages(), _tools())
    assert resp.tool_calls == []
    assert resp.content == "just text"


@pytest.mark.asyncio
async def test_tools_unsupported_falls_back(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                400, json={"error": {"message": "tools not supported by this model"}}
            )
        return httpx.Response(200, json={"choices": [{"message": {"content": "plain"}}]})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    resp = await provider.send_message_with_tools(_messages(), _tools())
    assert resp.tool_calls == []
    assert resp.content == "plain"


# -- errors (no silent fallback, no secret leak) ------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failures_raise_auth_error(monkeypatch, status):
    capture = _Capture(
        [
            httpx.Response(status, text="unauthorized"),
            httpx.Response(status, text="unauthorized"),
        ]
    )
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    with pytest.raises(ProviderAuthError):
        await _provider().send_message(_messages())
    with pytest.raises(ProviderAuthError):
        await _provider().send_message_with_tools(_messages(), _tools())


@pytest.mark.asyncio
async def test_429_rate_limit_preserved(monkeypatch):
    capture = _Capture([httpx.Response(429, text="rate limit exceeded")])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await _provider().send_message(_messages())
    assert "429" in str(exc_info.value)
    assert "rate limit" in str(exc_info.value).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 503])
async def test_5xx_raise_unavailable(monkeypatch, status):
    capture = _Capture([httpx.Response(status, text="upstream boom")])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await _provider().send_message(_messages())
    assert f"HTTP {status}" in str(exc_info.value)


@pytest.mark.asyncio
async def test_timeout_and_network(monkeypatch):
    class Slow:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ReadTimeout("too slow")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Slow)
    with pytest.raises(ProviderUnavailable, match="[Tt]imed out"):
        await _provider().send_message(_messages())

    class Refused:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("refused")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Refused)
    with pytest.raises(ProviderUnavailable, match="[Uu]nreachable"):
        await _provider().send_message(_messages())


@pytest.mark.asyncio
async def test_malformed_response(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"nope": "no choices"})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    with pytest.raises(ProviderUnavailable, match="[Mm]alformed"):
        await provider.send_message(_messages())


def test_error_messages_never_contain_key():
    from app.brain.provider import describe_http_error

    msg = describe_http_error(401, "bad key nvapi-secret-marker-123", "m")
    assert "nvapi-secret-marker-123" not in msg
    assert "test-nvapi-key" not in msg
    assert "HTTP 401" in msg


# -- health_detail categories -------------------------------------------------

@pytest.mark.asyncio
async def test_health_detail_ok(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{NVIDIA_DEFAULT_BASE_URL}/models"
        assert request.headers.get("authorization") == "Bearer test-nvapi-key"
        return httpx.Response(200, json={"data": [{"id": NVIDIA_DEFAULT_MODEL}]})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    assert await provider.health_check() is True
    detail = await provider.health_detail()
    assert detail["ok"] is True
    assert detail["status"] == "ok"
    assert NVIDIA_DEFAULT_MODEL in detail["detail"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected", [(401, "auth_error"), (403, "auth_error")])
async def test_health_detail_auth(monkeypatch, status, expected):
    capture = _Capture([httpx.Response(status, text="bad key")])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    detail = await _provider().health_detail()
    assert detail["ok"] is False
    assert detail["status"] == expected
    assert "test-nvapi-key" not in detail["detail"]


@pytest.mark.asyncio
async def test_health_detail_rate_limited(monkeypatch):
    capture = _Capture([httpx.Response(429, text="slow down")])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    detail = await _provider().health_detail()
    assert detail == {"ok": False, "status": "rate_limited", "detail": detail["detail"]}
    assert "429" in detail["detail"]


@pytest.mark.asyncio
async def test_health_detail_network_and_timeout(monkeypatch):
    class Refused:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("refused")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Refused)
    detail = await _provider().health_detail()
    assert detail["status"] == "network_error"
    assert detail["ok"] is False

    class Slow:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ReadTimeout("slow")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Slow)
    detail = await _provider().health_detail()
    assert detail["status"] == "timeout"


@pytest.mark.asyncio
async def test_health_detail_provider_error(monkeypatch):
    capture = _Capture([httpx.Response(500, text="boom")])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    detail = await _provider().health_detail()
    assert detail["status"] == "provider_error"
    assert detail["ok"] is False


# -- assistant integration (history / system prompt preserved) ---------------

@pytest.mark.asyncio
async def test_assistant_chat_uses_history_and_system_prompt(monkeypatch):
    from app.brain.assistant import NexusAssistant

    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.read().decode()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "nvidia reply"}}]})

    provider, fake = _patched(handler)
    monkeypatch.setattr(httpx, "AsyncClient", fake)
    settings = _nvidia_settings()
    assistant = NexusAssistant(settings=settings, provider=provider)
    reply = await assistant.achat("Explain GPU computing briefly.")
    assert reply == "nvidia reply"
    # System instructions + history + user request are all sent.
    roles = [m["role"] for m in seen["messages"]]
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert seen["messages"][-1]["content"] == "Explain GPU computing briefly."
    # History retained for the next turn.
    assert len(assistant.get_history()) == 2


@pytest.mark.asyncio
async def test_assistant_reports_provider_failure_honestly(monkeypatch):
    from app.brain.assistant import NexusAssistant

    capture = _Capture([httpx.Response(429, text="rate limit exceeded")])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    settings = _nvidia_settings()
    assistant = NexusAssistant(settings=settings, provider=_provider())
    reply = await assistant.achat("Tell me about GPUs.")
    # No silent fallback to another provider: honest outage reply.
    assert "trouble reaching" in reply.lower()
