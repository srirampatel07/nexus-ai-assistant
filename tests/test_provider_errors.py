"""Provider error categorization regression tests (Phase 6 audit).

Every failure mode must be distinguishable and honestly reported:
invalid definitions vs malformed requests vs auth vs rate-limit vs
upstream vs network vs timeout vs unsupported tools. Secrets redacted.
All HTTP mocked; offline.
"""

import httpx
import pytest

from app.brain.provider import (
    ChatMessage,
    OpenAICompatibleProvider,
    describe_http_error,
)
from app.core.exceptions import ProviderAuthError, ProviderUnavailable
from app.tools.schemas import LlmTool


def _provider():
    return OpenAICompatibleProvider(
        base_url="https://fake.test/v1", model="m", api_key="k"
    )


def _messages():
    return [ChatMessage(role="user", content="hi")]


def _tools():
    return [
        LlmTool(
            name="system_info",
            description="Get system info.",
            parameters={"type": "object", "properties": {}},
        )
    ]


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
                capture.requests.append(json)
                return capture._responses.pop(0)

        return FakeClient()


@pytest.mark.asyncio
async def test_400_invalid_name_raises_no_fallback(monkeypatch):
    """The exact live OpenRouter failure must surface, not fall back."""
    body = {
        "error": {
            "message": 'Function at index 0 has an invalid name: "system.info". '
            "Only a-z, A-Z, 0-9, underscores, and dashes are allowed."
        }
    }
    capture = _Capture([httpx.Response(400, json=body)])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await _provider().send_message_with_tools(_messages(), _tools())
    assert "HTTP 400" in str(exc_info.value)
    assert len(capture.requests) == 1  # no silent plain-text retry


@pytest.mark.asyncio
async def test_400_malformed_request_raises(monkeypatch):
    capture = _Capture([httpx.Response(400, json={"error": "bad request nonsense"})])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await _provider().send_message_with_tools(_messages(), _tools())
    assert "HTTP 400" in str(exc_info.value)
    assert len(capture.requests) == 1


@pytest.mark.asyncio
async def test_400_tools_unsupported_falls_back(monkeypatch):
    first = httpx.Response(
        400, json={"error": {"message": "tools not supported by this model"}}
    )
    second = httpx.Response(200, json={"choices": [{"message": {"content": "plain"}}]})
    capture = _Capture([first, second])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    resp = await _provider().send_message_with_tools(_messages(), _tools())
    assert resp.tool_calls == []
    assert resp.content == "plain"
    assert len(capture.requests) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failures_raise_auth_error(monkeypatch, status):
    capture = _Capture([httpx.Response(status, text="unauthorized")])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    with pytest.raises(ProviderAuthError):
        await _provider().send_message_with_tools(_messages(), _tools())


@pytest.mark.asyncio
async def test_429_rate_limit_is_not_hidden(monkeypatch):
    capture = _Capture([httpx.Response(429, text="rate limit exceeded, slow down")])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await _provider().send_message_with_tools(_messages(), _tools())
    assert "429" in str(exc_info.value)
    assert "rate limit" in str(exc_info.value).lower()
    assert len(capture.requests) == 1  # never masked as a good answer


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 503])
async def test_5xx_upstream_failures_raise(monkeypatch, status):
    capture = _Capture([httpx.Response(status, text="upstream boom")])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await _provider().send_message_with_tools(_messages(), _tools())
    assert f"HTTP {status}" in str(exc_info.value)


@pytest.mark.asyncio
async def test_network_error_is_unavailable(monkeypatch):
    class Boom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("conn refused")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Boom)
    with pytest.raises(ProviderUnavailable, match="[Uu]nreachable"):
        await _provider().send_message_with_tools(_messages(), _tools())


@pytest.mark.asyncio
async def test_timeout_is_unavailable(monkeypatch):
    class Slow:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ReadTimeout("too slow")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Slow)
    with pytest.raises(ProviderUnavailable, match="[Tt]imed out"):
        await _provider().send_message_with_tools(_messages(), _tools())


@pytest.mark.asyncio
async def test_plain_text_response_has_no_fabricated_calls(monkeypatch):
    capture = _Capture(
        [httpx.Response(200, json={"choices": [{"message": {"content": "just text"}}]})]
    )
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    resp = await _provider().send_message_with_tools(_messages(), _tools())
    assert resp.tool_calls == []
    assert resp.content == "just text"


def test_diagnostics_redact_secrets():
    msg = describe_http_error(
        429, "rate limited; Authorization: Bearer secret123; key gsk_abcdefgh12345678", "m"
    )
    assert "secret123" not in msg
    assert "gsk_abcdefgh12345678" not in msg
    assert "HTTP 429" in msg
    assert "rate limit" in msg.lower()
