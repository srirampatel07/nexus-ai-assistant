"""Vision describer tests (Phase 5.0). HTTP mocked; no images uploaded."""

import base64

import httpx
import pytest

from app.core.exceptions import ProviderAuthError, ProviderUnavailable
from app.vision.describer import (
    GroqVisionDescriber,
    MetadataDescriber,
    VisualDescription,
)
from app.vision.factories import create_describer
from app.core.config import Settings


def _patched_describer(monkeypatch, handler) -> GroqVisionDescriber:
    desc = GroqVisionDescriber(
        base_url="https://fake.test/v1",
        model="qwen/qwen3.8-27b",
        api_key="k",
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

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    return desc


@pytest.mark.asyncio
async def test_groq_payload_uses_vision_model_and_data_uri(monkeypatch):
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.read().decode()))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "a red door"}}]}
        )

    desc = _patched_describer(monkeypatch, handler)
    out = await desc.describe(b"\xff\xd8fake", "image/jpeg", "What is it?")
    assert out.text == "a red door"
    assert out.model == "qwen/qwen3.8-27b"
    assert seen["model"] == "qwen/qwen3.8-27b"
    content = seen["messages"][0]["content"]
    kinds = {p["type"] for p in content}
    assert {"text", "image_url"} <= kinds
    url = next(p["image_url"]["url"] for p in content if p["type"] == "image_url")
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8fake"


@pytest.mark.asyncio
async def test_groq_401_raises_auth(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    desc = _patched_describer(monkeypatch, handler)
    with pytest.raises(ProviderAuthError):
        await desc.describe(b"img", "image/jpeg", "hi")


@pytest.mark.asyncio
async def test_groq_error_hides_request_bytes(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server boom")

    desc = _patched_describer(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable) as exc:
        await desc.describe(b"bytes-should-never-appear", "image/jpeg", "hi")
    assert "bytes-should-never-appear" not in str(exc.value)


@pytest.mark.asyncio
async def test_groq_empty_content_raises(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {}}]})

    desc = _patched_describer(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable):
        await desc.describe(b"img", "image/jpeg", "hi")


@pytest.mark.asyncio
async def test_groq_truncates_runaway_description(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "x" * 9000}}]}
        )

    desc = _patched_describer(monkeypatch, handler)
    out = await desc.describe(b"img", "image/jpeg", "hi")
    assert len(out.text) <= 4001


@pytest.mark.asyncio
async def test_metadata_describer_is_offline():
    desc = MetadataDescriber(width=64, height=48)
    out = await desc.describe(b"12345", "image/jpeg", "What?")
    assert isinstance(out, VisualDescription)
    assert out.model == "metadata-offline"
    assert "Offline" in out.text
    assert "64x48" in out.text


def test_factory_echo_is_offline():
    desc = create_describer(Settings(ai_provider="echo"))
    assert isinstance(desc, MetadataDescriber)


def test_factory_openai_uses_vision_model():
    s = Settings(
        ai_provider="openai-compatible",
        ai_api_key="k",
        ai_base_url="https://api.groq.com/openai/v1",
        vision_model="qwen/qwen3.8-27b",
    )
    desc = create_describer(s)
    assert isinstance(desc, GroqVisionDescriber)
    assert desc.model == "qwen/qwen3.8-27b"


def test_groq_requires_credentials():
    with pytest.raises(ProviderAuthError):
        GroqVisionDescriber(base_url="", model="m", api_key="k")
    with pytest.raises(ProviderAuthError):
        GroqVisionDescriber(base_url="https://x/v1", model="m", api_key="")


@pytest.mark.asyncio
async def test_describe_logs_model_status_without_secrets(monkeypatch, caplog):
    """Diagnostics carry model/finish/preview only — never keys or pixels."""
    secret = b"\xff\xd8diag-marker-bytes"
    secret_b64 = base64.b64encode(secret).decode("ascii")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"content": "a red door"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    desc = _patched_describer(monkeypatch, handler)
    with caplog.at_level("DEBUG", logger="nexus.vision"):
        out = await desc.describe(secret, "image/jpeg", "What is it?")
    assert out.text == "a red door"
    assert "qwen/qwen3.8-27b" in caplog.text
    assert "stop" in caplog.text
    assert "a red door"[:20] in caplog.text
    assert secret_b64 not in caplog.text
    assert "Bearer" not in caplog.text
