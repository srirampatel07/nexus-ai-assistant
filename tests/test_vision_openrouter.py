"""OpenRouter vision regression tests (vision switch).

Vision uses VISION_MODEL (default `openrouter/free`) through VISION_BASE_URL
(or AI_BASE_URL when unset), reusing the chat AI_API_KEY — no second vision
credential. The chat model/settings are untouched. All HTTP mocked; offline.
"""

import base64

import httpx
import pytest

from app.brain.provider import create_provider
from app.core.config import Settings
from app.core.exceptions import ProviderAuthError, ProviderUnavailable
from app.vision.describer import (
    GroqVisionDescriber,
    MetadataDescriber,
    OpenRouterVisionDescriber,
)
from app.vision.errors import ImageRejectedError
from app.vision.factories import create_describer
from app.vision.loader import load_validated_image


def _settings(**kw):
    # Pin every vision field: these tests describe hosted behavior and must
    # not inherit a real user .env (e.g. VISION_PROVIDER=local).
    base = dict(
        ai_provider="openai-compatible",
        ai_model="openrouter/free",
        ai_api_key="chat-key-123",
        ai_base_url="https://openrouter.ai/api/v1",
        vision_provider="openai-compatible",
        vision_model="openrouter/free",
        vision_base_url="",
        vision_api_key="",
    )
    base.update(kw)
    return Settings(**base)


def _patched(monkeypatch, handler, **desc_kw):
    params = dict(
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/free",
        api_key="chat-key-123",
    )
    params.update(desc_kw)
    desc = OpenRouterVisionDescriber(**params)
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


# -- request construction ------------------------------------------------

@pytest.mark.asyncio
async def test_request_uses_vision_model_and_data_uri(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.read().decode()))
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "a cat"}}]}
        )

    desc = _patched(monkeypatch, handler)
    out = await desc.describe(b"\xff\xd8fake", "image/jpeg", "What is it?")
    assert out.text == "a cat"
    assert out.model == "openrouter/free"
    assert seen["model"] == "openrouter/free"
    content = seen["messages"][0]["content"]
    assert {p["type"] for p in content} == {"text", "image_url"}
    assert content[0]["text"] == "What is it?"
    url = next(p["image_url"]["url"] for p in content if p["type"] == "image_url")
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8fake"
    # Same chat key reused for vision — no second credential.
    assert seen["auth"] == "Bearer chat-key-123"


@pytest.mark.asyncio
async def test_chat_model_never_receives_images(monkeypatch):
    """Vision request carries VISION_MODEL even when AI_MODEL differs."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.read().decode()))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    settings = _settings(ai_model="chat-model-x", vision_model="vision-model-y")
    desc = create_describer(settings)
    assert desc.model == "vision-model-y"
    # Rebuild against the mock transport with the factory-chosen values.
    desc = _patched(
        monkeypatch, handler, base_url="https://openrouter.ai/api/v1",
        model=desc.model, api_key="chat-key-123",
    )
    await desc.describe(b"img", "image/jpeg", "hi")
    assert seen["model"] == "vision-model-y"
    assert seen["model"] != "chat-model-x"
    # And the chat provider still uses the chat model.
    assert create_provider(settings)._model == "chat-model-x"


# -- endpoint separation ---------------------------------------------------

def test_vision_base_url_override():
    s = _settings(
        ai_base_url="https://openrouter.ai/api/v1",
        vision_base_url="https://custom.example/v1",
    )
    desc = create_describer(s)
    assert isinstance(desc, OpenRouterVisionDescriber)
    assert desc.base_url == "https://custom.example/v1"


def test_vision_base_url_empty_reuses_chat_base():
    desc = create_describer(_settings(vision_base_url=""))
    assert isinstance(desc, OpenRouterVisionDescriber)
    assert desc.base_url == "https://openrouter.ai/api/v1"


def test_factory_echo_is_offline():
    assert isinstance(
        create_describer(Settings(ai_provider="echo",
                                  vision_provider="openai-compatible")),
        MetadataDescriber)


def test_legacy_groq_name_still_resolves():
    assert GroqVisionDescriber is OpenRouterVisionDescriber


# -- transparent errors (never fabricated) ----------------------------------

@pytest.mark.asyncio
async def test_400_capability_error_is_honest(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            400,
            json={"error": {"message": "This model does not support images."}},
        )

    desc = _patched(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await desc.describe(b"img", "image/jpeg", "hi")
    assert "HTTP 400" in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failures_raise_auth(monkeypatch, status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="bad credentials")

    desc = _patched(monkeypatch, handler)
    with pytest.raises(ProviderAuthError):
        await desc.describe(b"img", "image/jpeg", "hi")


@pytest.mark.asyncio
async def test_429_rate_limit_is_honest(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="rate limit exceeded")

    desc = _patched(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await desc.describe(b"img", "image/jpeg", "hi")
    assert "429" in str(exc_info.value)
    assert "rate limit" in str(exc_info.value).lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [500, 502, 503])
async def test_upstream_failures_raise(monkeypatch, status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="upstream boom")

    desc = _patched(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await desc.describe(b"img", "image/jpeg", "hi")
    assert f"HTTP {status}" in str(exc_info.value)


@pytest.mark.asyncio
async def test_timeout_and_network_are_unavailable(monkeypatch):
    class TimeoutBoom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ReadTimeout("too slow")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", TimeoutBoom)
    with pytest.raises(ProviderUnavailable, match="[Tt]imed out"):
        await OpenRouterVisionDescriber(
            base_url="https://openrouter.ai/api/v1",
            model="openrouter/free",
            api_key="k",
        ).describe(b"img", "image/jpeg", "hi")

    class ConnBoom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("refused")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", ConnBoom)
    with pytest.raises(ProviderUnavailable, match="[Uu]nreachable"):
        await OpenRouterVisionDescriber(
            base_url="https://openrouter.ai/api/v1",
            model="openrouter/free",
            api_key="k",
        ).describe(b"img", "image/jpeg", "hi")


@pytest.mark.asyncio
async def test_empty_description_is_never_fabricated(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {}}]})

    desc = _patched(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable):
        await desc.describe(b"img", "image/jpeg", "hi")


@pytest.mark.asyncio
async def test_empty_bytes_and_bad_mime_rejected_locally(monkeypatch):
    capture = []

    class NoNet:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            capture.append(1)
            raise AssertionError("no HTTP expected")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", NoNet)
    desc = OpenRouterVisionDescriber(
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/free",
        api_key="k",
    )
    with pytest.raises(ProviderUnavailable):
        await desc.describe(b"", "image/jpeg", "hi")
    with pytest.raises(ProviderUnavailable):
        await desc.describe(b"img", "image/gif", "hi")
    assert capture == []


@pytest.mark.asyncio
async def test_error_hides_pixels_and_key(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server boom")

    desc = _patched(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await desc.describe(b"pixel-marker-bytes", "image/jpeg", "hi")
    assert "pixel-marker-bytes" not in str(exc_info.value)
    assert "chat-key-123" not in str(exc_info.value)


# -- image-security restrictions preserved -----------------------------------

def test_loader_restrictions_preserved(tmp_path):
    from PIL import Image

    good = tmp_path / "ok.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(good)
    validated = load_validated_image(
        str(good), allowed_roots=[str(tmp_path)],
        max_bytes=10_485_760, max_dim=2048,
    )
    assert validated.mime == "image/jpeg"
    assert validated.data[:2] == b"\xff\xd8"

    outside = tmp_path / "elsewhere.png"
    Image.new("RGB", (8, 8)).save(outside)
    with pytest.raises(ImageRejectedError):
        load_validated_image(
            str(outside), allowed_roots=[str(tmp_path / "nonexistent")],
            max_bytes=10_485_760, max_dim=2048,
        )
    bad_ext = tmp_path / "evil.txt"
    bad_ext.write_text("not an image")
    with pytest.raises(ImageRejectedError):
        load_validated_image(
            str(bad_ext), allowed_roots=[str(tmp_path)],
            max_bytes=10_485_760, max_dim=2048,
        )
