"""NVIDIA cloud vision tests (NVIDIA-only operation).

All HTTP mocked — never uploads real images, never consumes quota, never
needs real credentials. Covers vision model selection, the OpenAI-compatible
image_url payload, validation, error categorization, no local fallback, and
no secret leakage.
"""

import base64

import httpx
import pytest

pytest.importorskip("PIL")

from pathlib import Path

from PIL import Image

from app.brain.provider import (
    NVIDIA_DEFAULT_BASE_URL,
    NVIDIA_DEFAULT_VISION_MODEL,
    resolve_nvidia_vision_settings,
)
from app.core.config import Settings
from app.core.exceptions import ProviderAuthError, ProviderUnavailable
from app.vision.describer import MetadataDescriber, OpenRouterVisionDescriber
from app.vision.errors import VisionUnavailableError
from app.vision.factories import create_describer, is_cloud_vision

VISION_BASE = NVIDIA_DEFAULT_BASE_URL
VISION_MODEL = NVIDIA_DEFAULT_VISION_MODEL


def _cloud_settings(**over):
    base = dict(
        ai_provider="nvidia",
        ai_model="nexus-default",
        ai_api_key="",
        ai_base_url="",
        ai_timeout_seconds=30.0,
        nvidia_api_key="test-nvapi-key",
        nvidia_base_url=NVIDIA_DEFAULT_BASE_URL,
        nvidia_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_vision_model=NVIDIA_DEFAULT_VISION_MODEL,
        vision_enabled=True,
        vision_provider="openai-compatible",
        vision_model="",
        vision_base_url="",
        vision_api_key="",
        nexus_allowed_roots=["C:\\chanti\\nexus"],
    )
    base.update(over)
    return Settings(**base)


def _patched(monkeypatch, handler, **desc_kw):
    params = dict(base_url=VISION_BASE, model=VISION_MODEL, api_key="test-nvapi-key")
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


def _make_png(path: Path, size=(48, 32)) -> Path:
    Image.new("RGB", size, (20, 40, 60)).save(path, format="PNG")
    return path


# -- model selection ---------------------------------------------------------

def test_default_vision_model_is_verified_nvidia_id():
    assert NVIDIA_DEFAULT_VISION_MODEL == "meta/llama-3.2-11b-vision-instruct"
    s = _cloud_settings()
    assert s.nvidia_vision_model == NVIDIA_DEFAULT_VISION_MODEL


def test_retired_model_override_falls_through_to_default():
    from app.brain.provider import nvidia_vision_stale_override

    retired = "nvidia/nemotron-nano-12b-v2-vl"
    _, model, _ = resolve_nvidia_vision_settings(
        _cloud_settings(nvidia_vision_model=retired)
    )
    assert model == NVIDIA_DEFAULT_VISION_MODEL
    _, model, _ = resolve_nvidia_vision_settings(
        _cloud_settings(vision_model=retired)
    )
    assert model == NVIDIA_DEFAULT_VISION_MODEL
    assert nvidia_vision_stale_override(_cloud_settings(nvidia_vision_model=retired)) == retired
    assert nvidia_vision_stale_override(_cloud_settings(vision_model=retired)) == retired
    assert nvidia_vision_stale_override(_cloud_settings()) is None


def test_cloud_vision_triggered_by_text_provider():
    assert is_cloud_vision(_cloud_settings()) is True
    assert is_cloud_vision(_cloud_settings(vision_provider="nvidia")) is True
    legacy = _cloud_settings(ai_provider="echo", vision_provider="openai-compatible")
    assert is_cloud_vision(legacy) is False


def test_factory_uses_nvidia_model_despite_local_leftovers():
    """Leftover Qwen :8081 config must never receive cloud images."""
    desc = create_describer(
        _cloud_settings(
            vision_provider="local",
            vision_base_url="http://127.0.0.1:8081/v1",
            vision_model="qwen2.5-vl-3b-local",
        )
    )
    assert isinstance(desc, OpenRouterVisionDescriber)
    assert desc.base_url == VISION_BASE
    assert desc.model == VISION_MODEL
    assert "127.0.0.1" not in desc.base_url


def test_factory_vision_key_prefers_vision_api_key():
    desc = create_describer(
        _cloud_settings(vision_api_key="vision-key-1", nvidia_api_key="nv-key")
    )
    assert desc._headers()["Authorization"] == "Bearer vision-key-1"


def test_factory_vision_key_falls_back_to_nvidia_key():
    desc = create_describer(_cloud_settings(vision_api_key="", nvidia_api_key="nv-key"))
    assert desc._headers()["Authorization"] == "Bearer nv-key"


def test_factory_vision_model_override_honored():
    desc = create_describer(_cloud_settings(vision_model="nvidia/custom-vl"))
    assert desc.model == "nvidia/custom-vl"


def test_factory_missing_key_is_explicit():
    with pytest.raises(VisionUnavailableError, match="NVIDIA_API_KEY|API key"):
        create_describer(_cloud_settings(vision_api_key="", nvidia_api_key="", ai_api_key=""))


def test_resolver_ignores_loopback_and_gguf_leftovers():
    base, model, _ = resolve_nvidia_vision_settings(
        _cloud_settings(
            vision_base_url="http://127.0.0.1:8081/v1",
            vision_model="qwen2.5-vl-3b-local",
        )
    )
    assert base == VISION_BASE
    assert model == VISION_MODEL


def test_non_nvidia_paths_unchanged():
    desc = create_describer(Settings(ai_provider="echo", vision_provider="openai-compatible"))
    assert isinstance(desc, MetadataDescriber)


# -- image request format -----------------------------------------------------

@pytest.mark.asyncio
async def test_request_uses_vision_model_and_data_uri(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.read().decode()))
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"choices": [{"message": {"content": "a cat"}}]})

    desc = _patched(monkeypatch, handler)
    out = await desc.describe(b"\xff\xd8fake", "image/jpeg", "What is it?")
    assert out.text == "a cat"
    assert out.model == VISION_MODEL
    assert seen["url"] == f"{VISION_BASE}/chat/completions"
    assert seen["model"] == VISION_MODEL
    assert seen["auth"] == "Bearer test-nvapi-key"
    content = seen["messages"][0]["content"]
    assert {p["type"] for p in content} == {"text", "image_url"}
    assert content[0]["text"] == "What is it?"
    url = next(p["image_url"]["url"] for p in content if p["type"] == "image_url")
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\xff\xd8fake"


@pytest.mark.asyncio
async def test_text_model_never_receives_images(monkeypatch):
    """Vision request carries the vision model even when text differs."""
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.read().decode()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    settings = _cloud_settings()
    desc = create_describer(settings)
    desc = _patched(monkeypatch, handler, base_url=desc.base_url, model=desc.model, api_key="test-nvapi-key")
    await desc.describe(b"img", "image/jpeg", "hi")
    assert seen["model"] == VISION_MODEL
    assert seen["model"] != "nvidia/nemotron-3.5-lightning-30b-a3b"


# -- validation (no HTTP on bad input) ----------------------------------------

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
    desc = OpenRouterVisionDescriber(base_url=VISION_BASE, model=VISION_MODEL, api_key="k")
    with pytest.raises(ProviderUnavailable):
        await desc.describe(b"", "image/jpeg", "hi")
    with pytest.raises(ProviderUnavailable):
        await desc.describe(b"img", "image/gif", "hi")
    assert capture == []


@pytest.mark.asyncio
async def test_oversized_local_image_rejected_before_upload(tmp_path):
    from app.vision.loader import load_validated_image
    from app.vision.errors import ImageRejectedError

    src = _make_png(tmp_path / "big.png", size=(64, 64))
    with pytest.raises(ImageRejectedError):
        load_validated_image(str(src), allowed_roots=[str(tmp_path)], max_bytes=10, max_dim=2048)


# -- errors (honest, categorized, redacted) ------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_failures_raise_auth(monkeypatch, status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, text="bad credentials")

    desc = _patched(monkeypatch, handler)
    with pytest.raises(ProviderAuthError):
        await desc.describe(b"img", "image/jpeg", "hi")


@pytest.mark.asyncio
async def test_400_unsupported_image_is_honest(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "This model does not support images."}})

    desc = _patched(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await desc.describe(b"img", "image/jpeg", "hi")
    assert "HTTP 400" in str(exc_info.value)


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
        await OpenRouterVisionDescriber(base_url=VISION_BASE, model=VISION_MODEL, api_key="k").describe(
            b"img", "image/jpeg", "hi"
        )

    class ConnBoom:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("refused")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", ConnBoom)
    with pytest.raises(ProviderUnavailable, match="[Uu]nreachable"):
        await OpenRouterVisionDescriber(base_url=VISION_BASE, model=VISION_MODEL, api_key="k").describe(
            b"img", "image/jpeg", "hi"
        )


@pytest.mark.asyncio
async def test_malformed_and_empty_responses_raise(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json={"nope": "no choices"})
        return httpx.Response(200, json={"choices": [{"message": {}}]})

    desc = _patched(monkeypatch, handler)
    with pytest.raises(ProviderUnavailable):
        await desc.describe(b"img", "image/jpeg", "hi")
    with pytest.raises(ProviderUnavailable):
        await desc.describe(b"img", "image/jpeg", "hi")


def test_error_hides_pixels_and_key(monkeypatch):
    import asyncio

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server boom")

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
    with pytest.raises(ProviderUnavailable) as exc_info:
        asyncio.run(
            OpenRouterVisionDescriber(
                base_url=VISION_BASE, model=VISION_MODEL, api_key="test-nvapi-key"
            ).describe(b"pixel-marker-bytes", "image/jpeg", "hi")
        )
    assert "pixel-marker-bytes" not in str(exc_info.value)
    assert "test-nvapi-key" not in str(exc_info.value)


# -- assistant integration (cloud vision + cloud text, no local) ---------------

class _FakeCloudVision:
    def __init__(self, text="a red bicycle", model=VISION_MODEL):
        self.text = text
        self.model = model
        self.calls = 0

    async def describe(self, image_data, mime, question):
        self.calls += 1
        assert image_data[:2] == b"\xff\xd8"
        assert mime == "image/jpeg"
        from app.vision.describer import VisualDescription

        return VisualDescription(text=self.text, model=self.model)


@pytest.mark.asyncio
async def test_assistant_vision_flows_through_nvidia_text(monkeypatch, tmp_path):
    import json

    from app.brain.assistant import NexusAssistant

    src = _make_png(tmp_path / "bike.png")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.read().decode()))
        return httpx.Response(200, json={"choices": [{"message": {"content": "synthesized answer"}}]})

    from app.brain.provider import NvidiaNemotronProvider

    text_provider = NvidiaNemotronProvider(
        base_url=NVIDIA_DEFAULT_BASE_URL,
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key="test-nvapi-key",
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
            return await self._client.get(url, headers=headers)

    # Text brain healthy so synthesis runs (mock /models + chat).
    async def fake_health():
        return True

    monkeypatch.setattr(text_provider, "health_check", fake_health)
    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    settings = _cloud_settings(nexus_allowed_roots=[str(tmp_path)])
    assistant = NexusAssistant(
        settings=settings, provider=text_provider, vision_describer=_FakeCloudVision()
    )
    reply = await assistant.achat_with_image(str(src), "What is in the image?")
    assert reply == "synthesized answer"
    assert "extra_body" not in seen  # text payload keeps the flattened fix
    history = assistant.get_history()
    assert len(history) == 2
    assert "bike.png" in history[0].content
    assert "\xff" not in history[0].content


@pytest.mark.asyncio
async def test_assistant_vision_failure_is_honest_no_local_fallback(tmp_path):
    from app.brain.assistant import NexusAssistant
    from app.brain.provider import NvidiaNemotronProvider
    from app.vision.describer import VisualDescription

    class FailingVision:
        async def describe(self, image_data, mime, question):
            from app.core.exceptions import ProviderUnavailable

            raise ProviderUnavailable("AI provider error (HTTP 429, rate limited / quota exhausted)")

    src = _make_png(tmp_path / "pic.png")
    settings = _cloud_settings(nexus_allowed_roots=[str(tmp_path)])
    assistant = NexusAssistant(
        settings=settings,
        provider=NvidiaNemotronProvider(
            base_url=NVIDIA_DEFAULT_BASE_URL,
            model="nvidia/nemotron-3.5-lightning-30b-a3b",
            api_key="test-nvapi-key",
        ),
        vision_describer=FailingVision(),
    )
    reply = await assistant.achat_with_image(str(src), "What is this?")
    assert "couldn't analyze" in reply.lower()
    assert "127.0.0.1" not in reply
    assert "8081" not in reply
    assert not hasattr(assistant, "synthesis_handoff")
