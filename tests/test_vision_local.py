"""Local vision mode tests (`vision_provider=local`).

Local vision analyzes images on a loopback llama.cpp server (e.g. a
Qwen2.5-VL GGUF + mmproj on :8081) with an empty key. Images must never
reach a hosted service in this mode, and missing pieces must raise
explicit errors instead of silently downgrading. All HTTP mocked.
"""

import httpx
import pytest

from app.brain.provider import create_provider
from app.core.config import Settings
from app.vision.describer import MetadataDescriber, OpenRouterVisionDescriber
from app.vision.errors import VisionUnavailableError
from app.vision.factories import create_describer, is_local_url

LOCAL_BASE = "http://127.0.0.1:8081/v1"
LOCAL_MODEL = "qwen2.5-vl-3b-local"


def _local_settings(**over):
    kwargs = dict(
        ai_provider="openai-compatible",
        ai_model="text-model",
        ai_api_key="",
        ai_base_url="http://127.0.0.1:8080/v1",
        vision_provider="local",
        vision_model=LOCAL_MODEL,
        vision_base_url=LOCAL_BASE,
        vision_api_key="",
    )
    kwargs.update(over)
    return Settings(**kwargs)


def _patched(monkeypatch, handler, **desc_kw):
    params = dict(base_url=LOCAL_BASE, model=LOCAL_MODEL, api_key="")
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


# -- configuration -------------------------------------------------------

def test_local_config_defaults_preserve_hosted_behavior():
    # Library defaults via field introspection: hermetic against any real
    # user .env (bare Settings() would inherit it).
    fields = Settings.model_fields
    assert fields["vision_provider"].default == "openai-compatible"
    assert fields["vision_api_key"].default.get_secret_value() == ""
    # Default factory path unchanged: echo -> offline, honestly.
    assert isinstance(
        create_describer(Settings(ai_provider="echo",
                                  vision_provider="openai-compatible")),
        MetadataDescriber)


def test_safe_dump_redacts_vision_key():
    s = Settings(vision_api_key="secret-xyz")
    assert "secret-xyz" not in str(s.model_dump_safe())
    assert Settings(vision_api_key="").model_dump_safe()["vision_api_key"] == ""


def test_is_local_url():
    assert is_local_url("http://127.0.0.1:8081/v1")
    assert is_local_url("http://localhost:8081/v1")
    assert is_local_url("http://[::1]:8081/v1")
    assert not is_local_url("https://openrouter.ai/api/v1")
    assert not is_local_url("http://192.168.1.10:8081/v1")
    assert not is_local_url("http://example.com/v1")
    assert not is_local_url("")
    assert not is_local_url("not a url")


# -- endpoint selection ----------------------------------------------------

def test_local_factory_uses_loopback_endpoint_and_model():
    desc = create_describer(_local_settings())
    assert isinstance(desc, OpenRouterVisionDescriber)
    assert desc.base_url == LOCAL_BASE
    assert desc.model == LOCAL_MODEL
    assert "Authorization" not in desc._headers()


def test_local_mode_ignores_hosted_chat_key():
    """Even with AI_API_KEY set (hosted text), local vision sends nothing."""
    desc = create_describer(_local_settings(ai_api_key="hosted-secret"))
    assert "Authorization" not in desc._headers()
    assert "hosted-secret" not in str(desc._headers())


def test_local_missing_base_url_is_explicit():
    with pytest.raises(VisionUnavailableError, match="VISION_BASE_URL"):
        create_describer(_local_settings(vision_base_url=""))


def test_local_missing_model_is_explicit():
    with pytest.raises(VisionUnavailableError, match="VISION_MODEL"):
        create_describer(_local_settings(vision_model="  "))


def test_local_refuses_non_loopback_endpoint():
    with pytest.raises(VisionUnavailableError, match="[Ll]oopback|hosted"):
        create_describer(
            _local_settings(vision_base_url="https://openrouter.ai/api/v1")
        )
    with pytest.raises(VisionUnavailableError, match="[Ll]oopback|hosted"):
        create_describer(
            _local_settings(vision_base_url="http://192.168.1.10:8081/v1")
        )


def test_unknown_vision_provider_is_explicit():
    with pytest.raises(VisionUnavailableError, match="Unknown VISION_PROVIDER"):
        create_describer(_local_settings(vision_provider="nope"))


def test_hosted_mode_still_reuses_chat_key():
    s = Settings(
        ai_provider="openai-compatible",
        ai_api_key="k",
        ai_base_url="https://example.com/v1",
        vision_provider="openai-compatible",
        vision_model="v-model",
        vision_api_key="",
    )
    desc = create_describer(s)
    assert isinstance(desc, OpenRouterVisionDescriber)
    assert desc._headers()["Authorization"] == "Bearer k"


# -- responses ---------------------------------------------------------------

@pytest.mark.asyncio
async def test_local_successful_description(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.read().decode()))
        seen["auth"] = request.headers.get("authorization")
        seen["host"] = request.url.host
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "a red door"}}]}
        )

    desc = _patched(monkeypatch, handler)
    out = await desc.describe(b"\xff\xd8fake", "image/jpeg", "What is it?")
    assert out.text == "a red door"
    assert out.model == LOCAL_MODEL
    assert seen["host"] in ("127.0.0.1", "localhost", "::1")  # loopback only
    assert seen["auth"] is None
    content = seen["messages"][0]["content"]
    assert {p["type"] for p in content} == {"text", "image_url"}


@pytest.mark.asyncio
async def test_local_unavailable_server(monkeypatch):
    class Refused:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ConnectError("connection refused")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Refused)
    from app.core.exceptions import ProviderUnavailable

    desc = OpenRouterVisionDescriber(
        base_url=LOCAL_BASE, model=LOCAL_MODEL, api_key=""
    )
    with pytest.raises(ProviderUnavailable, match="[Uu]nreachable"):
        await desc.describe(b"img", "image/jpeg", "hi")


@pytest.mark.asyncio
async def test_local_timeout_and_malformed(monkeypatch):
    from app.core.exceptions import ProviderUnavailable

    class Slow:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ReadTimeout("too slow")

        async def __aexit__(self, *a):
            return False

    def bad_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    desc = _patched(monkeypatch, bad_handler)
    with pytest.raises(ProviderUnavailable, match="[Mm]alformed"):
        await desc.describe(b"img", "image/jpeg", "hi")

    monkeypatch.setattr(httpx, "AsyncClient", Slow)
    desc = OpenRouterVisionDescriber(
        base_url=LOCAL_BASE, model=LOCAL_MODEL, api_key=""
    )
    with pytest.raises(ProviderUnavailable, match="[Tt]imed out"):
        await desc.describe(b"img", "image/jpeg", "hi")


@pytest.mark.asyncio
async def test_local_unsupported_mime_rejected_without_http(monkeypatch):
    from app.core.exceptions import ProviderUnavailable

    calls = []

    class NoNet:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            calls.append(1)
            raise AssertionError("no HTTP expected")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", NoNet)
    desc = OpenRouterVisionDescriber(
        base_url=LOCAL_BASE, model=LOCAL_MODEL, api_key=""
    )
    with pytest.raises(ProviderUnavailable, match="[Uu]nsupported image"):
        await desc.describe(b"img", "image/gif", "hi")
    assert calls == []


# -- text provider independence -----------------------------------------------

def test_text_provider_uses_chat_settings_only():
    provider = create_provider(_local_settings())
    assert provider._model == "text-model"
    assert provider._base_url == "http://127.0.0.1:8080/v1"
