"""NVIDIA-only architecture tests (all mocked; no network, no credentials).

Covers what the feature-specific suites do not state explicitly:
NVIDIA is the single AI path (failures raise, never fall back), no local
model servers or handoffs exist, loopback/legacy leftovers are ignored,
timeouts come from settings, health reports reachability (never mere key
presence), voice STT is explicitly unavailable, web retrieval feeds NVIDIA
synthesis, and secrets stay redacted.
"""

import httpx
import pytest

from app.brain.provider import NvidiaNemotronProvider, create_provider
from app.core.config import Settings
from app.core.exceptions import (
    ConfigError,
    ProviderAuthError,
    ProviderUnavailable,
)


def _nvidia_settings(**over):
    base = dict(
        ai_provider="nvidia",
        ai_model="nexus-default",
        ai_api_key="",
        ai_base_url="",
        ai_timeout_seconds=30.0,
        nvidia_api_key="test-nvapi-key",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_vision_model="meta/llama-3.2-11b-vision-instruct",
    )
    base.update(over)
    return Settings(**base)


def _patched_post(monkeypatch, handler):
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


# -- single provider: NVIDIA configured, failures never fall back ------------

def test_nvidia_is_the_configured_ai_provider():
    provider = create_provider(_nvidia_settings())
    assert type(provider) is NvidiaNemotronProvider
    assert provider.name == "nvidia"


def test_unknown_provider_rejected():
    with pytest.raises(ConfigError):
        create_provider(Settings(ai_provider="watson-x"))


@pytest.mark.asyncio
async def test_nvidia_failure_raises_never_falls_back(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patched_post(monkeypatch, handler)
    provider = create_provider(_nvidia_settings())
    from app.brain.provider import ChatMessage

    with pytest.raises(ProviderUnavailable):
        await provider.send_message([ChatMessage(role="user", content="hi")])


def test_nvidia_requires_key_no_silent_anonymous():
    with pytest.raises(ProviderAuthError, match="NVIDIA_API_KEY"):
        create_provider(_nvidia_settings(nvidia_api_key="", ai_api_key=""))


# -- no local servers, hooks, or loopback inference ----------------------------

def test_no_handoff_plumbing_on_assistant_or_factory():
    import inspect

    from app.brain.assistant import NexusAssistant
    from app.main import build_assistant

    assert "synthesis_handoff" not in inspect.signature(
        NexusAssistant.__init__).parameters
    assert "synthesis_handoff" not in inspect.signature(
        build_assistant).parameters


def test_resolvers_ignore_loopback_and_gguf_leftovers():
    from app.brain.provider import (
        NVIDIA_DEFAULT_BASE_URL,
        NVIDIA_DEFAULT_MODEL,
        resolve_nvidia_settings,
        resolve_nvidia_vision_settings,
    )

    base, model, _ = resolve_nvidia_settings(_nvidia_settings(
        ai_base_url="http://127.0.0.1:8080/v1",
        ai_model="C:\\cache\\Qwen3-8B-Q4_K_M.gguf",
    ))
    assert base == NVIDIA_DEFAULT_BASE_URL
    assert model == NVIDIA_DEFAULT_MODEL
    v_base, v_model, _ = resolve_nvidia_vision_settings(_nvidia_settings(
        vision_base_url="http://127.0.0.1:8081/v1",
        vision_model="qwen2.5-vl-3b-local",
    ))
    assert v_base == NVIDIA_DEFAULT_BASE_URL
    assert v_model == "meta/llama-3.2-11b-vision-instruct"


def test_vision_local_is_refused():
    from app.vision.errors import VisionUnavailableError
    from app.vision.factories import create_describer

    with pytest.raises(VisionUnavailableError, match="[Nn][Vv][Ii][Dd][Ii][Aa]"):
        create_describer(_nvidia_settings(
            ai_provider="echo",
            vision_provider="local",
            vision_base_url="http://127.0.0.1:8081/v1",
            vision_model="qwen2.5-vl-3b-local",
        ))


# -- timeouts and health accuracy ----------------------------------------------

def test_request_timeout_comes_from_settings():
    provider = create_provider(_nvidia_settings(ai_timeout_seconds=180.0))
    assert provider._timeout == 180.0


@pytest.mark.asyncio
async def test_health_probe_timeout_names_probe_budget(monkeypatch):
    class Slow:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            raise httpx.ReadTimeout("slow")

        async def __aexit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "AsyncClient", Slow)
    detail = await create_provider(_nvidia_settings()).health_detail()
    assert detail["ok"] is False
    assert detail["status"] == "timeout"
    assert "probe" in detail["detail"]


@pytest.mark.asyncio
async def test_key_presence_alone_is_not_healthy(monkeypatch):
    """A configured key with a 401 endpoint must not report healthy."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="invalid key")

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

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    detail = await create_provider(_nvidia_settings()).health_detail()
    assert detail == {
        "ok": False,
        "status": "auth_error",
        "detail": detail["detail"],
    }
    assert "401" in detail["detail"]


def test_dotenv_duplicate_resolves_last_wins(tmp_path):
    """Within-file duplicates resolve last-wins (python-dotenv dict order)."""
    from dotenv import dotenv_values

    env = tmp_path / ".env"
    env.write_text(
        "AI_TIMEOUT_SECONDS=30\nAI_TIMEOUT_SECONDS=180\n", encoding="utf-8")
    assert dotenv_values(str(env))["AI_TIMEOUT_SECONDS"] == "180"


def test_os_environment_overrides_dotenv(monkeypatch, tmp_path):
    from app.core.config import reload_settings

    env = tmp_path / ".env"
    env.write_text("AI_MODEL=from-file\n", encoding="utf-8")
    monkeypatch.setenv("AI_MODEL", "from-os")
    s = Settings(_env_file=str(env), ai_provider="echo")
    assert s.ai_model == "from-os"
    reload_settings()


# -- voice: NVIDIA cloud STT live, local-AI names refused --------------------

def test_voice_stt_unavailable_for_all_live_backends():
    from app.voice.errors import VoiceUnavailableError
    from app.voice.factories import create_stt

    for backend in ("whisper", "vosk", "watson-x"):
        with pytest.raises(VoiceUnavailableError):
            create_stt(_nvidia_settings(voice_stt_backend=backend))


def test_voice_nvidia_stt_builds_with_key():
    from app.voice.factories import create_stt
    from app.voice.stt import NvidiaSTT

    assert isinstance(
        create_stt(_nvidia_settings(voice_stt_backend="nvidia")), NvidiaSTT
    )


def test_voice_script_and_null_tts_still_work():
    from app.voice.factories import create_stt, create_tts
    from app.voice.stt import ScriptedSTT
    from app.voice.tts import NullTTS

    assert isinstance(create_stt(_nvidia_settings(), script=["hi"]), ScriptedSTT)
    assert isinstance(create_tts(_nvidia_settings(voice_tts_backend="null")), NullTTS)


def test_voice_unavailable_never_breaks_text_mode(monkeypatch, tmp_path, capsys):
    import run
    from app.core.config import reload_settings

    monkeypatch.setenv("AI_PROVIDER", "echo")
    monkeypatch.setenv("AI_MODEL", "test-model")
    # Isolate from any real .env key so live voice refuses via
    # missing-credentials (exit 2) without touching mic/network.
    monkeypatch.setenv("NVIDIA_API_KEY", "")
    monkeypatch.setenv("AI_API_KEY", "")
    monkeypatch.setenv("VOICE_STT_BACKEND", "nvidia")
    reload_settings()
    try:
        # Live voice refuses cleanly (exit 2, no traceback)...
        assert run.main(["--voice"]) == 2
        # ...while text mode is unaffected.
        monkeypatch.setattr("builtins.input", lambda _prompt="": "/exit")
        assert run.main(["--text"]) == 0
    finally:
        reload_settings()
    assert "text mode" in capsys.readouterr().out


# -- web retrieval feeds NVIDIA synthesis ---------------------------------------

@pytest.mark.asyncio
async def test_web_research_synthesized_from_retrieved_source(tmp_path, monkeypatch):
    """Retrieval (mocked HTTP, no AI) + provider synthesis (mocked NVIDIA)."""
    from app.brain.assistant import NexusAssistant
    from app.brain.provider import ChatMessage, ChatResponse
    from app.security.audit import AuditLogger
    from app.tools import build_default_registry
    from app.tools.executor import ToolExecutor
    from app.tools.schemas import LlmToolCall
    from app.web.client import WebContext

    settings = _nvidia_settings(
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
        web_search_url="https://api.duckduckgo.com/",
    )
    registry = build_default_registry(settings)
    search = registry.get("web.search")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "Heading": "Example",
                "AbstractText": "Example domain for docs.",
                "AbstractURL": "https://example.com/about",
                "AbstractSource": "Example",
                "RelatedTopics": [],
            },
        )
    )
    search.ctx = WebContext(
        search_url="https://api.duckduckgo.com/",
        client_factory=lambda timeout: httpx.Client(
            transport=transport, timeout=timeout),
        dns_resolver=lambda host: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(tmp_path / "audit.jsonl"))

    calls = {"n": 0}

    class FakeNvidia:
        name = "nvidia"

        async def health_check(self):
            return True

        async def send_message(self, messages):
            return ChatResponse(content="unused", model="m", provider="nvidia")

        async def send_message_with_tools(self, messages, tools):
            calls["n"] += 1
            if calls["n"] == 1:
                return ChatResponse(
                    content="", model="m", provider="nvidia",
                    tool_calls=[LlmToolCall(
                        id="c1", name="web.search",
                        arguments={"query": "example domain"})],
                )
            tool_blob = " ".join(m.content for m in messages)
            assert "example.com/about" in tool_blob  # evidence reached synthesis
            return ChatResponse(
                content="Per https://example.com/about, it is a docs domain.",
                model="m", provider="nvidia")

    assistant = NexusAssistant(
        settings=settings, provider=FakeNvidia(),
        registry=registry, executor=executor)
    reply = await assistant.achat("What is example.com for?")
    assert "https://example.com/about" in reply


# -- secrets stay redacted -------------------------------------------------------

def test_nvidia_and_vision_keys_redacted_in_safe_dump():
    s = _nvidia_settings(ai_api_key="test-ai-key", vision_api_key="test-vision-key")
    dumped = s.model_dump_safe()
    assert dumped["nvidia_api_key"] == "***"
    assert dumped["ai_api_key"] == "***"
    assert dumped["vision_api_key"] == "***"
    blob = str(dumped)
    assert "test-nvapi-key" not in blob
    assert "test-ai-key" not in blob
    assert "test-vision-key" not in blob


def test_audit_sanitizes_nvidia_key_material():
    from app.security.audit import sanitize

    cleaned = sanitize({
        "authorization": "Bearer nvapi-abc123xyz456",
        "token": "nvapi-abc123xyz456",
        "note": "key nvapi-abc123xyz456 inside prose",
    })
    assert cleaned["authorization"] == "***"
    assert cleaned["token"] == "***"
    assert "nvapi-abc123xyz456" not in str(cleaned)
