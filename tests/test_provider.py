"""AI provider abstraction tests (Phase 1). All offline/mocked."""

import pytest

from app.brain.provider import (
    ChatMessage,
    EchoProvider,
    OpenAICompatibleProvider,
    create_provider,
)
from app.core.config import Settings
from app.core.exceptions import ConfigError, ProviderAuthError


@pytest.mark.asyncio
async def test_echo_send_message(echo_provider):
    out = await echo_provider.send_message(
        [ChatMessage(role="user", content="hello")]
    )
    assert "hello" in out.content
    assert out.provider == "echo"


@pytest.mark.asyncio
async def test_echo_stream_message(echo_provider):
    chunks = [
        c
        async for c in echo_provider.stream_message(
            [ChatMessage(role="user", content="hi")]
        )
    ]
    assert len(chunks) == 1
    assert "hi" in chunks[0]


@pytest.mark.asyncio
async def test_echo_health_check(echo_provider):
    assert await echo_provider.health_check() is True


@pytest.mark.asyncio
async def test_echo_tool_call_stub_returns_none(echo_provider):
    assert (
        await echo_provider.generate_tool_call(
            [ChatMessage(role="user", content="open notepad")]
        )
        is None
    )


def test_factory_creates_echo():
    p = create_provider(Settings(ai_provider="echo"))
    assert isinstance(p, EchoProvider)


def test_factory_rejects_unknown():
    with pytest.raises(ConfigError):
        create_provider(Settings(ai_provider="nope"))


def test_openai_provider_requires_key():
    with pytest.raises(ProviderAuthError):
        OpenAICompatibleProvider(base_url="https://example.com/v1", model="m", api_key="")


def test_openai_provider_requires_url():
    with pytest.raises(ProviderAuthError):
        OpenAICompatibleProvider(base_url="", model="m", api_key="key")
