"""Assistant core tests (Phase 1). Offline via EchoProvider."""

import pytest

from app.brain.assistant import NexusAssistant
from app.brain.planner import PlanAction, Planner
from app.brain.provider import ChatMessage, EchoProvider
from app.brain.router import ToolRouter
from app.core.config import Settings
from app.core.exceptions import ProviderUnavailable, ToolNotAvailable


def make_assistant(**kwargs) -> NexusAssistant:
    settings = Settings(ai_provider="echo", ai_model="test", **kwargs)
    return NexusAssistant(settings=settings, provider=EchoProvider())


@pytest.mark.asyncio
async def test_greeting_is_natural():
    a = make_assistant()
    assert await a.achat("hello") == "Hello. How can I help?"


@pytest.mark.asyncio
async def test_time_intent():
    a = make_assistant()
    reply = await a.achat("what time is it?")
    assert "currently" in reply.lower()


@pytest.mark.asyncio
async def test_empty_input_prompts_user():
    a = make_assistant()
    reply = await a.achat("   ")
    assert "listening" in reply.lower()


@pytest.mark.asyncio
async def test_fallback_goes_to_provider():
    a = make_assistant()
    reply = await a.achat("tell me something interesting")
    assert reply.startswith("You said:")


@pytest.mark.asyncio
async def test_history_is_bounded():
    a = make_assistant(max_history_messages=4)
    for i in range(6):
        await a.achat(f"message {i}")
    assert len(a.get_history()) <= 4


@pytest.mark.asyncio
async def test_clear_history():
    a = make_assistant()
    await a.achat("hello")
    a.clear_history()
    assert a.get_history() == []


@pytest.mark.asyncio
async def test_provider_failure_is_honest():
    class Broken(EchoProvider):
        async def send_message(self, messages):
            raise ProviderUnavailable("down")

    settings = Settings(ai_provider="echo")
    a = NexusAssistant(settings=settings, provider=Broken())
    reply = await a.achat("some novel question xyz")
    assert "trouble" in reply.lower()
    assert "success" not in reply.lower()


def test_sync_chat_wrapper():
    a = make_assistant()
    assert "Hello" in a.chat("hello")


def test_planner_phase1_always_responds_directly():
    plan = Planner().plan("open notepad")
    assert plan.action == PlanAction.RESPOND_DIRECTLY


def test_router_phase1_has_no_tools():
    router = ToolRouter()
    assert router.list_tools() == []
    from app.brain.planner import Plan

    with pytest.raises(ToolNotAvailable):
        router.route(Plan(action=PlanAction.USE_TOOL, reasoning="t", tool_name="x"))
