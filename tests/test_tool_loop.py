"""Tool-loop completion tests: multi-turn research must end in an answer.

Reproduces the live Phase 6 failure: search -> search -> fetch exhausted
the default 3-turn budget and the user got "couldn't compose a final
answer" despite all tool results being available. All providers scripted;
no network, no credentials.
"""

import pytest

from app.brain.assistant import NexusAssistant
from app.brain.provider import AIProvider, ChatMessage, ChatResponse
from app.core.config import Settings
from app.security.audit import AuditLogger
from app.security.confirmation import auto_deny
from app.tools import build_default_registry
from app.tools.executor import ToolExecutor
from app.tools.schemas import LlmToolCall


class ScriptedLoopProvider(AIProvider):
    """Scripted tool-calling turns plus a scripted final synthesis reply."""

    def __init__(self, turns, synthesis="FINAL SYNTHESIS"):
        self.turns = list(turns)
        self.synthesis = synthesis
        self.with_tools_calls = 0
        self.plain_calls = 0
        self.seen_messages = []
        self.tool_turn_messages = []

    @property
    def name(self) -> str:
        return "scripted-loop"

    async def send_message(self, messages: list[ChatMessage]) -> ChatResponse:
        self.plain_calls += 1
        self.seen_messages = list(messages)
        return ChatResponse(content=self.synthesis, model="s", provider="s")

    async def stream_message(self, messages):
        yield self.synthesis  # pragma: no cover

    async def send_message_with_tools(self, messages, tools):
        self.with_tools_calls += 1
        self.tool_turn_messages.append(list(messages))
        if not self.turns:  # script exhausted: model stops calling tools
            return ChatResponse(
                content=self.synthesis, model="s", provider="s"
            )
        turn = self.turns.pop(0)
        return ChatResponse(
            content=turn.get("content", ""), model="s", provider="s",
            tool_calls=turn.get("calls", []),
        )

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def wired(tmp_path):
    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
        max_tool_iterations=3,
    )
    registry = build_default_registry(settings)
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(settings.tools_audit_file),
        confirm=auto_deny,
    )
    return settings, registry, executor


def _assistant(settings, provider, registry, executor):
    return NexusAssistant(
        settings=settings, provider=provider,
        registry=registry, executor=executor,
    )


def _call(name, cid, args=None):
    return LlmToolCall(id=cid, name=name, arguments=args or {})


@pytest.mark.asyncio
async def test_research_loop_ends_in_synthesized_answer(wired):
    """search, search, fetch (budget exhausted) -> final answer, not fallback."""
    settings, registry, executor = wired
    provider = ScriptedLoopProvider(
        turns=[
            {"calls": [_call("system.info", "c1")]},
            {"calls": [_call("system.cpu", "c2")]},
            {"calls": [_call("system.memory", "c3")]},
        ],
        synthesis="Summary: research complete with 3 tool results.",
    )
    assistant = _assistant(settings, provider, registry, executor)
    reply = await assistant.achat("research this topic thoroughly")
    assert "couldn't compose" not in reply
    assert "Summary: research complete" in reply
    assert provider.with_tools_calls == 3
    assert provider.plain_calls == 1  # one final synthesis, no more tools


@pytest.mark.asyncio
async def test_multiple_calls_in_one_turn_all_execute(wired):
    """One turn may carry several calls; every result reaches the model."""
    settings, registry, executor = wired
    provider = ScriptedLoopProvider(
        turns=[
            {
                "calls": [
                    _call("system.info", "c1"),
                    _call("system.cpu", "c2"),
                    _call("system.platform", "c3"),
                ]
            },
        ],
        synthesis="done",
    )
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("tell me everything") == "done"
    assert provider.with_tools_calls == 2
    tool_msgs = [m for m in provider.tool_turn_messages[-1] if m.role == "tool"]
    assert {m.tool_call_id for m in tool_msgs} == {"c1", "c2", "c3"}


@pytest.mark.asyncio
async def test_endless_tool_requests_terminate(wired):
    """A model that always requests tools cannot loop forever."""
    settings, registry, executor = wired

    class AlwaysCalls(ScriptedLoopProvider):
        async def send_message_with_tools(self, messages, tools):
            self.with_tools_calls += 1
            n = self.with_tools_calls
            return ChatResponse(
                content="", model="s", provider="s",
                tool_calls=[_call("system.info", f"c{n}")],
            )

    provider = AlwaysCalls(turns=[], synthesis="gave up synthesizing")
    assistant = _assistant(settings, provider, registry, executor)
    reply = await assistant.achat("loop forever please")
    assert provider.with_tools_calls == 3  # bounded by max_tool_iterations
    assert provider.plain_calls == 1
    assert reply == "gave up synthesizing"


@pytest.mark.asyncio
async def test_synthesis_failure_is_honest(wired):
    """Budget exhausted + synthesis empty -> honest fallback, never blank."""
    settings, registry, executor = wired
    provider = ScriptedLoopProvider(
        turns=[{"calls": [_call("system.info", "c1")]}] * 3,
        synthesis="   ",
    )
    assistant = _assistant(settings, provider, registry, executor)
    reply = await assistant.achat("research this")
    assert reply.strip()
    assert "couldn't compose" in reply
    assert "3 tool call" in reply
