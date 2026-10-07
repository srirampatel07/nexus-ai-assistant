"""Assistant + tool loop integration tests (Phase 2). AI is scripted, never real."""

from collections.abc import AsyncIterator

import pytest

from app.brain.assistant import NexusAssistant
from app.brain.provider import AIProvider, ChatMessage, ChatResponse
from app.core.config import Settings
from app.security.audit import AuditLogger
from app.security.confirmation import auto_approve, auto_deny
from app.tools import build_default_registry
from app.tools.executor import ToolExecutor
from app.tools.schemas import LlmToolCall


class ScriptedProvider(AIProvider):
    """Replays queued turns: ChatResponse or list[LlmToolCall]."""

    def __init__(self, script: list):
        self.script = list(script)
        self.seen_tools: list | None = None

    @property
    def name(self) -> str:
        return "scripted"

    async def send_message(self, messages: list[ChatMessage]) -> ChatResponse:
        return ChatResponse(content="plain", model="s", provider="scripted")

    async def stream_message(self, messages) -> AsyncIterator[str]:
        yield "plain"

    async def send_message_with_tools(
        self, messages: list[ChatMessage], tools: list
    ) -> ChatResponse:
        self.seen_tools = tools
        turn = self.script.pop(0)
        if isinstance(turn, list):
            return ChatResponse(content="", model="s", provider="s", tool_calls=turn)
        return ChatResponse(content=turn, model="s", provider="s")

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
        registry=registry, audit=AuditLogger(settings.tools_audit_file), confirm=auto_deny
    )
    return settings, registry, executor


def _assistant(settings, script, executor=None, registry=None):
    return NexusAssistant(
        settings=settings,
        provider=ScriptedProvider(script),
        registry=registry,
        executor=executor,
    )


@pytest.mark.asyncio
async def test_tool_loop_executes_and_answers(wired):
    settings, registry, executor = wired
    assistant = _assistant(
        settings,
        [[LlmToolCall(id="c1", name="system.info", arguments={})], "System is Windows."],
        executor,
        registry,
    )
    reply = await assistant.achat("tell me about this system")
    assert reply == "System is Windows."
    assert assistant.provider.seen_tools  # definitions were offered to the model
    history = assistant.get_history()
    assert history[0].role == "user"
    assert history[-1].role == "assistant"


@pytest.mark.asyncio
async def test_denied_tool_result_returned_to_model(wired, tmp_path):
    settings, registry, executor = wired
    target = tmp_path / "blocked.txt"
    assistant = _assistant(
        settings,
        [
            [
                LlmToolCall(
                    id="c1",
                    name="filesystem.write",
                    arguments={"path": str(target), "content": "x"},
                )
            ],
            "I need your approval to write that file.",
        ],
        executor,
        registry,
    )
    reply = await assistant.achat("write a file", confirm=auto_deny)
    assert "approval" in reply
    assert not target.exists()


@pytest.mark.asyncio
async def test_approved_write_succeeds_in_loop(wired, tmp_path):
    settings, registry, executor = wired
    target = tmp_path / "ok.txt"
    assistant = _assistant(
        settings,
        [
            [
                LlmToolCall(
                    id="c1",
                    name="filesystem.write",
                    arguments={"path": str(target), "content": "hi"},
                )
            ],
            "Done. I've written the file.",
        ],
        executor,
        registry,
    )
    reply = await assistant.achat("write a file", confirm=auto_approve)
    assert "written" in reply
    assert target.read_text() == "hi"


@pytest.mark.asyncio
async def test_max_iterations_stops_loop(wired):
    settings, registry, executor = wired
    settings.max_tool_iterations = 2
    assistant = _assistant(
        settings,
        [
            [LlmToolCall(id="c1", name="system.info", arguments={})],
            [LlmToolCall(id="c2", name="system.cpu", arguments={})],
        ],
        executor,
        registry,
    )
    reply = await assistant.achat("loop forever please")
    assert "couldn't compose" in reply


@pytest.mark.asyncio
async def test_no_registry_keeps_legacy_behavior(wired):
    settings, _, _ = wired
    assistant = NexusAssistant(settings=settings, provider=ScriptedProvider(["plain ok"]))
    # Legacy path uses send_message (script returns "plain" for any shape).
    reply = await assistant.achat("some novel question")
    assert reply == "plain"
    assert not assistant.tools_enabled


def test_tools_enabled_flag(wired):
    settings, registry, executor = wired
    full = NexusAssistant(
        settings=settings,
        provider=ScriptedProvider([]),
        registry=registry,
        executor=executor,
    )
    bare = NexusAssistant(settings=settings, provider=ScriptedProvider([]))
    assert full.tools_enabled
    assert not bare.tools_enabled
