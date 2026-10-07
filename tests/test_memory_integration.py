"""Assistant memory integration + CLI command tests (Phase 3). Offline only."""

import pytest

from app.brain.assistant import NexusAssistant
from app.brain.provider import EchoProvider
from app.core.config import Settings
from app.memory.database import create_memory_engine
from app.memory.manager import MemoryManager
from run import cmd_forget, cmd_memory_list, cmd_memory_search


@pytest.fixture
def settings(tmp_path):
    return Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
        memory_database_path=str(tmp_path / "nexus.db"),
    )


@pytest.fixture
def assistant(settings):
    engine = create_memory_engine(settings.memory_database_path)
    memory = MemoryManager(engine)
    return NexusAssistant(
        settings=settings,
        provider=EchoProvider(),
        conversation_id="test-conv",
        memory=memory,
    )


def test_conversation_id_generated_by_default(settings):
    a = NexusAssistant(settings=settings, provider=EchoProvider())
    b = NexusAssistant(settings=settings, provider=EchoProvider())
    assert a.conversation_id and b.conversation_id
    assert a.conversation_id != b.conversation_id


def test_conversation_id_explicit_preserved(settings):
    a = NexusAssistant(
        settings=settings, provider=EchoProvider(), conversation_id="keep-me"
    )
    assert a.conversation_id == "keep-me"


@pytest.mark.asyncio
async def test_turns_persisted_as_conversation_memories(assistant):
    await assistant.achat("hello")
    memories = assistant.memory.get_recent("test-conv", limit=10)
    texts = [m.content for m in memories]
    assert "hello" in texts
    assert "Hello. How can I help?" in texts
    assert all(m.category == "conversation" for m in memories)


@pytest.mark.asyncio
async def test_memory_context_reaches_model(assistant):
    from app.brain.provider import ChatMessage

    seen: list[list[ChatMessage]] = []
    original = assistant.provider.send_message

    async def spy(messages):
        seen.append(messages)
        return await original(messages)

    assistant.provider.send_message = spy  # type: ignore[method-assign]
    assistant.memory.remember(
        "my favorite color is blue", category="preference", importance=0.9
    )
    await assistant.achat("what is my favorite color?")
    assert seen, "provider was never called"
    system_texts = [m.content for m in seen[0] if m.role == "system"]
    assert any("blue" in t for t in system_texts)


@pytest.mark.asyncio
async def test_memory_failure_never_breaks_chat(settings):
    class BrokenMemory:
        def get_context(self, *a, **k):
            raise RuntimeError("db down")

        def remember(self, *a, **k):
            raise RuntimeError("db down")

        def format_context(self, records):
            return ""

    a = NexusAssistant(
        settings=settings,
        provider=EchoProvider(),
        conversation_id="c",
        memory=BrokenMemory(),  # type: ignore[arg-type]
    )
    reply = await a.achat("tell me something interesting")
    assert reply.startswith("You said:")


@pytest.mark.asyncio
async def test_explicit_remember_stores_fact(assistant):
    reply = await assistant.achat("remember that my favorite color is blue")
    assert reply == "Noted. I'll remember that."
    hits = assistant.memory.search("blue", category="preference")
    assert len(hits) == 1


@pytest.mark.asyncio
async def test_explicit_remember_rejects_secret_without_echo(assistant):
    secret = "gsk_abcdefgh12345678"
    reply = await assistant.achat(f"remember that my key is {secret}")
    assert "secret" in reply.lower()
    assert secret not in reply
    stored = assistant.memory.search("", limit=50)
    assert all(secret not in r.content for r in stored)


@pytest.mark.asyncio
async def test_no_memory_keeps_legacy_behavior(settings):
    a = NexusAssistant(settings=settings, provider=EchoProvider())
    assert await a.achat("hello") == "Hello. How can I help?"
    assert await a.achat("remember that x is y") == "You said: remember that x is y"


@pytest.mark.asyncio
async def test_cross_conversation_recall(settings, tmp_path):
    db = str(tmp_path / "shared.db")

    def make_assistant(conv: str) -> NexusAssistant:
        return NexusAssistant(
            settings=settings,
            provider=EchoProvider(),
            conversation_id=conv,
            memory=MemoryManager(create_memory_engine(db)),
        )

    first = make_assistant("conv-one")
    assert await first.achat("remember that my favorite color is blue") == (
        "Noted. I'll remember that."
    )
    second = make_assistant("conv-two")  # simulates a restart
    records = second.memory.get_context("conv-two", query="favorite color")
    assert any("blue" in r.content for r in records)


def test_cli_memory_list(assistant):
    assert "disabled" not in cmd_memory_list(assistant)
    assert "No memories" in cmd_memory_list(assistant)
    assistant.memory.remember("a fact", category="fact", conversation_id="test-conv")
    out = cmd_memory_list(assistant)
    assert "[fact]" in out and "a fact" in out and "(test-conv)" in out


def test_cli_memory_list_disabled(settings):
    a = NexusAssistant(settings=settings, provider=EchoProvider())
    assert "disabled" in cmd_memory_list(a).lower()


def test_cli_memory_search(assistant):
    assistant.memory.remember("blue note", category="fact", conversation_id="test-conv")
    assert "blue note" in cmd_memory_search(assistant, "blue")
    assert "No memories" in cmd_memory_search(assistant, "zzz-no-match")
    assert "Usage" in cmd_memory_search(assistant, "  ")


def test_cli_forget(assistant):
    rec = assistant.memory.remember("gone soon", conversation_id="test-conv")
    assert cmd_forget(assistant, str(rec.id)) == f"Forgot memory #{rec.id}."
    assert "No memory" in cmd_forget(assistant, str(rec.id))
    assert "Usage" in cmd_forget(assistant, "not-a-number")


def test_cli_forget_works_across_conversations(assistant):
    rec = assistant.memory.remember("elsewhere", conversation_id="other-conv")
    assert cmd_forget(assistant, str(rec.id)) == f"Forgot memory #{rec.id}."
    assert assistant.memory.get(rec.id) is None
