"""Tool-call execution path regression tests (Phase 6 audit).

Verifies the full lifecycle with model-visible (external) names: parse a
single call, parse multiple calls, reject malformed/unknown safely,
preserve call IDs, associate each result with its call, return actual
executor output to the model, ground the final answer in it, and enforce
the iteration budget. All providers scripted or mocked; offline.
"""

import json

import httpx
import pytest

from app.brain.assistant import NexusAssistant
from app.brain.provider import AIProvider, ChatMessage, ChatResponse
from app.core.config import Settings
from app.security.audit import AuditLogger
from app.security.confirmation import auto_deny
from app.tools import build_default_registry
from app.tools.executor import ToolExecutor
from app.tools.schemas import LlmToolCall, parse_openai_tool_calls


class Scripted(AIProvider):
    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.tool_turns = []

    @property
    def name(self):
        return "scripted-exec"

    async def send_message(self, messages):
        self.seen = list(messages)
        return ChatResponse(content="synthesis from evidence", model="s", provider="s")

    async def stream_message(self, messages):
        yield "synthesis from evidence"

    async def send_message_with_tools(self, messages, tools):
        self.tool_turns.append(list(messages))
        turn = self.script.pop(0)
        if isinstance(turn, list):
            return ChatResponse(content="", model="s", provider="s", tool_calls=turn)
        return ChatResponse(content=turn, model="s", provider="s")

    async def health_check(self):
        return True


def _wired(tmp_path, **kw):
    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
        max_tool_iterations=kw.get("max_iters", 3),
    )
    registry = build_default_registry(settings)
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(settings.tools_audit_file),
        confirm=auto_deny,
    )
    return settings, registry, executor


@pytest.mark.asyncio
async def test_single_external_call_executes(tmp_path):
    settings, registry, executor = _wired(tmp_path)
    assistant = NexusAssistant(
        settings=settings,
        provider=Scripted([[LlmToolCall(id="c1", name="system_info", arguments={})],
                           "done"]),
        registry=registry, executor=executor,
    )
    assert await assistant.achat("inspect the system") == "done"


@pytest.mark.asyncio
async def test_multiple_calls_keep_ids_and_results(tmp_path):
    settings, registry, executor = _wired(tmp_path)
    provider = Scripted([[
        LlmToolCall(id="c1", name="system_info", arguments={}),
        LlmToolCall(id="c2", name="system_cpu", arguments={}),
    ], "both done"])
    assistant = NexusAssistant(
        settings=settings, provider=provider, registry=registry, executor=executor
    )
    assert await assistant.achat("everything") == "both done"
    # Both tool results were computed and are visible on the next model turn.
    assert len(provider.tool_turns) == 2
    tool_msgs = [m for m in provider.tool_turns[1] if m.role == "tool"]
    assert {m.tool_call_id for m in tool_msgs} == {"c1", "c2"}
    for msg in tool_msgs:
        payload = json.loads(msg.content)
        assert payload["ok"] is True


@pytest.mark.asyncio
async def test_unknown_tool_result_is_honest_failure(tmp_path):
    settings, registry, executor = _wired(tmp_path)
    provider = Scripted([[LlmToolCall(id="c9", name="not_a_tool", arguments={})],
                         "I could not run that."])
    assistant = NexusAssistant(
        settings=settings, provider=provider, registry=registry, executor=executor
    )
    assert await assistant.achat("do the thing") == "I could not run that."
    tool_msgs = [m for m in provider.tool_turns[1] if m.role == "tool"]
    assert len(tool_msgs) == 1
    payload = json.loads(tool_msgs[0].content)
    assert payload["ok"] is False
    assert "Unknown tool" in payload["error"]


@pytest.mark.asyncio
async def test_final_answer_uses_actual_tool_output(tmp_path):
    """Executor output (not model invention) reaches the synthesis turn."""
    # One budgeted tool turn, then the budget is exhausted and the model
    # gets a final tool-free synthesis turn carrying the evidence.
    settings, registry, executor = _wired(tmp_path, max_iters=1)
    provider = Scripted(
        [[LlmToolCall(id="c1", name="system_info", arguments={})]]
    )
    assistant = NexusAssistant(
        settings=settings, provider=provider, registry=registry, executor=executor
    )
    reply = await assistant.achat("what OS?")
    assert reply == "synthesis from evidence"
    tool_msgs = [m for m in provider.seen if m.role == "tool"]
    assert tool_msgs, "synthesis turn must include the tool result"
    assert "os" in json.loads(tool_msgs[0].content)["output"]


def test_parse_single_and_multiple_and_malformed():
    single = parse_openai_tool_calls({
        "tool_calls": [{"id": "a", "type": "function",
                        "function": {"name": "system_info", "arguments": "{}"}}]
    })
    assert len(single) == 1 and single[0].id == "a"
    multi = parse_openai_tool_calls({
        "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "system_info", "arguments": "{}"}},
            {"id": "b", "type": "function",
             "function": {"name": "system_cpu", "arguments": '{"x": 1}'}},
        ]
    })
    assert [c.id for c in multi] == ["a", "b"]
    assert multi[1].arguments == {"x": 1}
    # Malformed entries skipped, never crash.
    bad = parse_openai_tool_calls({
        "tool_calls": [
            {"junk": True},
            {"id": "x", "function": {"name": "", "arguments": "{}"}},
            {"id": "y", "function": {"name": "ok", "arguments": "{not json"}},
            "not-a-dict",
        ]
    })
    assert bad == []
    # Arguments given as a dict (not a JSON string) are accepted.
    as_dict = parse_openai_tool_calls({
        "tool_calls": [{"id": "d", "type": "function",
                        "function": {"name": "web_search", "arguments": {"query": "x"}}}]
    })
    assert as_dict[0].arguments == {"query": "x"}


@pytest.mark.asyncio
async def test_iteration_budget_enforced_with_external_names(tmp_path):
    settings, registry, executor = _wired(tmp_path, max_iters=2)

    class AlwaysTools(Scripted):
        def __init__(self):
            super().__init__([])
            self.n = 0

        async def send_message_with_tools(self, messages, tools):
            self.n += 1
            return ChatResponse(
                content="", model="s", provider="s",
                tool_calls=[LlmToolCall(id=f"c{self.n}", name="system_info",
                                        arguments={})],
            )

    provider = AlwaysTools()
    assistant = NexusAssistant(
        settings=settings, provider=provider, registry=registry, executor=executor
    )
    reply = await assistant.achat("loop")
    assert provider.n == 2
    assert reply == "synthesis from evidence"
