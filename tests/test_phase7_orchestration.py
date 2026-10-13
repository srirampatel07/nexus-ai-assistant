"""Phase 7 orchestration tests: smarter multi-step tool use, all mocked.

Covers single/multiple calls, result feedback, multi-step completion,
invalid args, unknown tools, tool exceptions, provider errors, iteration
and time budgets, duplicate-execution prevention, approvals, secret
redaction, truncation, empty responses, task tracking, and no-local-startup
on the NVIDIA cloud path. No network, no credentials, no real servers.
"""

import asyncio
import json

import httpx
import pytest

from app.brain.assistant import PROVIDER_ERROR_REPLY, NexusAssistant
from app.brain.provider import AIProvider, ChatMessage, ChatResponse
from app.brain.tasks import TaskTracker
from app.core.config import Settings
from app.core.events import EventType, event_bus
from app.core.exceptions import ProviderUnavailable
from app.security.audit import AuditLogger
from app.security.confirmation import auto_approve, auto_deny
from app.tools import build_default_registry
from app.tools.executor import ToolExecutor
from app.tools.schemas import LlmToolCall


class ScriptedProvider(AIProvider):
    """Queued with_tools turns; records messages; scripted plain replies."""

    def __init__(self, turns, synthesis="SYNTHESIZED"):
        self.turns = list(turns)
        self.synthesis = synthesis
        self.with_tools_calls = 0
        self.plain_calls = 0
        self.tool_turns = []
        self.plain_turns = []

    @property
    def name(self) -> str:
        return "scripted-phase7"

    async def send_message(self, messages):
        self.plain_calls += 1
        self.plain_turns.append(list(messages))
        return ChatResponse(content=self.synthesis, model="s", provider="s")

    async def stream_message(self, messages):
        yield self.synthesis

    async def send_message_with_tools(self, messages, tools):
        self.with_tools_calls += 1
        self.tool_turns.append(list(messages))
        if not self.turns:
            return ChatResponse(content=self.synthesis, model="s", provider="s")
        turn = self.turns.pop(0)
        if isinstance(turn, list):
            return ChatResponse(content="", model="s", provider="s", tool_calls=turn)
        return ChatResponse(content=turn, model="s", provider="s")

    async def health_check(self) -> bool:
        return True


def _call(name, cid, args=None):
    return LlmToolCall(id=cid, name=name, arguments=args or {})


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


def _assistant(settings, provider, registry, executor):
    return NexusAssistant(
        settings=settings, provider=provider, registry=registry, executor=executor
    )


def _audit_lines(tmp_path):
    return (tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()


# -- single + multiple calls, feedback shape ---------------------------------

@pytest.mark.asyncio
async def test_single_tool_result_fed_back_with_status(wired):
    settings, registry, executor = wired
    provider = ScriptedProvider(
        turns=[[ _call("system_info", "c1")]],
        synthesis="final",
    )
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("inspect the system") == "final"
    tool_msgs = [m for m in provider.tool_turns[-1] if m.role == "tool"]
    assert len(tool_msgs) == 1
    body = json.loads(tool_msgs[0].content)
    assert body["ok"] is True
    assert "output" in body and "error" in body
    assert "truncated" not in body


@pytest.mark.asyncio
async def test_parallel_distinct_calls_all_execute_once(wired, tmp_path):
    settings, registry, executor = wired
    provider = ScriptedProvider(
        turns=[[
            _call("system_info", "c1"),
            _call("system_cpu", "c2"),
            _call("system_platform", "c3"),
        ]],
        synthesis="done",
    )
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("everything") == "done"
    assert len(_audit_lines(tmp_path)) == 3


# -- duplicate prevention -----------------------------------------------------

@pytest.mark.asyncio
async def test_repeated_call_id_reuses_cache(wired, tmp_path):
    settings, registry, executor = wired
    settings.max_tool_iterations = 3
    provider = ScriptedProvider(
        turns=[[ _call("system_info", "c1")]] * 3,
        synthesis="   ",
    )
    assistant = _assistant(settings, provider, registry, executor)
    reply = await assistant.achat("again and again")
    assert reply.strip()
    assert len(_audit_lines(tmp_path)) == 1  # executed once, never re-run
    assert provider.with_tools_calls == 3  # model still bounded by iterations


@pytest.mark.asyncio
async def test_same_name_and_args_new_id_reuses_cache(wired, tmp_path):
    settings, registry, executor = wired
    settings.max_tool_iterations = 2
    provider = ScriptedProvider(
        turns=[[ _call("system_info", "c1")], [_call("system_info", "c2")]],
        synthesis="done",
    )
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("poll twice") == "done"
    assert len(_audit_lines(tmp_path)) == 1
    reused = [m for m in provider.plain_turns[-1] if m.role == "tool"]
    assert any("reused" in m.content for m in reused)


# -- multi-step completion + summary ------------------------------------------

@pytest.mark.asyncio
async def test_multi_step_ends_with_synthesis(wired, tmp_path):
    settings, registry, executor = wired
    provider = ScriptedProvider(
        turns=[
            [_call("system_info", "c1")],
            [_call("system_cpu", "c2")],
        ],
        synthesis="System report complete.",
    )
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("full system report") == "System report complete."
    # Both steps executed exactly once; the loop then asks once more and
    # the model answers without tools (final synthesis via that turn).
    assert len(_audit_lines(tmp_path)) == 2
    assert provider.plain_calls == 0


@pytest.mark.asyncio
async def test_budget_exhaustion_summarizes_steps_and_blockers(wired):
    settings, registry, executor = wired
    settings.max_tool_iterations = 2
    provider = ScriptedProvider(
        turns=[
            [_call("system_info", "c1")],
            [_call("does_not_exist", "c2")],
        ],
        synthesis="   ",
    )
    assistant = _assistant(settings, provider, registry, executor)
    reply = await assistant.achat("do two things")
    assert "couldn't compose" in reply
    assert "1 of 2" in reply
    assert "does_not_exist" in reply


# -- invalid args / unknown tools surfaced to the model ------------------------

@pytest.mark.asyncio
async def test_invalid_arguments_reported_not_executed(wired, tmp_path):
    settings, registry, executor = wired
    provider = ScriptedProvider(
        turns=[[ _call("filesystem_write", "c1", {"path": str(tmp_path / "x.txt")})]],
        synthesis="needs content",
    )
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("write it") == "needs content"
    tool_msgs = [m for m in provider.tool_turns[-1] if m.role == "tool"]
    body = json.loads(tool_msgs[0].content)
    assert body["ok"] is False
    assert body["error"]
    assert not (tmp_path / "x.txt").exists()


@pytest.mark.asyncio
async def test_unknown_tool_name_reported(wired):
    settings, registry, executor = wired
    provider = ScriptedProvider(
        turns=[[ _call("nope_missing", "c1")]],
        synthesis="unknown handled",
    )
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("use missing") == "unknown handled"
    tool_msgs = [m for m in provider.tool_turns[-1] if m.role == "tool"]
    assert "Unknown tool" in json.loads(tool_msgs[0].content)["error"]


# -- tool exceptions + provider errors -----------------------------------------

@pytest.mark.asyncio
async def test_crashing_tool_becomes_failed_result(wired, tmp_path):
    from pydantic import BaseModel

    from app.tools.base import Tool, ToolResult
    from app.tools.registry import ToolRegistry
    from app.security.policy import PermissionLevel

    class BoomInput(BaseModel):
        pass

    class BoomTool(Tool):
        name = "test.boom"
        description = "Always raises (bug simulation)."
        risk_level = PermissionLevel.SAFE
        input_model = BoomInput

        def execute(self, validated):
            raise RuntimeError("simulated bug")

    settings, _, _ = wired
    registry = ToolRegistry()
    registry.register(BoomTool())
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(tmp_path / "boom.jsonl"), confirm=auto_deny
    )
    result = await executor.execute("test.boom", {})
    assert result.ok is False
    assert "failed" in (result.error or "").lower()
    lines = (tmp_path / "boom.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["status"] == "failed"

    provider = ScriptedProvider(turns=[[ _call("test_boom", "c1")]], synthesis="recovered")
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("run boom") == "recovered"


@pytest.mark.asyncio
async def test_provider_error_mid_loop_is_honest(wired):
    settings, registry, executor = wired

    class Flaky(ScriptedProvider):
        async def send_message_with_tools(self, messages, tools):
            self.with_tools_calls += 1
            if self.with_tools_calls == 1:
                return ChatResponse(
                    content="", model="s", provider="s",
                    tool_calls=[_call("system_info", "c1")],
                )
            raise ProviderUnavailable("AI provider error (HTTP 503, upstream)")

    assistant = _assistant(settings, Flaky([]), registry, executor)
    assert await assistant.achat("research") == PROVIDER_ERROR_REPLY


# -- budgets: iterations + wall clock ------------------------------------------

@pytest.mark.asyncio
async def test_wall_clock_budget_stops_loop(wired):
    settings, registry, executor = wired
    settings.max_tool_iterations = 5
    settings.tool_loop_timeout_seconds = 0.05

    class SlowFirst(ScriptedProvider):
        async def send_message_with_tools(self, messages, tools):
            self.with_tools_calls += 1
            if self.with_tools_calls == 1:
                await asyncio.sleep(0.1)
            return ChatResponse(
                content="", model="s", provider="s",
                tool_calls=[_call("system_info", f"c{self.with_tools_calls}")],
            )

    provider = SlowFirst([])
    assistant = _assistant(settings, provider, registry, executor)
    reply = await assistant.achat("take forever")
    assert provider.with_tools_calls == 1  # first turn always runs
    assert provider.plain_calls == 1  # then synthesis, no more tools
    assert reply == "SYNTHESIZED"


# -- truncation -----------------------------------------------------------------

@pytest.mark.asyncio
async def test_large_tool_result_truncated_with_marker(wired, tmp_path):
    settings, registry, executor = wired
    settings.tool_result_max_chars = 60
    big = tmp_path / "big.txt"
    big.write_text("A" * 500)
    provider = ScriptedProvider(
        turns=[[ _call("filesystem_read", "c1", {"path": str(big)})]],
        synthesis="summarized",
    )
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("read it") == "summarized"
    tool_msgs = [m for m in provider.tool_turns[-1] if m.role == "tool"]
    body = json.loads(tool_msgs[0].content)
    assert body.get("truncated")
    assert len(body["output"]) <= 60


# -- empty responses --------------------------------------------------------------

@pytest.mark.asyncio
async def test_empty_model_turn_never_returns_blank(wired):
    settings, registry, executor = wired

    class Empty(ScriptedProvider):
        async def send_message_with_tools(self, messages, tools):
            self.with_tools_calls += 1
            return ChatResponse(content="   ", model="s", provider="s", tool_calls=[])

    assistant = _assistant(settings, Empty([]), registry, executor)
    reply = await assistant.achat("say nothing?")
    assert reply.strip()


@pytest.mark.asyncio
async def test_empty_final_turn_keeps_earlier_text(wired):
    settings, registry, executor = wired
    provider = ScriptedProvider(
        turns=[[_call("system_info", "c1")], "half answer"],
        synthesis="",
    )
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("tell me") == "half answer"


# -- approvals ---------------------------------------------------------------------

@pytest.mark.asyncio
async def test_high_risk_delete_denied_without_approval(wired, tmp_path):
    settings, registry, executor = wired
    target = tmp_path / "keep.txt"
    target.write_text("important")
    provider = ScriptedProvider(
        turns=[[ _call("filesystem_delete", "c1", {"path": str(target)})]],
        synthesis="I need your approval to delete that file.",
    )
    assistant = _assistant(settings, provider, registry, executor)
    reply = await assistant.achat("delete it", confirm=auto_deny)
    assert "approval" in reply.lower()
    assert target.exists()


@pytest.mark.asyncio
async def test_high_risk_delete_approved_executes(wired, tmp_path):
    settings, registry, executor = wired
    target = tmp_path / "gone.txt"
    target.write_text("bye")
    provider = ScriptedProvider(
        turns=[[ _call("filesystem_delete", "c1", {"path": str(target)})]],
        synthesis="Deleted.",
    )
    assistant = _assistant(settings, provider, registry, executor)
    assert await assistant.achat("delete it", confirm=auto_approve) == "Deleted."
    assert not target.exists()


# -- secret redaction (audit + events) ----------------------------------------------

@pytest.mark.asyncio
async def test_secrets_redacted_in_audit_and_events(wired, tmp_path):
    settings, registry, executor = wired
    seen = []

    def grab(event):
        seen.append(event)

    event_bus.subscribe(EventType.TOOL_STARTED, grab)
    try:
        provider = ScriptedProvider(
            turns=[[ _call(
                "system_info", "c1",
                {"api_key": "super-secret-xyz", "token": "nvapi-abc123xyz"},
            )]],
            synthesis="done",
        )
        assistant = _assistant(settings, provider, registry, executor)
        assert await assistant.achat("inspect") == "done"
    finally:
        event_bus.unsubscribe(EventType.TOOL_STARTED, grab)
    dumped = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "super-secret-xyz" not in dumped
    assert "nvapi-abc123xyz" not in dumped
    assert "***" in dumped
    assert seen, "expected a TOOL_STARTED event"
    assert "super-secret-xyz" not in json.dumps(seen[0].payload, default=str)
    assert "nvapi-abc123xyz" not in json.dumps(seen[0].payload, default=str)


# -- task tracker unit behavior ------------------------------------------------------

def test_task_tracker_progress_and_summary():
    tracker = TaskTracker(goal="demo")
    assert tracker.progress_note() == ""
    assert "No tool steps" in tracker.summary()
    tracker.record("web_search", True)
    tracker.record("web_fetch", False)
    note = tracker.progress_note()
    assert "web_search → ok" in note
    assert "web_fetch → FAILED" in note
    assert "Do not repeat" in note
    summary = tracker.summary()
    assert "1 of 2" in summary
    assert "web_fetch" in summary


# -- NVIDIA cloud path starts no local models ------------------------------------------

@pytest.mark.asyncio
async def test_nvidia_cloud_achat_starts_no_local_servers(tmp_path, monkeypatch):
    import importlib.util
    import subprocess

    from app.brain.provider import NvidiaNemotronProvider

    # Local-model server machinery was removed: nothing to invoke.
    assert importlib.util.find_spec("app.launcher.servers") is None
    assert importlib.util.find_spec("app.launcher.handoff") is None

    def boom(*a, **k):
        raise AssertionError("cloud path must not start servers")

    monkeypatch.setattr(subprocess, "Popen", boom)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "cloud reply"}}]})

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
    settings = Settings(
        ai_provider="nvidia",
        ai_model="nexus-default",
        ai_api_key="",
        ai_base_url="",
        ai_timeout_seconds=30.0,
        nvidia_api_key="test-nvapi-key",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )
    from app.main import build_assistant

    assistant = build_assistant(settings)
    assert not hasattr(assistant, "synthesis_handoff")
    assert assistant.provider.name == "nvidia"
    assert await assistant.achat("Explain GPU computing briefly.") == "cloud reply"


def test_nvidia_provider_payload_untouched_by_phase7():
    from app.brain.provider import NvidiaNemotronProvider

    provider = NvidiaNemotronProvider(
        base_url="https://integrate.api.nvidia.com/v1",
        model="nvidia/nemotron-3.5-lightning-30b-a3b",
        api_key="test-key",
    )
    extras = provider._extra_payload()
    assert extras["temperature"] == 1.0
    assert extras["top_p"] == 0.95
    assert extras["max_tokens"] == 16384
    assert "extra_body" not in extras
    assert extras["reasoning_budget"] == 16384
