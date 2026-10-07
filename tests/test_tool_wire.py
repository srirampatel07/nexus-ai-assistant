"""Wire-format regression tests for OpenAI-compatible tool calling (Phase 2 fix).

Groq strictly rejects `tool_calls` on non-assistant messages and echoed
assistant calls without `type`/`function`. These tests lock the wire
hygiene at the HTTP level using httpx.MockTransport. No real API calls.
"""

import json

import httpx
import pytest

from app.brain.provider import (
    ChatMessage,
    OpenAICompatibleProvider,
    describe_http_error,
)
from app.core.config import Settings
from app.core.exceptions import ProviderUnavailable
from app.security.audit import AuditLogger
from app.tools import build_default_registry
from app.tools.executor import ToolExecutor
from app.tools.schemas import LlmTool, LlmToolCall


def _provider():
    return OpenAICompatibleProvider(
        base_url="https://fake.test/v1", model="m", api_key="k"
    )


class _Capture:
    """Fake httpx.AsyncClient capturing requests, replaying queued responses."""

    real_client = httpx.AsyncClient

    def __init__(self, responses: list[httpx.Response]):
        self._responses = list(responses)
        self.requests: list[dict] = []

    def __call__(self, *a, **k):
        capture = self

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                capture.requests.append(json)
                return capture._responses.pop(0)

            async def get(self, *a, **k):
                raise AssertionError("unexpected GET")

        return FakeClient()


@pytest.fixture
def tools():
    return [
        LlmTool(
            name="system.info",
            description="Get system info.",
            parameters={"type": "object", "properties": {}},
        )
    ]


@pytest.fixture
def messages():
    return [
        ChatMessage(role="system", content="You are NEXUS."),
        ChatMessage(role="user", content="What operating system am I running?"),
    ]


def test_to_wire_drops_empty_tool_fields():
    wire = ChatMessage(role="user", content="hi").to_wire()
    assert "tool_calls" not in wire
    assert "tool_call_id" not in wire
    assert wire == {"role": "user", "content": "hi"}


def test_to_wire_keeps_populated_tool_calls():
    call = LlmToolCall(id="c1", name="system.info", arguments={}).to_wire()
    wire = ChatMessage(role="assistant", content="", tool_calls=[call]).to_wire()
    assert wire["tool_calls"][0]["type"] == "function"


def test_llm_tool_call_to_wire_shape():
    wire = LlmToolCall(id="c1", name="system.info", arguments={"a": 1}).to_wire()
    assert wire["id"] == "c1"
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "system.info"
    assert json.loads(wire["function"]["arguments"]) == {"a": 1}


@pytest.mark.asyncio
async def test_plain_request_has_no_tool_fields(monkeypatch, messages):
    capture = _Capture(
        [httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})]
    )
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    await _provider().send_message(messages)
    for msg in capture.requests[0]["messages"]:
        assert "tool_calls" not in msg
        assert "tool_call_id" not in msg


@pytest.mark.asyncio
async def test_with_tools_request_shape(monkeypatch, messages, tools):
    capture = _Capture(
        [httpx.Response(200, json={"choices": [{"message": {"content": "done"}}]})]
    )
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    resp = await _provider().send_message_with_tools(messages, tools)
    assert resp.tool_calls == []
    body = capture.requests[0]
    assert body["tool_choice"] == "auto"
    assert body["tools"][0]["function"]["name"] == "system.info"
    for msg in body["messages"]:
        assert "tool_calls" not in msg


@pytest.mark.asyncio
async def test_echoed_tool_calls_carry_type(monkeypatch, messages, tools, tmp_path):
    first = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {"name": "system.info", "arguments": "{}"},
                            }
                        ],
                    }
                }
            ]
        },
    )
    second = httpx.Response(
        200, json={"choices": [{"message": {"content": "Windows."}}]}
    )
    capture = _Capture([first, second])
    monkeypatch.setattr(httpx, "AsyncClient", capture)

    from app.brain.assistant import NexusAssistant

    settings = Settings(ai_provider="echo", nexus_allowed_roots=["C:\\chanti\\nexus"])
    registry = build_default_registry(settings)
    executor = ToolExecutor(
        registry=registry,
        audit=AuditLogger(tmp_path / "audit.jsonl"),
        confirm=None,
    )
    assistant = NexusAssistant(
        settings=settings,
        provider=_provider(),
        registry=registry,
        executor=executor,
    )
    assert await assistant.achat("What operating system am I running?") == "Windows."

    turn2 = capture.requests[1]["messages"]
    assistant_echo = next(m for m in turn2 if m["role"] == "assistant")
    assert assistant_echo["tool_calls"][0]["type"] == "function"
    assert assistant_echo["tool_calls"][0]["function"]["name"] == "system.info"
    tool_msg = next(m for m in turn2 if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_malformed_tool_calls_yield_plain_answer(monkeypatch, messages, tools):
    body = httpx.Response(
        200,
        json={
            "choices": [
                {"message": {"content": "plain", "tool_calls": [{"junk": True}]}}
            ]
        },
    )
    capture = _Capture([body])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    resp = await _provider().send_message_with_tools(messages, tools)
    assert resp.tool_calls == []
    assert resp.content == "plain"


@pytest.mark.asyncio
async def test_400_error_includes_body_snippet(monkeypatch, messages, tools):
    capture = _Capture(
        [httpx.Response(400, json={"error": {"message": "bad things happened"}})]
    )
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await _provider().send_message_with_tools(messages, tools)
    assert "HTTP 400" in str(exc_info.value)
    assert "bad things happened" in str(exc_info.value)


@pytest.mark.asyncio
async def test_400_error_body_is_truncated(monkeypatch, messages, tools):
    capture = _Capture([httpx.Response(400, text="x" * 5000)])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    with pytest.raises(ProviderUnavailable) as exc_info:
        await _provider().send_message_with_tools(messages, tools)
    assert len(str(exc_info.value)) < 2000


def test_describe_http_error_redacts_secrets():
    msg = describe_http_error(
        400, 'bad key "Authorization: Bearer secret123" gsk_abcdefgh12345678', "m"
    )
    assert "secret123" not in msg
    assert "gsk_abcdefgh12345678" not in msg
    assert "HTTP 400" in msg
