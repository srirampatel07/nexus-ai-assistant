"""Provider-neutral tool-call structures + OpenAI-compatible mapping (Phase 2).

No real API calls: HTTP is faked with httpx.MockTransport.
"""

import httpx
import pytest

from app.brain.provider import (
    ChatMessage,
    EchoProvider,
    OpenAICompatibleProvider,
)
from app.tools.schemas import (
    LlmTool,
    parse_openai_tool_calls,
    to_openai_tools,
    tool_result_message,
)


def _tool():
    return LlmTool(
        name="system.info",
        description="Get system info.",
        parameters={"type": "object", "properties": {}},
    )


def test_to_openai_tools_shape():
    wire = to_openai_tools([_tool()])
    assert wire[0]["type"] == "function"
    assert wire[0]["function"]["name"] == "system.info"
    assert wire[0]["function"]["parameters"]["type"] == "object"


def test_parse_valid_tool_calls():
    msg = {
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "system.info", "arguments": "{}"},
            }
        ]
    }
    calls = parse_openai_tool_calls(msg)
    assert len(calls) == 1
    assert calls[0].name == "system.info"
    assert calls[0].arguments == {}


def test_parse_skips_malformed_entries():
    msg = {
        "tool_calls": [
            {"id": "bad", "function": {"name": "", "arguments": "{}"}},
            {"id": "bad2", "function": {"name": "x", "arguments": "{not json"}},
            "not-a-dict",
            {"id": "good", "function": {"name": "system.info", "arguments": {"a": 1}}},
        ]
    }
    calls = parse_openai_tool_calls(msg)
    assert [c.name for c in calls] == ["system.info"]


def test_parse_no_tool_calls():
    assert parse_openai_tool_calls({"content": "hi"}) == []
    assert parse_openai_tool_calls({"tool_calls": "nonsense"}) == []


def test_tool_result_message_shape():
    msg = tool_result_message("call_1", {"ok": True})
    assert msg["role"] == "tool"
    assert msg["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_echo_falls_back_to_no_calls():
    provider = EchoProvider()
    resp = await provider.send_message_with_tools(
        [ChatMessage(role="user", content="hi")], [_tool()]
    )
    assert resp.tool_calls == []
    assert resp.content


def _patched_provider(monkeypatch, handler):
    provider = OpenAICompatibleProvider(
        base_url="https://fake.test/v1", model="m", api_key="k"
    )
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
    return provider


@pytest.mark.asyncio
async def test_openai_provider_parses_tool_calls(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert "tools" in request.read().decode()
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_9",
                                    "type": "function",
                                    "function": {
                                        "name": "system.info",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            },
        )

    provider = _patched_provider(monkeypatch, handler)
    resp = await provider.send_message_with_tools(
        [ChatMessage(role="user", content="info please")], [_tool()]
    )
    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].id == "call_9"


@pytest.mark.asyncio
async def test_openai_provider_falls_back_when_tools_rejected(monkeypatch):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(400, json={"error": {"message": "tools not supported"}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "plain"}}]})

    provider = _patched_provider(monkeypatch, handler)
    resp = await provider.send_message_with_tools(
        [ChatMessage(role="user", content="hi")], [_tool()]
    )
    assert resp.tool_calls == []
    assert resp.content == "plain"
