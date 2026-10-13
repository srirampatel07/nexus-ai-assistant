"""Gemini tool-calling compatibility tests (Phase 6 fix).

Gemini thinking models (e.g. gemini-3.8-flash) attach opaque
thought-signature metadata to function-call parts, and the API rejects
(HTTP 400) any continuation that drops it:
"Function call is missing a thought_signature in functionCall parts."

NEXUS must therefore preserve provider tool-call metadata verbatim:
parse keeps unknown entry fields, and the echoed assistant message
re-emits them byte-identical. Nothing is fabricated — signatures are
only ever echoed back exactly as received. All HTTP mocked; offline.
"""

import httpx
import pytest

from app.brain.assistant import NexusAssistant
from app.brain.provider import OpenAICompatibleProvider
from app.core.config import Settings
from app.security.audit import AuditLogger
from app.tools import build_default_registry
from app.tools.executor import ToolExecutor
from app.tools.schemas import LlmToolCall, parse_openai_tool_calls

_SIG_A = "CiQyMGFiYzEyMw=="
_SIG_B = "CiQ0NTZkZWYzOTE="


def _gemini_entry(call_id, name, args, sig):
    """Gemini OpenAI-compatible tool-call entry with signature metadata."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": args},
        "extra_content": {"google": {"thought_signature": sig}},
    }


def test_parse_preserves_signature_metadata():
    message = {
        "content": "",
        "tool_calls": [_gemini_entry("call_1", "web.search", '{"query": "x"}', _SIG_A)],
    }
    calls = parse_openai_tool_calls(message)
    assert len(calls) == 1
    assert calls[0].name == "web.search"
    assert calls[0].arguments == {"query": "x"}
    assert calls[0].extra == {
        "extra_content": {"google": {"thought_signature": _SIG_A}}
    }


def test_wire_reemits_signature_verbatim():
    call = LlmToolCall(
        id="call_1",
        name="web.search",
        arguments={"query": "x"},
        extra={"extra_content": {"google": {"thought_signature": _SIG_A}}},
    )
    wire = call.to_wire()
    assert wire["id"] == "call_1"
    assert wire["type"] == "function"
    assert wire["function"]["name"] == "web.search"
    assert wire["extra_content"] == {"google": {"thought_signature": _SIG_A}}


def test_wire_without_metadata_is_unchanged():
    """Strict-provider path: no extras in, no extra keys out."""
    wire = LlmToolCall(id="c1", name="system.info", arguments={}).to_wire()
    assert set(wire) == {"id", "type", "function"}
    assert set(wire["function"]) == {"name", "arguments"}


class _Capture:
    """Fake httpx.AsyncClient capturing requests, replaying responses."""

    real_client = httpx.AsyncClient

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

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

        return FakeClient()


@pytest.mark.asyncio
async def test_gemini_signature_survives_continuation(monkeypatch, tmp_path):
    """Exact live failure: turn-2 continuation must echo the signature."""
    first = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            _gemini_entry(
                                "call_1",
                                "web.search",
                                '{"query": "latest Python release"}',
                                _SIG_A,
                            )
                        ],
                    }
                }
            ]
        },
    )
    second = httpx.Response(
        200, json={"choices": [{"message": {"content": "Python 3.14 is latest."}}]}
    )
    capture = _Capture([first, second])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    provider = OpenAICompatibleProvider(
        base_url="https://fake.test/v1", model="gemini-3.8-flash", api_key="k"
    )
    settings = Settings(ai_provider="echo", nexus_allowed_roots=["C:\\chanti\\nexus"])
    # Point search at the mock transport instead of the live endpoint.
    from app.web.client import WebContext

    registry = build_default_registry(settings)
    search = registry.get("web.search")
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body><p>no results</p></body></html>",
        )
    )
    search.ctx = WebContext(
        client_factory=lambda timeout: httpx.Client(
            transport=transport, timeout=timeout
        ),
        dns_resolver=lambda host: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(tmp_path / "audit.jsonl")
    )
    assistant = NexusAssistant(
        settings=settings, provider=provider, registry=registry, executor=executor
    )
    reply = await assistant.achat("Search for the latest Python release.")
    assert reply == "Python 3.14 is latest."
    assert len(capture.requests) == 2
    turn2 = capture.requests[1]["messages"]
    echo = next(m for m in turn2 if m["role"] == "assistant")
    entry = echo["tool_calls"][0]
    assert entry["id"] == "call_1"
    assert entry["extra_content"] == {"google": {"thought_signature": _SIG_A}}


@pytest.mark.asyncio
async def test_gemini_signatures_preserved_across_turns(monkeypatch, tmp_path):
    """Sequential tool steps each keep their own signature in the echo."""
    turn1 = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            _gemini_entry("c1", "system.info", "{}", _SIG_A)
                        ],
                    }
                }
            ]
        },
    )
    turn2 = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": "",
                        "tool_calls": [
                            _gemini_entry("c2", "system.cpu", "{}", _SIG_B)
                        ],
                    }
                }
            ]
        },
    )
    turn3 = httpx.Response(
        200, json={"choices": [{"message": {"content": "done."}}]}
    )
    capture = _Capture([turn1, turn2, turn3])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    provider = OpenAICompatibleProvider(
        base_url="https://fake.test/v1", model="gemini-3.8-flash", api_key="k"
    )
    settings = Settings(ai_provider="echo", nexus_allowed_roots=["C:\\chanti\\nexus"])
    registry = build_default_registry(settings)
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(tmp_path / "audit.jsonl")
    )
    assistant = NexusAssistant(
        settings=settings, provider=provider, registry=registry, executor=executor
    )
    assert await assistant.achat("inspect the system") == "done."
    assert len(capture.requests) == 3
    final_turn = capture.requests[2]["messages"]
    echoes = [m for m in final_turn if m["role"] == "assistant"]
    sigs = [
        tc.get("extra_content", {}).get("google", {}).get("thought_signature")
        for echo in echoes
        for tc in echo["tool_calls"]
    ]
    assert sigs == [_SIG_A, _SIG_B]
