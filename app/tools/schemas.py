"""Provider-neutral AI tool-call structures (Phase 2).

NEXUS talks about tools in its own vocabulary (`LlmTool`, `LlmToolCall`);
only the conversion helpers in this module know the OpenAI-compatible
wire format. No vendor logic lives in the assistant core.

Conversation shape supported per turn:
    messages -> tools -> tool_calls -> tool results -> final response
"""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field


class LlmTool(BaseModel):
    """One tool definition as presented to the AI model."""

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class LlmToolCall(BaseModel):
    """One tool call requested by the AI model (provider-neutral core)."""

    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        """Serialize back to the OpenAI-compatible `tool_calls` entry shape.

        Providers require echoed calls to carry `type: function` and
        `function: {name, arguments-as-JSON-string}` — a bare
        `{id, name, arguments}` object is rejected (HTTP 400).
        """
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }


def to_openai_tools(tools: list[LlmTool]) -> list[dict[str, Any]]:
    """Convert provider-neutral definitions to OpenAI-compatible `tools`."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters or {"type": "object", "properties": {}},
            },
        }
        for t in tools
    ]


def parse_openai_tool_calls(message: dict[str, Any]) -> list[LlmToolCall]:
    """Parse `tool_calls` from an OpenAI-compatible response message.

    Malformed entries are skipped (never faked, never crash the loop).
    """
    calls: list[LlmToolCall] = []
    raw = message.get("tool_calls") or []
    if not isinstance(raw, list):
        return calls
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        func = entry.get("function") or {}
        if not isinstance(func, dict) or not func.get("name"):
            continue
        args = func.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except (ValueError, json.JSONDecodeError):
                continue
        if not isinstance(args, dict):
            continue
        calls.append(
            LlmToolCall(
                id=str(entry.get("id") or ""),
                name=str(func.get("name")),
                arguments=args,
            )
        )
    return calls


def tool_result_message(call_id: str, result: dict[str, Any]) -> dict[str, Any]:
    """Format a tool outcome as an OpenAI-compatible `tool` message."""
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "content": json.dumps(result, default=str),
    }
