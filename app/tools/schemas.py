"""Provider-neutral AI tool-call structures (Phase 2).

NEXUS talks about tools in its own vocabulary (`LlmTool`, `LlmToolCall`);
only the conversion helpers in this module know the OpenAI-compatible
wire format. No vendor logic lives in the assistant core.

Conversation shape supported per turn:
    messages -> tools -> tool_calls -> tool results -> final response
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field

# OpenAI-compatible function names (enforced by OpenRouter and other
# strict providers): only letters, digits, underscores, and hyphens.
# Internal NEXUS tool names historically use dots (e.g. "system.info"),
# which upstream rejects with HTTP 400. Model-visible names are therefore
# the normalized form (dots -> underscores); the registry owns the
# bidirectional mapping and the executor resolves both spellings.
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def to_external_name(internal_name: str) -> str:
    """Map an internal tool name to its model-visible form.

    Deterministic: dots become underscores, then any remaining character
    outside [A-Za-z0-9_-] becomes an underscore. E.g. "system.info" ->
    "system_info". This function never fails; use `is_valid_tool_name`
    to check whether the result is transmittable.
    """
    return re.sub(r"[^A-Za-z0-9_-]", "_", (internal_name or "").replace(".", "_"))


def is_valid_tool_name(name: str) -> bool:
    """True when `name` satisfies the provider naming restriction."""
    return bool(name and TOOL_NAME_RE.match(name))


def validate_tool_definitions(tools: list[LlmTool]) -> None:
    """Fail locally on malformed tool definitions with useful diagnostics.

    Checks: non-empty valid function names, no post-normalization
    collisions, string descriptions, and object-shaped JSON schemas.
    Raises ValueError describing the first problem found. Callers
    (provider) convert this into a redacted ProviderError without any
    network round-trip.
    """
    seen: dict[str, str] = {}
    for index, tool in enumerate(tools):
        where = f"tool definition at index {index}"
        name = getattr(tool, "name", "")
        if not name or not is_valid_tool_name(name):
            raise ValueError(
                f"Invalid {where}: function name {name!r} is not allowed. "
                "Only a-z, A-Z, 0-9, underscores, and dashes are allowed. "
                f"Internal dotted names must be mapped first (e.g. "
                f"{to_external_name(name or 'system.info')!r})."
            )
        if name in seen:
            raise ValueError(
                f"Invalid {where}: duplicate function name {name!r} "
                f"(collides after normalization)."
            )
        seen[name] = name
        if not isinstance(getattr(tool, "description", ""), str):
            raise ValueError(f"Invalid {where} ({name!r}): description must be a string.")
        params = getattr(tool, "parameters", None)
        if not isinstance(params, dict):
            raise ValueError(
                f"Invalid {where} ({name!r}): parameters must be an object schema."
            )
        if params.get("type", "object") != "object":
            raise ValueError(
                f"Invalid {where} ({name!r}): parameters type must be 'object'."
            )


class LlmTool(BaseModel):
    """One tool definition as presented to the AI model.

    `name` must already be the model-visible (external) form satisfying
    the provider naming restriction. Build definitions via
    `ToolRegistry.tool_definitions()` — never hand-craft dotted internal
    names for the wire.
    """

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class LlmToolCall(BaseModel):
    """One tool call requested by the AI model (provider-neutral core).

    `extra` carries provider-specific metadata verbatim (e.g. Gemini's
    `extra_content` thought signatures, which its API requires back
    unchanged on continuation). It is never interpreted, only echoed.
    """

    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        """Serialize back to the OpenAI-compatible `tool_calls` entry shape.

        Providers require echoed calls to carry `type: function` and
        `function: {name, arguments-as-JSON-string}` — a bare
        `{id, name, arguments}` object is rejected (HTTP 400). Any
        provider metadata received with the call (e.g. thought
        signatures) is re-emitted verbatim alongside those fields;
        entries without metadata serialize exactly as before.
        """
        wire: dict[str, Any] = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments),
            },
        }
        for key, value in self.extra.items():
            if key not in wire:
                wire[key] = value
        return wire


def to_openai_tools(tools: list[LlmTool]) -> list[dict[str, Any]]:
    """Convert provider-neutral definitions to OpenAI-compatible `tools`.

    Validates first: incompatible names or malformed schemas raise
    ValueError locally (never sent, never a silent HTTP 400).
    """
    validate_tool_definitions(tools)
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
    Provider-specific entry fields beyond id/type/function (e.g. Gemini's
    `extra_content` thought signatures) are preserved verbatim on the
    call so the echo in the next request carries them back unchanged.
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
        extra = {
            key: value
            for key, value in entry.items()
            if key not in ("id", "type", "function")
        }
        calls.append(
            LlmToolCall(
                id=str(entry.get("id") or ""),
                name=str(func.get("name")),
                arguments=args,
                extra=extra,
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
