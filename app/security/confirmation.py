"""Confirmation system for NEXUS tools (Phase 2).

Tools flagged as requiring confirmation are NEVER executed silently.
The ToolExecutor pauses and asks a `ConfirmCallback`; the text CLI injects
an interactive console prompt, while tests/non-interactive contexts inject
an auto-deny callback (safe default).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConfirmationRequest:
    tool_name: str
    description: str
    risk_level: str
    arguments_summary: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""


ConfirmCallback = Callable[[ConfirmationRequest], Awaitable[bool]]


async def auto_deny(_request: ConfirmationRequest) -> bool:
    """Safe default: deny every confirmation (used when non-interactive)."""
    return False


async def auto_approve(_request: ConfirmationRequest) -> bool:
    """Approve every confirmation. TESTS ONLY — never use in production."""
    return True


def format_request(request: ConfirmationRequest) -> str:
    """Render a human-readable approval prompt (used by the console CLI)."""
    lines = [
        f"Tool request: {request.tool_name}",
        f"Description: {request.description}",
        f"Risk: {request.risk_level}",
    ]
    if request.arguments_summary:
        lines.append(f"Arguments: {request.arguments_summary}")
    lines.append("Allow? [y/N]")
    return "\n".join(lines)
