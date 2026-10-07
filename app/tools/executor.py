"""Central tool executor (Phase 2).

No tool bypasses the executor. Every call goes through:
find -> validate -> assess risk -> permission -> confirmation ->
execute (timeout) -> audit -> structured result.

Expected failures (unknown tool, bad input, denied, timeout, tool error)
are returned as failed `ToolResult`s — never raised — so the AI loop can
report them honestly. Only programming errors raise.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from app.core.exceptions import ToolError
from app.core.logger import get_logger
from app.security.audit import AuditLogger
from app.security.confirmation import ConfirmCallback, ConfirmationRequest, auto_deny
from app.security.policy import PermissionLevel, decide
from app.tools.base import ToolResult
from app.tools.registry import ToolRegistry

log = get_logger("nexus.executor")


class ToolExecutor:
    """Enforcing dispatcher for registered tools."""

    def __init__(
        self,
        registry: ToolRegistry,
        audit: AuditLogger | None = None,
        confirm: ConfirmCallback | None = None,
    ) -> None:
        self.registry = registry
        self.audit = audit or AuditLogger()
        self.confirm = confirm or auto_deny

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        confirm: ConfirmCallback | None = None,
    ) -> ToolResult:
        """Execute one tool call with full enforcement. Never raises for
        expected failures; returns a failed ToolResult instead."""
        args = arguments or {}
        ask = confirm or self.confirm
        started = time.monotonic()

        def elapsed_ms() -> int:
            return int((time.monotonic() - started) * 1000)

        # 1. Find tool.
        try:
            tool = self.registry.get(tool_name)
        except ToolError as exc:
            self.audit.record(
                tool_name=tool_name,
                arguments=args,
                risk_level="UNKNOWN",
                permission="denied:unknown_tool",
                confirmation="not_required",
                status="failed",
                duration_ms=elapsed_ms(),
                error=str(exc),
            )
            return ToolResult(ok=False, error=f"Unknown tool '{tool_name}'.")

        # 2. Validate input.
        try:
            validated = tool.validate(args)
        except ToolError as exc:
            self.audit.record(
                tool_name=tool_name,
                arguments=args,
                risk_level=tool.risk_level.value,
                permission="denied:validation",
                confirmation="not_required",
                status="failed",
                duration_ms=elapsed_ms(),
                error=str(exc),
            )
            return ToolResult(ok=False, error=str(exc))

        # 3. Assess dynamic risk + permission decision.
        try:
            risk = tool.assess(validated)
        except ToolError as exc:
            # Tool refused these arguments (e.g. blocked command).
            self.audit.record(
                tool_name=tool_name,
                arguments=args,
                risk_level=PermissionLevel.BLOCKED.value,
                permission="denied:policy",
                confirmation="not_required",
                status="denied",
                duration_ms=elapsed_ms(),
                error=str(exc),
            )
            return ToolResult(ok=False, error=str(exc))
        decision = decide(risk)
        if not decision.allowed:
            self.audit.record(
                tool_name=tool_name,
                arguments=args,
                risk_level=risk.value,
                permission="denied:policy",
                confirmation="not_required",
                status="denied",
                duration_ms=elapsed_ms(),
                error=decision.reason,
            )
            return ToolResult(ok=False, error=f"Denied by policy: {decision.reason}")

        # 4-5. Confirmation when required (level or tool flag).
        confirmation = "not_required"
        if decision.requires_confirmation or tool.requires_confirmation:
            confirmation = "pending"
            request = ConfirmationRequest(
                tool_name=tool.name,
                description=tool.description,
                risk_level=risk.value,
                arguments_summary=dict(args),
            )
            try:
                approved = await ask(request)
            except Exception as exc:  # a broken callback must not execute tools
                log.warning("confirmation callback failed: %s", exc)
                approved = False
            confirmation = "approved" if approved else "denied"
            if not approved:
                self.audit.record(
                    tool_name=tool_name,
                    arguments=args,
                    risk_level=risk.value,
                    permission="allowed",
                    confirmation=confirmation,
                    status="denied",
                    duration_ms=elapsed_ms(),
                    error="Denied: confirmation required and not granted.",
                )
                return ToolResult(
                    ok=False,
                    error="Not executed: this tool needs your approval. "
                    "Re-run in interactive mode and confirm to allow it.",
                )

        # 6-8. Execute with timeout, capture result.
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(tool.execute, validated),
                timeout=tool.timeout_seconds,
            )
        except TimeoutError:
            self.audit.record(
                tool_name=tool_name,
                arguments=args,
                risk_level=risk.value,
                permission="allowed",
                confirmation=confirmation,
                status="timeout",
                duration_ms=elapsed_ms(),
                error=f"Timed out after {tool.timeout_seconds}s.",
            )
            return ToolResult(
                ok=False, error=f"Tool timed out after {tool.timeout_seconds}s."
            )
        result.duration_ms = elapsed_ms()

        # 9-10. Audit + return.
        self.audit.record(
            tool_name=tool_name,
            arguments=args,
            risk_level=risk.value,
            permission="allowed",
            confirmation=confirmation,
            status="completed" if result.ok else "failed",
            duration_ms=result.duration_ms,
            error=result.error,
        )
        return result
