"""Tool-facing permission facade (Phase 2).

Thin bridge between tools/executor and the security policy in
`app.security.policy` (single source of truth — no duplicated rules here).
"""

from __future__ import annotations

from app.security.policy import PermissionDecision, PermissionLevel, decide
from app.tools.base import Tool


def permission_for(tool: Tool, effective_risk: PermissionLevel | None = None) -> PermissionDecision:
    """Decide enforcement for a tool (optionally with its dynamic risk)."""
    return decide(effective_risk or tool.risk_level)


def needs_approval(tool: Tool, effective_risk: PermissionLevel | None = None) -> bool:
    """True when executing this tool must ask the user first."""
    decision = permission_for(tool, effective_risk)
    return (not decision.allowed) or decision.requires_confirmation or tool.requires_confirmation
