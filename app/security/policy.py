"""Permission model for NEXUS tools (Phase 2).

The permission system is separate from the tools themselves: tools declare
a risk level, and this module decides what that level means. The
ToolExecutor is the only component that enforces these decisions.

Levels:
    SAFE     - read-only, no side effects (system info, allowed dir listing)
    LOW      - minor side effects (launch allowlisted app, read a file)
    MEDIUM   - normal side effects (write file, run non-destructive command)
    HIGH     - dangerous side effects (delete, destructive commands, config)
    BLOCKED  - never execute automatically (always denied)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PermissionLevel(str, Enum):
    SAFE = "SAFE"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    requires_confirmation: bool
    reason: str


def decide(level: PermissionLevel) -> PermissionDecision:
    """Map a risk level to an enforcement decision."""
    if level == PermissionLevel.SAFE:
        return PermissionDecision(True, False, "SAFE operations need no approval.")
    if level == PermissionLevel.LOW:
        return PermissionDecision(True, False, "LOW operations need no approval.")
    if level == PermissionLevel.MEDIUM:
        return PermissionDecision(True, True, "MEDIUM operations require confirmation.")
    if level == PermissionLevel.HIGH:
        return PermissionDecision(True, True, "HIGH operations require confirmation.")
    return PermissionDecision(False, False, "BLOCKED operations are always denied.")
