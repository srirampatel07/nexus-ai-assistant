"""Planner (Phase 1: respond directly; tool planning arrives in Phase 2/9)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class PlanAction(str, Enum):
    RESPOND_DIRECTLY = "respond_directly"
    USE_TOOL = "use_tool"  # reserved for Phase 2


@dataclass
class Plan:
    action: PlanAction
    reasoning: str
    tool_name: Optional[str] = None


class Planner:
    """Phase 1 planner: always responds directly (no tools registered yet)."""

    def plan(self, user_text: str, history_count: int = 0) -> Plan:
        text = user_text.strip()
        if not text:
            return Plan(
                action=PlanAction.RESPOND_DIRECTLY,
                reasoning="Empty input; ask the user what they need.",
            )
        return Plan(
            action=PlanAction.RESPOND_DIRECTLY,
            reasoning="Phase 1 has no tools; answer conversationally.",
        )
