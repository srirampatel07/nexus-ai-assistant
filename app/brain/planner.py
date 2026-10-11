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
    """Diagnostic planner: always responds directly.

    Tool selection is NOT done here — the agent loop in
    `NexusAssistant._achat_with_tools` offers the registry's tool
    definitions to the model and dispatches returned calls through the
    executor whenever tools are enabled. This plan record is diagnostic
    only and never blocks the tool path.
    """

    def plan(self, user_text: str, history_count: int = 0) -> Plan:
        text = user_text.strip()
        if not text:
            return Plan(
                action=PlanAction.RESPOND_DIRECTLY,
                reasoning="Empty input; ask the user what they need.",
            )
        return Plan(
            action=PlanAction.RESPOND_DIRECTLY,
            reasoning=(
                "Conversational reply; tool selection is handled dynamically "
                "by the agent loop when a registry/executor is configured."
            ),
        )
