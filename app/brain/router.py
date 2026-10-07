"""Tool router stub (Phase 1).

The real plugin-style registry arrives in Phase 2. This stub keeps the
assistant core's orchestration path stable: plan -> router -> execution.
"""

from __future__ import annotations

from app.brain.planner import Plan, PlanAction
from app.core.exceptions import ToolNotAvailable


class ToolRouter:
    """Phase 1 router: no tools registered yet."""

    def list_tools(self) -> list[dict]:
        return []

    def route(self, plan: Plan) -> None:
        if plan.action == PlanAction.USE_TOOL:
            raise ToolNotAvailable(
                f"Tool '{plan.tool_name}' is not available in Phase 1."
            )
