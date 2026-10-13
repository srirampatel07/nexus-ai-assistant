"""Routine tools: schedule / list / run-due / delete (institute parity)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.routines.store import RoutineStore
from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


class RoutineScheduleInput(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    prompt: str = Field(min_length=1, max_length=4000)
    interval_seconds: int = Field(default=86400, ge=60, le=2592000)


class RoutineDeleteInput(BaseModel):
    routine_id: int = Field(ge=1)


class _NoInput(BaseModel):
    pass


class RoutinesScheduleTool(Tool):
    name = "routines.schedule"
    description = "Schedule a recurring plain-language task (min 60s interval). Returns id + next run."
    category = "routines"
    risk_level = PermissionLevel.MEDIUM
    requires_confirmation = True
    input_model = RoutineScheduleInput
    timeout_seconds = 15.0

    def __init__(self, db_path: str = "data/routines.db") -> None:
        self.db_path = db_path

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, RoutineScheduleInput)
        try:
            routine = RoutineStore(self.db_path).add(
                validated.name, validated.prompt, validated.interval_seconds)
        except ValueError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(ok=True, output={"id": routine.id, "name": routine.name,
                                           "next_run_epoch": routine.next_run_epoch})


class RoutinesListTool(Tool):
    name = "routines.list"
    description = "List scheduled routines with next-run times and last status."
    category = "routines"
    risk_level = PermissionLevel.SAFE
    input_model = _NoInput
    timeout_seconds = 15.0

    def __init__(self, db_path: str = "data/routines.db") -> None:
        self.db_path = db_path

    def execute(self, validated: BaseModel) -> ToolResult:
        routines = RoutineStore(self.db_path).list()
        return ToolResult(ok=True, output={"routines": [
            {"id": r.id, "name": r.name, "prompt": r.prompt[:200],
             "interval_seconds": r.interval_seconds, "next_run_epoch": r.next_run_epoch,
             "enabled": r.enabled, "last_status": r.last_status} for r in routines]})


class RoutinesDueTool(Tool):
    name = "routines.due"
    description = "List overdue routines (catch-up after downtime)."
    category = "routines"
    risk_level = PermissionLevel.SAFE
    input_model = _NoInput
    timeout_seconds = 15.0

    def __init__(self, db_path: str = "data/routines.db") -> None:
        self.db_path = db_path

    def execute(self, validated: BaseModel) -> ToolResult:
        due = RoutineStore(self.db_path).due()
        return ToolResult(ok=True, output={"due": [
            {"id": r.id, "name": r.name, "prompt": r.prompt} for r in due]})


class RoutinesDeleteTool(Tool):
    name = "routines.delete"
    description = "Delete a scheduled routine by id."
    category = "routines"
    risk_level = PermissionLevel.MEDIUM
    requires_confirmation = True
    input_model = RoutineDeleteInput
    timeout_seconds = 15.0

    def __init__(self, db_path: str = "data/routines.db") -> None:
        self.db_path = db_path

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, RoutineDeleteInput)
        ok = RoutineStore(self.db_path).remove(validated.routine_id)
        if not ok:
            return ToolResult(ok=False, error=f"No routine #{validated.routine_id}.")
        return ToolResult(ok=True, output={"deleted": validated.routine_id})
