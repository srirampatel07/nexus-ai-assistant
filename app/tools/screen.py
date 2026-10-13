"""Screen tools: capture + Show-Me pointer (Jarvis getjarvis.eu parity)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


class _NoInput(BaseModel):
    pass


class PointInput(BaseModel):
    x: int = Field(ge=0, le=10000)
    y: int = Field(ge=0, le=10000)
    label: str = Field(default="", max_length=80)
    duration_ms: int = Field(default=2500, ge=500, le=10000)


class ScreenCaptureTool(Tool):
    name = "screen.capture"
    description = "Capture the primary monitor to data/screenshots/ (explicit only). Returns path + size for /vision."
    category = "screen"
    risk_level = PermissionLevel.LOW
    input_model = _NoInput
    timeout_seconds = 30.0

    def __init__(self, privacy_mode: str = "guarded", project_root: str = ".") -> None:
        self.privacy_mode = (privacy_mode or "guarded").lower()
        self.project_root = project_root

    def execute(self, validated: BaseModel) -> ToolResult:
        if self.privacy_mode == "strict":
            return ToolResult(ok=False, error="Screen capture disabled in STRICT privacy mode.")
        try:
            from app.screen.capture import capture_screen
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
        try:
            shot = capture_screen(project_root=self.project_root)
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(ok=True, output={
            "path": shot.path, "width": shot.width,
            "height": shot.height, "bytes": shot.bytes,
            "hint": "Use filesystem.read? No — pass path to /vision or vision tool.",
        })


class ScreenPointTool(Tool):
    name = "screen.point"
    description = "Draw a temporary on-screen circle at (x,y) with optional label (Show-Me). Coordinates in pixels."
    category = "screen"
    risk_level = PermissionLevel.LOW
    input_model = PointInput
    timeout_seconds = 15.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, PointInput)
        try:
            from app.screen.overlay import point_on_screen
        except Exception as exc:
            return ToolResult(ok=False, error=str(exc))
        out = point_on_screen(validated.x, validated.y,
                              label=validated.label, duration_ms=validated.duration_ms)
        if out.get("shown"):
            return ToolResult(ok=True, output=out)
        return ToolResult(ok=False, error=str(out.get("error", "overlay unavailable")))
