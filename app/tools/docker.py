"""Docker tools (Jarvis institute parity) — read-mostly, mutating needs approval."""

from __future__ import annotations

import shutil
import subprocess

from pydantic import BaseModel, Field

from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


def _docker() -> str | None:
    return shutil.which("docker")


class _NoInput(BaseModel):
    pass


class DockerNameInput(BaseModel):
    name: str = Field(min_length=1, max_length=128)


class DockerLogsInput(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    tail: int = Field(default=100, ge=1, le=1000)


def _run(args: list[str]) -> ToolResult:
    exe = _docker()
    if not exe:
        return ToolResult(ok=False, error="docker is not available.")
    try:
        proc = subprocess.run([exe, *args], capture_output=True, text=True,
                              timeout=25, shell=False)
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, error="docker timed out.")
    except OSError as exc:
        return ToolResult(ok=False, error=f"Cannot run docker: {exc}")
    out = (proc.stdout or "")[:20000]
    if proc.returncode != 0:
        return ToolResult(ok=False, error=(proc.stderr or "")[:2000] or "docker failed",
                          output={"stdout": out})
    return ToolResult(ok=True, output={"stdout": out})


class DockerPsTool(Tool):
    name = "docker.ps"
    description = "List containers (docker ps --all)."
    category = "docker"
    risk_level = PermissionLevel.SAFE
    input_model = _NoInput
    timeout_seconds = 30.0

    def execute(self, validated: BaseModel) -> ToolResult:
        return _run(["ps", "--all", "--format",
                     "table {{.Names}}\t{{.Image}}\t{{.Status}}"])


class DockerImagesTool(Tool):
    name = "docker.images"
    description = "List local images."
    category = "docker"
    risk_level = PermissionLevel.SAFE
    input_model = _NoInput
    timeout_seconds = 30.0

    def execute(self, validated: BaseModel) -> ToolResult:
        return _run(["images", "--format", "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"])


class DockerLogsTool(Tool):
    name = "docker.logs"
    description = "Tail logs for a container."
    category = "docker"
    risk_level = PermissionLevel.LOW
    input_model = DockerLogsInput
    timeout_seconds = 30.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, DockerLogsInput)
        return _run(["logs", "--tail", str(validated.tail), validated.name])


class DockerStartTool(Tool):
    name = "docker.start"
    description = "Start a container (needs approval)."
    category = "docker"
    risk_level = PermissionLevel.MEDIUM
    requires_confirmation = True
    input_model = DockerNameInput
    timeout_seconds = 30.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, DockerNameInput)
        return _run(["start", validated.name])


class DockerStopTool(Tool):
    name = "docker.stop"
    description = "Stop a container (needs approval)."
    category = "docker"
    risk_level = PermissionLevel.MEDIUM
    requires_confirmation = True
    input_model = DockerNameInput
    timeout_seconds = 30.0

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, DockerNameInput)
        return _run(["stop", validated.name])
