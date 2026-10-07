"""Application launcher tool (Phase 2).

Only allowlisted applications may be launched — arbitrary executable paths
are rejected by default. This keeps `applications.launch` at LOW risk:
it can open helpers (notepad, explorer, terminals, editors) but cannot be
abused to run system-modifying binaries.
"""

from __future__ import annotations

import re
import shutil
import subprocess

from pydantic import BaseModel, Field

from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult

# Allowlist name -> executable resolved via PATH at launch time.
_DEFAULT_APP_MAP = {
    "notepad": "notepad.exe",
    "explorer": "explorer.exe",
    "powershell": "powershell.exe",
    "code": "code.cmd",
}

# Argument patterns that are never accepted (shell metachars / encoded payloads).
_BAD_ARG_RES = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"-EncodedCommand",
        r"(^|\s)-enc(\s|$)",
        r"[|;&`]",
        r"\$\(",
        r"DownloadString",
        r"Invoke-Expression",
    ]
]


class LaunchInput(BaseModel):
    application: str = Field(min_length=1, max_length=64)
    arguments: list[str] = Field(default_factory=list, max_length=8)


class ApplicationsLaunchTool(Tool):
    name = "applications.launch"
    description = "Launch an allowlisted application (notepad, explorer, powershell, code)."
    category = "applications"
    risk_level = PermissionLevel.LOW
    input_model = LaunchInput

    def __init__(self, allowed_apps: list[str]) -> None:
        self.allowed = {a.strip().lower() for a in allowed_apps if a.strip()}

    def _validate_args(self, args: list[str]) -> str | None:
        for arg in args:
            if len(arg) > 512:
                return f"Argument too long ({len(arg)} chars)."
            for rx in _BAD_ARG_RES:
                if rx.search(arg):
                    return f"Argument rejected by policy: {rx.pattern}"
        return None

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, LaunchInput)
        app = validated.application.strip().lower()
        if app not in self.allowed or app not in _DEFAULT_APP_MAP:
            return ToolResult(
                ok=False,
                error=f"Application '{validated.application}' is not allowlisted. "
                f"Allowed: {sorted(self.allowed)}.",
            )
        problem = self._validate_args(validated.arguments)
        if problem:
            return ToolResult(ok=False, error=problem)
        exe = shutil.which(_DEFAULT_APP_MAP[app])
        if not exe:
            return ToolResult(ok=False, error=f"Executable not found: {_DEFAULT_APP_MAP[app]}")
        try:
            proc = subprocess.Popen([exe, *validated.arguments], shell=False)  # noqa: S603 - allowlisted exe only
        except OSError as exc:
            return ToolResult(ok=False, error=f"Cannot launch application: {exc}")
        return ToolResult(
            ok=True, output={"application": app, "pid": proc.pid, "launched": True}
        )
