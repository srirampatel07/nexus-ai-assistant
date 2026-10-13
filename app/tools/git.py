"""Git tools (Jarvis institute parity) — safe, confined, audited."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.exceptions import ToolDenied
from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


def _git() -> str | None:
    return shutil.which("git")


class GitBase(Tool):
    category = "git"
    timeout_seconds = 30.0

    def __init__(self, allowed_roots: list[str]) -> None:
        self.allowed_roots = [Path(r).resolve() for r in allowed_roots]

    def _resolve(self, raw: str | None) -> Path:
        cand = Path(raw).expanduser().resolve() if raw else self.allowed_roots[0]
        for root in self.allowed_roots:
            if cand == root or root in cand.parents:
                return cand
        raise ToolDenied(f"Path '{raw}' is outside allowed roots.")

    def _run(self, args: list[str], cwd: Path) -> ToolResult:
        exe = _git()
        if not exe:
            return ToolResult(ok=False, error="git is not installed.")
        try:
            proc = subprocess.run([exe, *args], cwd=str(cwd), capture_output=True,
                                  text=True, timeout=25, shell=False)
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error="git timed out.")
        except OSError as exc:
            return ToolResult(ok=False, error=f"Cannot run git: {exc}")
        out = (proc.stdout or "")[:20000]
        err = (proc.stderr or "")[:4000]
        if proc.returncode != 0:
            return ToolResult(ok=False, error=err or f"git exit {proc.returncode}",
                              output={"stdout": out})
        return ToolResult(ok=True, output={"stdout": out})


class GitStatusInput(BaseModel):
    path: str | None = Field(default=None, max_length=500)


class GitStatusTool(GitBase):
    name = "git.status"
    description = "git status --short --branch for a repo inside allowed roots."
    risk_level = PermissionLevel.SAFE
    input_model = GitStatusInput

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, GitStatusInput)
        try:
            cwd = self._resolve(validated.path)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        return self._run(["status", "--short", "--branch"], cwd)


class GitLogInput(BaseModel):
    path: str | None = Field(default=None, max_length=500)
    limit: int = Field(default=10, ge=1, le=50)


class GitLogTool(GitBase):
    name = "git.log"
    description = "Recent commits (oneline) for a repo inside allowed roots."
    risk_level = PermissionLevel.SAFE
    input_model = GitLogInput

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, GitLogInput)
        try:
            cwd = self._resolve(validated.path)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        return self._run(["log", f"--max-count={validated.limit}", "--oneline", "--decorate"], cwd)


class GitDiffInput(BaseModel):
    path: str | None = Field(default=None, max_length=500)


class GitDiffTool(GitBase):
    name = "git.diff"
    description = "Working-tree diff (stat + patch, truncated) for a repo."
    risk_level = PermissionLevel.LOW
    input_model = GitDiffInput

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, GitDiffInput)
        try:
            cwd = self._resolve(validated.path)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        return self._run(["diff", "--stat", "-p", "--", "."], cwd)


class GitCommitInput(BaseModel):
    path: str | None = Field(default=None, max_length=500)
    message: str = Field(min_length=1, max_length=500)


class GitCommitTool(GitBase):
    name = "git.commit"
    description = "Stage all + commit with message (needs approval). Push stays in terminal.execute."
    risk_level = PermissionLevel.HIGH
    requires_confirmation = True
    input_model = GitCommitInput

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, GitCommitInput)
        try:
            cwd = self._resolve(validated.path)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        exe = _git()
        if not exe:
            return ToolResult(ok=False, error="git is not installed.")
        try:
            subprocess.run([exe, "add", "-A"], cwd=str(cwd), capture_output=True,
                           timeout=20, shell=False, check=False)
            proc = subprocess.run([exe, "commit", "-m", validated.message],
                                  cwd=str(cwd), capture_output=True, text=True,
                                  timeout=25, shell=False)
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error="git commit timed out.")
        except OSError as exc:
            return ToolResult(ok=False, error=f"Cannot run git: {exc}")
        if proc.returncode != 0:
            return ToolResult(ok=False, error=(proc.stderr or proc.stdout or "")[:2000])
        return ToolResult(ok=True, output={"stdout": (proc.stdout or "")[:8000]})
