"""Safe terminal tool for Windows (Phase 2).

`terminal.execute` runs ONE PowerShell command per call via
`powershell.exe -NoProfile -NonInteractive -Command <text>` with
`shell=False` (no extra cmd layer), a hard timeout, captured/truncated
output, and a validated working directory confined to allowed roots.

Command policy (best-effort regex classification, NOT a sandbox):
    SAFE    - informational commands run immediately (LOW risk)
    CONFIRM - installers / mutating commands need approval (MEDIUM risk)
    BLOCKED - destructive / credential-theft patterns are always refused

LIMITATIONS (documented honestly): pattern filtering cannot catch
obfuscated or novel destructive commands. The real safety layers are
allowed-root confinement, mandatory confirmation for MEDIUM+ risk, audit
logging, and output secret-redaction. Treat unknown commands with care.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, Field

from app.core.exceptions import ToolDenied
from app.security.audit import sanitize
from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult

_BLOCKED_PATTERNS = [
    r"\bformat\s+[a-z]:",  # disk formatting
    r"\bdiskpart\b",
    r"\bshutdown(\.exe)?\b",
    r"\brestart-computer\b",
    r"\bstop-computer\b",
    r"vssadmin\s+delete",  # shadow copy deletion (ransomware pattern)
    r"\bbcdedit\b",
    r"cipher\s+/w",  # secure wipe
    r"wmic\s+shadowcopy\s+delete",
    r"wbadmin\s+delete",
    r"reg\s+(delete|add)\b.*hklm",  # HKLM registry destruction
    r"mimikatz|sekurlsa|lsass\s*dump|ntdsutil",  # credential theft
    r"Set-MpPreference|Add-MpPreference.*-Exclusion",  # Defender tampering
    r"amsi.*bypass|bypass.*amsi",
    r"IEX\s*\(.*(downloadstring|http)|Invoke-Expression.*http|\|\s*iex\b",  # remote code exec
    r"net\s+user\s+\S+.*\s/add|net\s+localgroup.*\s/add",  # account creation
    r"takeown\s+.*c:\\windows|icacls\s+c:\\windows",  # system ACL takeover
    r"Get-ChildItem\s+Env:|gci\s+env:|dir\s+env:|\$env:|printenv",  # env dumping
    r"Get-Content\s+.*\.env\b",  # secret file exfiltration
    # Recursive force-delete aimed at protected locations:
    r"Remove-Item\b.*-Recurse.*(c:\\windows|c:\\program files|c:\\programdata|~[/\\]?$)",
]

_CONFIRM_PATTERNS = [
    r"pip\s+install|npm\s+(install|i\b)|winget\s+install|choco\s+install|scoop\s+install",
    r"\bgit\s+(push|reset|clean|rebase|filter-branch)\b",
    r"\b(Remove-Item|del|erase|rmdir|rd)\b",
    r"docker\s+(rm|rmi|system\s+prune|volume\s+rm)",
    r"\breg\s+(add|delete|import)\b",
    r"Set-ExecutionPolicy|-ExecutionPolicy\s+Bypass",
    r"Start-Service|Stop-Service|Restart-Service|New-Service",
    r"\bsc\s+(config|delete|stop|create)\b",
    r"\btaskkill\b|Stop-Process",
    r"schtasks\s+/create|New-ScheduledTask",
]

_BLOCKED_RES = [re.compile(p, re.IGNORECASE) for p in _BLOCKED_PATTERNS]
_CONFIRM_RES = [re.compile(p, re.IGNORECASE) for p in _CONFIRM_PATTERNS]


def classify(command: str) -> tuple[str, str]:
    """Classify a command: ('safe'|'confirm'|'blocked', reason)."""
    for rx in _BLOCKED_RES:
        if rx.search(command):
            return "blocked", f"Matches blocked pattern: {rx.pattern}"
    for rx in _CONFIRM_RES:
        if rx.search(command):
            return "confirm", f"Matches sensitive pattern: {rx.pattern}"
    return "safe", "No sensitive patterns detected."


class TerminalInput(BaseModel):
    command: str = Field(min_length=1, max_length=4000)
    working_directory: str | None = Field(default=None)
    timeout_seconds: float | None = Field(default=None, ge=1, le=300)


class TerminalExecuteTool(Tool):
    name = "terminal.execute"
    description = (
        "Run a single PowerShell command on Windows with timeout and "
        "captured output. Destructive patterns are blocked; installers and "
        "mutating commands need approval."
    )
    category = "terminal"
    risk_level = PermissionLevel.MEDIUM
    input_model = TerminalInput

    def __init__(
        self,
        allowed_roots: list[str],
        default_timeout: float = 30.0,
        max_output_chars: int = 65536,
    ) -> None:
        self.allowed_roots = [Path(r).resolve() for r in allowed_roots]
        self.default_timeout = default_timeout
        self.max_output_chars = max_output_chars
        self.shell = shutil.which("powershell") or shutil.which("powershell.exe")

    # -- dynamic risk ------------------------------------------------------
    def assess(self, validated: BaseModel) -> PermissionLevel:
        assert isinstance(validated, TerminalInput)
        verdict, reason = classify(validated.command)
        if verdict == "blocked":
            raise ToolDenied(f"Blocked command: {reason}")
        if verdict == "confirm":
            return PermissionLevel.MEDIUM
        return PermissionLevel.LOW

    # -- working directory -------------------------------------------------
    def _resolve_cwd(self, raw: str | None) -> Path:
        candidate = Path(raw).expanduser().resolve() if raw else self.allowed_roots[0]
        if not candidate.is_dir():
            raise ToolDenied(f"Working directory does not exist: {raw or candidate}")
        for root in self.allowed_roots:
            if candidate == root or root in candidate.parents:
                return candidate
        raise ToolDenied(f"Working directory '{raw}' is outside allowed roots.")

    # -- execution ---------------------------------------------------------
    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, TerminalInput)
        if not self.shell:
            return ToolResult(ok=False, error="PowerShell is not available on this system.")
        try:
            cwd = self._resolve_cwd(validated.working_directory)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        timeout = validated.timeout_seconds or self.default_timeout
        started = time.monotonic()
        try:
            proc = subprocess.run(
                [self.shell, "-NoProfile", "-NonInteractive", "-Command", validated.command],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(ok=False, error=f"Command timed out after {timeout}s.")
        except OSError as exc:
            return ToolResult(ok=False, error=f"Cannot start process: {exc}")
        duration_ms = int((time.monotonic() - started) * 1000)
        # Secrets must never flow back into AI responses: redact + truncate.
        stdout = sanitize(proc.stdout or "")
        stderr = sanitize(proc.stderr or "")
        assert isinstance(stdout, str) and isinstance(stderr, str)
        out_truncated = len(stdout) > self.max_output_chars
        err_truncated = len(stderr) > self.max_output_chars
        return ToolResult(
            ok=proc.returncode == 0,
            output={
                "exit_code": proc.returncode,
                "stdout": stdout[: self.max_output_chars],
                "stderr": stderr[: self.max_output_chars],
                "stdout_truncated": out_truncated,
                "stderr_truncated": err_truncated,
            },
            error=None if proc.returncode == 0 else f"Exit code {proc.returncode}.",
            duration_ms=duration_ms,
        )
