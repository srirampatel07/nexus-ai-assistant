"""Terminal tool tests (Phase 2). Only safe commands are executed."""

import shutil

import pytest

from app.core.exceptions import ToolDenied
from app.security.policy import PermissionLevel
from app.tools.terminal import TerminalExecuteTool, TerminalInput, classify

needs_powershell = pytest.mark.skipif(
    shutil.which("powershell") is None and shutil.which("powershell.exe") is None,
    reason="PowerShell not available",
)


@pytest.fixture
def tool(tmp_path):
    return TerminalExecuteTool(allowed_roots=[str(tmp_path)])


def test_classify_safe():
    verdict, _ = classify("python --version")
    assert verdict == "safe"
    assert classify("Get-ChildItem")[0] == "safe"
    assert classify("whoami")[0] == "safe"


def test_classify_confirm():
    assert classify("pip install requests")[0] == "confirm"
    assert classify("git push origin main")[0] == "confirm"
    assert classify("Remove-Item old.txt")[0] == "confirm"


def test_classify_blocked():
    assert classify("format C:")[0] == "blocked"
    assert classify("shutdown /s")[0] == "blocked"
    assert classify("Get-ChildItem Env:")[0] == "blocked"
    assert classify("Get-Content .env")[0] == "blocked"


def test_assess_maps_verdict_to_risk(tool, tmp_path):
    assert tool.assess(TerminalInput(command="whoami")) == PermissionLevel.LOW
    assert tool.assess(TerminalInput(command="pip install x")) == PermissionLevel.MEDIUM
    with pytest.raises(ToolDenied):
        tool.assess(TerminalInput(command="format C:"))


@needs_powershell
def test_execute_echo(tool, tmp_path):
    result = tool.execute(
        TerminalInput(command="Write-Output hello-nexus", working_directory=str(tmp_path))
    )
    assert result.ok
    assert result.output["exit_code"] == 0
    assert "hello-nexus" in result.output["stdout"]


@needs_powershell
def test_execute_failure_reports_exit_code(tool, tmp_path):
    result = tool.execute(
        TerminalInput(command="exit 7", working_directory=str(tmp_path))
    )
    assert not result.ok
    assert result.output["exit_code"] == 7


@needs_powershell
def test_execute_timeout(tool, tmp_path):
    tool.default_timeout = 60
    result = tool.execute(
        TerminalInput(
            command="Start-Sleep -Seconds 10",
            working_directory=str(tmp_path),
            timeout_seconds=1,
        )
    )
    assert not result.ok
    assert "timed out" in result.error


def test_execute_rejects_bad_cwd(tool, tmp_path):
    result = tool.execute(
        TerminalInput(command="whoami", working_directory="C:\\Windows")
    )
    assert not result.ok


def test_execute_truncates_output(tmp_path):
    small = TerminalExecuteTool(allowed_roots=[str(tmp_path)], max_output_chars=10)
    if small.shell is None:
        pytest.skip("PowerShell not available")
    result = small.execute(
        TerminalInput(
            command="Write-Output 0123456789ABCDEF", working_directory=str(tmp_path)
        )
    )
    assert result.ok
    assert result.output["stdout_truncated"] is True
    assert len(result.output["stdout"]) == 10
