"""Application launcher + system tool tests (Phase 2). Never launches apps."""

import json
from unittest.mock import patch

from app.tools.applications import ApplicationsLaunchTool, LaunchInput
from app.tools.system import (
    SystemCpuTool,
    SystemInfoTool,
    SystemMemoryTool,
    SystemPlatformTool,
)


class _FakeProc:
    pid = 1234


def _tool():
    return ApplicationsLaunchTool(["notepad", "explorer", "powershell", "code"])


def test_launch_allowlisted_with_mock():
    with patch("subprocess.Popen", return_value=_FakeProc()) as popen, patch(
        "shutil.which", return_value="C:\\fake\\notepad.exe"
    ):
        result = _tool().execute(LaunchInput(application="notepad"))
    assert result.ok
    assert result.output["pid"] == 1234
    popen.assert_called_once()


def test_launch_case_insensitive():
    with patch("subprocess.Popen", return_value=_FakeProc()), patch(
        "shutil.which", return_value="x"
    ):
        assert _tool().execute(LaunchInput(application="NotePad")).ok


def test_launch_rejects_unknown_app():
    result = _tool().execute(LaunchInput(application="cmd"))
    assert not result.ok
    assert "allowlist" in result.error


def test_launch_rejects_arbitrary_path():
    result = _tool().execute(LaunchInput(application="C:\\Windows\\regedit.exe"))
    assert not result.ok


def test_launch_rejects_dangerous_args():
    with patch("subprocess.Popen", return_value=_FakeProc()), patch(
        "shutil.which", return_value="x"
    ):
        result = _tool().execute(
            LaunchInput(application="powershell", arguments=["-EncodedCommand", "ew=="])
        )
    assert not result.ok


def test_launch_missing_executable():
    with patch("shutil.which", return_value=None):
        result = _tool().execute(LaunchInput(application="code"))
    assert not result.ok


def test_system_tools_return_facts():
    for tool_cls in (SystemInfoTool, SystemCpuTool, SystemMemoryTool, SystemPlatformTool):
        result = tool_cls().execute(tool_cls.input_model())
        assert result.ok, tool_cls.name
        assert isinstance(result.output, dict) and result.output


def test_system_info_has_expected_keys():
    out = SystemInfoTool().execute(SystemInfoTool.input_model()).output
    for key in ("os", "python_version", "hostname", "cpu_count", "nexus_version"):
        assert key in out


def test_system_output_contains_no_secrets(monkeypatch):
    monkeypatch.setenv("AI_API_KEY", "gsk_test-secret-should-never-leak")
    blob = json.dumps(
        [t().execute(t.input_model()).output for t in (SystemInfoTool, SystemCpuTool)]
    )
    assert "gsk_test-secret-should-never-leak" not in blob
