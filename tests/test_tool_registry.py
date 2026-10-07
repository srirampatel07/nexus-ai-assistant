"""Tool registry + permission policy + confirmation tests (Phase 2)."""

import pytest

from app.core.exceptions import ToolNotAvailable
from app.security.confirmation import (
    ConfirmationRequest,
    auto_approve,
    auto_deny,
    format_request,
)
from app.security.policy import PermissionLevel, decide
from app.tools.base import Tool, ToolResult
from app.tools.permissions import needs_approval, permission_for
from app.tools.registry import ToolRegistry
from app.tools.system import SystemInfoTool


def _registry_with_system() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(SystemInfoTool())
    return reg


def test_register_and_lookup():
    reg = _registry_with_system()
    assert reg.has("system.info")
    assert isinstance(reg.get("system.info"), SystemInfoTool)
    assert len(reg) == 1


def test_duplicate_registration_rejected():
    reg = _registry_with_system()
    with pytest.raises(ValueError):
        reg.register(SystemInfoTool())


def test_unnamed_tool_rejected():
    class Nameless(Tool):
        input_model = SystemInfoTool.input_model

        def execute(self, validated):
            return ToolResult(ok=True)

    with pytest.raises(ValueError):
        _registry_with_system().register(Nameless())


def test_unknown_tool_raises():
    with pytest.raises(ToolNotAvailable):
        _registry_with_system().get("nope.missing")


def test_unregister():
    reg = _registry_with_system()
    assert reg.unregister("system.info") is True
    assert reg.unregister("system.info") is False
    assert not reg.has("system.info")


def test_tool_definitions_metadata():
    defs = _registry_with_system().tool_definitions()
    assert len(defs) == 1
    assert defs[0].name == "system.info"
    assert defs[0].description
    assert defs[0].parameters.get("type") == "object"


def test_list_tools_returns_registered():
    assert [t.name for t in _registry_with_system().list_tools()] == ["system.info"]


@pytest.mark.parametrize(
    "level,allowed,confirm",
    [
        (PermissionLevel.SAFE, True, False),
        (PermissionLevel.LOW, True, False),
        (PermissionLevel.MEDIUM, True, True),
        (PermissionLevel.HIGH, True, True),
        (PermissionLevel.BLOCKED, False, False),
    ],
)
def test_policy_decisions(level, allowed, confirm):
    decision = decide(level)
    assert decision.allowed is allowed
    assert decision.requires_confirmation is confirm


def test_permission_facade_matches_policy():
    tool = SystemInfoTool()
    decision = permission_for(tool)
    assert decision.allowed and not decision.requires_confirmation
    assert needs_approval(tool) is False
    assert needs_approval(tool, PermissionLevel.HIGH) is True


@pytest.mark.asyncio
async def test_auto_deny_denies():
    req = ConfirmationRequest(tool_name="x", description="y", risk_level="HIGH")
    assert await auto_deny(req) is False


@pytest.mark.asyncio
async def test_auto_approve_approves():
    req = ConfirmationRequest(tool_name="x", description="y", risk_level="HIGH")
    assert await auto_approve(req) is True


def test_format_request_is_transparent():
    req = ConfirmationRequest(
        tool_name="terminal.execute",
        description="Run a command",
        risk_level="MEDIUM",
        arguments_summary={"command": "dir"},
    )
    text = format_request(req)
    assert "terminal.execute" in text
    assert "MEDIUM" in text
    assert "dir" in text
    assert "Allow? [y/N]" in text
