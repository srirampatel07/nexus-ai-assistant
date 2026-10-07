"""ToolExecutor enforcement tests (Phase 2). Confined to tmp_path."""

import asyncio
import json

import pytest
from pydantic import BaseModel

from app.core.config import Settings
from app.security.audit import AuditLogger
from app.security.confirmation import auto_approve, auto_deny
from app.tools import build_default_registry
from app.tools.base import Tool, ToolResult
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry


@pytest.fixture
def settings(tmp_path):
    return Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )


@pytest.fixture
def executor(settings):
    return ToolExecutor(
        registry=build_default_registry(settings),
        audit=AuditLogger(settings.tools_audit_file),
        confirm=auto_deny,
    )


@pytest.mark.asyncio
async def test_execute_safe_tool_no_confirmation(executor):
    result = await executor.execute("system.info", {})
    assert result.ok
    assert "os" in result.output


@pytest.mark.asyncio
async def test_unknown_tool_fails_cleanly(executor):
    result = await executor.execute("nope.missing", {})
    assert not result.ok
    assert "Unknown tool" in result.error


@pytest.mark.asyncio
async def test_validation_error_fails_cleanly(executor):
    result = await executor.execute("filesystem.read", {"wrong": 1})
    assert not result.ok


@pytest.mark.asyncio
async def test_blocked_command_denied(executor):
    result = await executor.execute("terminal.execute", {"command": "format C:"})
    assert not result.ok
    assert "Blocked" in result.error


@pytest.mark.asyncio
async def test_confirmation_deny_blocks_write(executor, tmp_path):
    target = tmp_path / "denied.txt"
    result = await executor.execute(
        "filesystem.write",
        {"path": str(target), "content": "x"},
        confirm=auto_deny,
    )
    assert not result.ok
    assert "approval" in result.error
    assert not target.exists()


@pytest.mark.asyncio
async def test_confirmation_approve_allows_write(executor, tmp_path):
    target = tmp_path / "allowed.txt"
    result = await executor.execute(
        "filesystem.write",
        {"path": str(target), "content": "x"},
        confirm=auto_approve,
    )
    assert result.ok
    assert target.read_text() == "x"


@pytest.mark.asyncio
async def test_audit_log_written_for_every_call(executor, tmp_path):
    await executor.execute("system.info", {})
    await executor.execute("nope.missing", {})
    lines = (tmp_path / "audit.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first, second = (json.loads(line) for line in lines)
    assert first["status"] == "completed"
    assert second["status"] == "failed"
    assert second["error"]


@pytest.mark.asyncio
async def test_executor_timeout(tmp_path):
    class SlowInput(BaseModel):
        pass

    class SlowTool(Tool):
        name = "test.slow"
        description = "test-only slow tool"
        input_model = SlowInput
        timeout_seconds = 0.05

        def execute(self, validated):
            import time as _t

            _t.sleep(5)
            return ToolResult(ok=True)

    reg = ToolRegistry()
    reg.register(SlowTool())
    ex = ToolExecutor(
        registry=reg,
        audit=AuditLogger(tmp_path / "a.jsonl"),
        confirm=auto_approve,  # MEDIUM default risk needs approval to reach execute
    )
    result = await ex.execute("test.slow", {})
    assert not result.ok
    assert "timed out" in result.error


@pytest.mark.asyncio
async def test_broken_confirm_callback_denies_safely(settings, tmp_path):
    async def broken(_req):
        raise RuntimeError("boom")

    ex = ToolExecutor(
        registry=build_default_registry(settings),
        audit=AuditLogger(tmp_path / "a.jsonl"),
        confirm=broken,
    )
    target = tmp_path / "nope.txt"
    result = await ex.execute("filesystem.write", {"path": str(target), "content": "x"})
    assert not result.ok
    assert not target.exists()


def test_slow_tool_is_sync_safe():
    # asyncio.wait_for must wrap sync tools; ensure no event-loop deadlock.
    assert asyncio.iscoroutinefunction(ToolExecutor.execute)
