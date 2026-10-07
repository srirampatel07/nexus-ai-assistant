"""Default tool wiring (Phase 2).

`build_default_registry(settings)` registers every built-in tool with its
settings-derived confinement (allowed roots/apps, timeouts, size caps).
The assistant and CLI both use this factory — one place to add new tools.
"""

from __future__ import annotations

from app.core.config import Settings
from app.security.audit import AuditLogger
from app.tools.applications import ApplicationsLaunchTool
from app.tools.executor import ToolExecutor
from app.tools.filesystem import (
    FilesystemContext,
    FilesystemCreateDirectoryTool,
    FilesystemDeleteTool,
    FilesystemListTool,
    FilesystemReadTool,
    FilesystemWriteTool,
)
from app.tools.registry import ToolRegistry
from app.tools.system import (
    SystemCpuTool,
    SystemInfoTool,
    SystemMemoryTool,
    SystemPlatformTool,
)
from app.tools.terminal import TerminalExecuteTool

__all__ = ["build_default_registry", "build_default_executor"]


def build_default_registry(settings: Settings) -> ToolRegistry:
    """Create a registry with all built-in Phase 2 tools."""
    registry = ToolRegistry()
    fs_ctx = FilesystemContext(
        allowed_roots=settings.nexus_allowed_roots,
        max_read_bytes=settings.filesystem_max_read_bytes,
        max_write_bytes=settings.filesystem_max_write_bytes,
    )
    tools = [
        SystemInfoTool(),
        SystemCpuTool(),
        SystemMemoryTool(),
        SystemPlatformTool(),
        FilesystemListTool(fs_ctx),
        FilesystemReadTool(fs_ctx),
        FilesystemWriteTool(fs_ctx),
        FilesystemCreateDirectoryTool(fs_ctx),
        FilesystemDeleteTool(fs_ctx),
        ApplicationsLaunchTool(settings.nexus_allowed_apps),
        TerminalExecuteTool(
            allowed_roots=settings.nexus_allowed_roots,
            default_timeout=settings.terminal_default_timeout_seconds,
            max_output_chars=settings.terminal_max_output_chars,
        ),
    ]
    for tool in tools:
        registry.register(tool)
    return registry


def build_default_executor(
    settings: Settings, registry: ToolRegistry | None = None
) -> ToolExecutor:
    """Create the enforcing executor with JSON Lines audit logging."""
    return ToolExecutor(
        registry=registry or build_default_registry(settings),
        audit=AuditLogger(settings.tools_audit_file),
    )
