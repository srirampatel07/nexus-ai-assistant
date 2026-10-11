"""Default tool wiring (Phase 2).

`build_default_registry(settings)` registers every built-in tool with its
settings-derived confinement (allowed roots/apps, timeouts, size caps).
The assistant and CLI both use this factory — one place to add new tools.
"""

from __future__ import annotations

from app.core.config import Settings
from app.security.audit import AuditLogger
from app.tools.applications import ApplicationsLaunchTool
from app.tools.brave import BraveOpenTool
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
from app.tools.web import WebFetchTool, WebSearchTool

__all__ = ["build_default_registry", "build_default_executor"]


def build_default_registry(settings: Settings) -> ToolRegistry:
    """Create a registry with all built-in Phase 2 tools."""
    registry = ToolRegistry()
    fs_ctx = FilesystemContext(
        allowed_roots=settings.nexus_allowed_roots,
        max_read_bytes=settings.filesystem_max_read_bytes,
        max_write_bytes=settings.filesystem_max_write_bytes,
    )
    from app.web.search_session import SearchSession

    # Current-search session shared by web.search (records real results) and
    # brave.open (resolves "result N" references to exact real URLs).
    search_session = SearchSession()
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
        # Visible desktop browser launcher: any valid public http(s) URL or
        # current-search reference. Independent of WEB_ENABLED (desktop
        # capability, like applications.launch) and separate from the
        # internal reader sessions below.
        BraveOpenTool(session=search_session),
    ]
    if settings.web_enabled:
        from app.browser.session import BrowserManager
        from app.tools.browser import (
            BrowserBackTool,
            BrowserClickTool,
            BrowserCloseTool,
            BrowserLinksTool,
            BrowserNavigateTool,
            BrowserReadTool,
        )
        from app.web.client import WebContext

        web_ctx = WebContext.from_settings(settings)
        browser_mgr = BrowserManager(web_ctx)
        tools.extend(
            [
                WebSearchTool(web_ctx, session=search_session),
                WebFetchTool(web_ctx),
                BrowserNavigateTool(web_ctx, browser_mgr),
                BrowserReadTool(browser_mgr),
                BrowserLinksTool(browser_mgr),
                BrowserClickTool(web_ctx, browser_mgr),
                BrowserBackTool(browser_mgr),
                BrowserCloseTool(browser_mgr),
            ]
        )
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
