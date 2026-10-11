"""Browser exception hierarchy (Phase 6).

All browser errors derive from `BrowserError` (a `NexusError`). Messages
never carry page contents, credentials, or response bodies — only safe
diagnostics (URLs without query strings where avoidable, link ids).
"""

from __future__ import annotations

from app.core.exceptions import NexusError


class BrowserError(NexusError):
    """Base class for all browser-session errors."""


class BrowserUnavailableError(BrowserError):
    """Browser capability cannot run (disabled or no session)."""


class NoPageError(BrowserError):
    """No page is loaded in this session (navigate first)."""


class ElementNotFoundError(BrowserError):
    """No supported element matches the requested reference."""
