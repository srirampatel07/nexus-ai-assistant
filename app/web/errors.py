"""Web exception hierarchy (Phase 6).

All web errors derive from `WebError` (a `NexusError`). Messages never
carry API keys, credentials, response bodies, or page contents — only safe
diagnostics (host, status code, sizes).
"""

from __future__ import annotations

from app.core.exceptions import NexusError, ToolDenied


class WebError(NexusError):
    """Base class for all web-interface errors."""


class WebUnavailableError(WebError):
    """Web capability cannot run (disabled or no usable backend)."""


class UnsafeDestinationError(ToolDenied):
    """A URL was refused by SSRF protection (also a ToolDenied so the
    executor audits it as denied:policy when raised from assess())."""
