"""NEXUS reader browser (Phase 6: document navigation sessions)."""

from __future__ import annotations

from app.browser.errors import (
    BrowserError,
    BrowserUnavailableError,
    ElementNotFoundError,
    NoPageError,
)
from app.browser.session import (
    DEFAULT_SESSION_ID,
    BrowserManager,
    ReaderPage,
    ReaderSession,
)

__all__ = [
    "DEFAULT_SESSION_ID",
    "BrowserError",
    "BrowserManager",
    "BrowserUnavailableError",
    "ElementNotFoundError",
    "NoPageError",
    "ReaderPage",
    "ReaderSession",
]
