"""NEXUS web interface (Phase 6: search + retrieval + browser).

`app.web` holds the HTTP layer (SSRF guard, extraction, search/fetch
client); `app.browser` holds stateful reader sessions. Tools in
`app.tools.web` / `app.tools.browser` expose both through the standard
tool pipeline — no parallel execution architecture.
"""

from __future__ import annotations

from app.web.client import (
    FetchedPage,
    SearchResult,
    SearchResults,
    WebContext,
    fetch_url,
    search_ddg,
)
from app.web.errors import UnsafeDestinationError, WebError, WebUnavailableError
from app.web.extract import (
    PageLink,
    PageText,
    SearchHit,
    extract_readable,
    extract_search_hits,
)
from app.web.ssrf import check_url, default_resolver

__all__ = [
    "FetchedPage",
    "PageLink",
    "PageText",
    "SearchHit",
    "SearchResult",
    "SearchResults",
    "UnsafeDestinationError",
    "WebContext",
    "WebError",
    "WebUnavailableError",
    "check_url",
    "default_resolver",
    "extract_readable",
    "extract_search_hits",
    "fetch_url",
    "search_ddg",
]
