"""Reader browser sessions for NEXUS (Phase 6).

A deliberate, honest scope: document navigation over HTTP — no JavaScript
engine, no form submission, no downloads, no plugins. Each session keeps
its own page history in memory; closing a session drops it.

Supported workflow:
  navigate(url) -> read() title/text -> links() -> click(id|url) / back()

Supported elements: page links (anchors) only. Anything else — forms,
buttons backed by scripts, file pickers, logins, checkouts — is reported
as unsupported, never faked. Pages are fetched through the shared web
client, so SSRF protection, size caps, and readable-type limits apply
identically to `web.fetch`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.browser.errors import (
    BrowserUnavailableError,
    ElementNotFoundError,
    NoPageError,
)
from app.web.client import WebContext, fetch_url
from app.web.errors import UnsafeDestinationError, WebError
from app.web.extract import PageLink

DEFAULT_SESSION_ID = "default"


@dataclass
class ReaderPage:
    """One loaded document: evidence, not interpretation."""

    url: str
    title: str = ""
    text: str = ""
    links: list[PageLink] = field(default_factory=list)
    redirects: int = 0


class ReaderSession:
    """One stateful reading session (single owner; not thread-safe)."""

    def __init__(self, session_id: str, ctx: WebContext) -> None:
        self.id = session_id
        self._ctx = ctx
        self._pages: list[ReaderPage] = []
        self._index = -1
        self.navigations = 0

    # -- state ------------------------------------------------------------
    @property
    def has_page(self) -> bool:
        return 0 <= self._index < len(self._pages)

    @property
    def current(self) -> ReaderPage:
        if not self.has_page:
            raise NoPageError(
                f"Session '{self.id}' has no page loaded; navigate first."
            )
        return self._pages[self._index]

    def history(self) -> list[str]:
        """Visited URLs oldest-first (for back/close UX, not content)."""
        return [p.url for p in self._pages]

    # -- operations ---------------------------------------------------------
    def navigate(self, url: str) -> ReaderPage:
        """Load `url` as the new current page (may raise WebError)."""
        try:
            fetched = fetch_url(self._ctx, url)
        except UnsafeDestinationError as exc:
            raise exc
        except WebError:
            raise
        except Exception as exc:  # never leak internals
            raise WebError(f"Navigation failed: {type(exc).__name__}.") from exc
        page = ReaderPage(
            url=fetched.final_url,
            title=fetched.title,
            text=fetched.text,
            links=fetched.links,
            redirects=fetched.redirects,
        )
        del self._pages[self._index + 1 :]
        self._pages.append(page)
        self._index = len(self._pages) - 1
        self.navigations += 1
        return page

    def back(self) -> ReaderPage:
        """Return to the previous page (cached; no refetch)."""
        if self._index <= 0:
            raise NoPageError(
                f"Session '{self.id}' is already at its first page."
            )
        self._index -= 1
        return self._pages[self._index]

    def click(self, link_id: int) -> ReaderPage:
        """Follow one of the current page's links by id (1-based)."""
        page = self.current
        for link in page.links:
            if link.id == link_id:
                return self.navigate(link.url)
        raise ElementNotFoundError(
            f"Link #{link_id} is not on the current page "
            f"(has {len(page.links)} link(s))."
        )

    def close(self) -> None:
        """Release session state (idempotent)."""
        self._pages.clear()
        self._index = -1


class BrowserManager:
    """Owns reader sessions; one manager per assistant wiring."""

    def __init__(self, ctx: WebContext) -> None:
        self._ctx = ctx
        self._sessions: dict[str, ReaderSession] = {}

    def open(self, session_id: str = DEFAULT_SESSION_ID) -> ReaderSession:
        """Get-or-create a session (names are plain labels, not paths)."""
        key = (session_id or "").strip() or DEFAULT_SESSION_ID
        session = self._sessions.get(key)
        if session is None:
            session = ReaderSession(key, self._ctx)
            self._sessions[key] = session
        return session

    def get(self, session_id: str = DEFAULT_SESSION_ID) -> ReaderSession:
        """Return an existing session or raise BrowserUnavailableError."""
        key = (session_id or "").strip() or DEFAULT_SESSION_ID
        try:
            return self._sessions[key]
        except KeyError:
            raise BrowserUnavailableError(
                f"Session '{key}' does not exist; navigate first."
            ) from None

    def close(self, session_id: str = DEFAULT_SESSION_ID) -> bool:
        """Drop a session and its history. True if one existed."""
        key = (session_id or "").strip() or DEFAULT_SESSION_ID
        session = self._sessions.pop(key, None)
        if session is None:
            return False
        session.close()
        return True

    def session_ids(self) -> list[str]:
        return sorted(self._sessions)
