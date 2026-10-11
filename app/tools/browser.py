"""Browser session tools (Phase 6).

Document-navigation tools over `BrowserManager` reader sessions. Read-only
operations run immediately; `browser.click` follows a link (a navigation
that can land anywhere, including state-changing GET endpoints), so it is
MEDIUM risk and always asks for approval first.

Unsupported operations (forms, script-backed buttons, logins, downloads,
payments) are reported as unsupported — never faked.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.browser.errors import (
    BrowserError,
    ElementNotFoundError,
    NoPageError,
)
from app.browser.session import DEFAULT_SESSION_ID, BrowserManager
from app.core.exceptions import ToolDenied
from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult
from app.web.client import WebContext
from app.web.errors import UnsafeDestinationError, WebError

_SESSION_PATTERN = r"^[A-Za-z0-9_-]{1,32}$"


class _SessionInput(BaseModel):
    session_id: str = Field(default=DEFAULT_SESSION_ID, pattern=_SESSION_PATTERN)


class NavigateInput(_SessionInput):
    url: str = Field(description="Page URL to open (http/https only).")


class ClickInput(_SessionInput):
    target: str = Field(
        description="Link id (e.g. '3') from browser.links, or a full URL."
    )


def _page_output(session, page, extra: dict | None = None) -> dict:
    out = {
        "session_id": session.id,
        "url": page.url,
        "title": page.title,
        "text": page.text,
        "link_count": len(page.links),
    }
    if extra:
        out.update(extra)
    return out


class BrowserNavigateTool(Tool):
    name = "browser.navigate"
    description = (
        "Open a web page in a browser session. Restricted destinations "
        "are refused; failures are reported honestly."
    )
    category = "browser"
    risk_level = PermissionLevel.LOW
    input_model = NavigateInput

    def __init__(
        self,
        ctx: WebContext,
        manager: BrowserManager,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.ctx = ctx
        self.manager = manager
        self.timeout_seconds = timeout_seconds

    def assess(self, validated: BaseModel) -> PermissionLevel:
        assert isinstance(validated, NavigateInput)
        from app.web.ssrf import check_url

        try:
            check_url(validated.url.strip(), dns_resolver=self.ctx.resolve_host)
        except UnsafeDestinationError as exc:
            raise ToolDenied(str(exc)) from exc
        return self.risk_level

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, NavigateInput)
        session = self.manager.open(validated.session_id)
        try:
            page = session.navigate(validated.url.strip())
        except UnsafeDestinationError as exc:
            return ToolResult(ok=False, error=f"Refused: {exc}")
        except (WebError, BrowserError) as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:
            return ToolResult(
                ok=False, error=f"Navigation failed: {type(exc).__name__}."
            )
        return ToolResult(ok=True, output=_page_output(session, page))


class BrowserReadTool(Tool):
    name = "browser.read"
    description = "Read the current page title and visible text in a session."
    category = "browser"
    risk_level = PermissionLevel.SAFE
    input_model = _SessionInput

    def __init__(self, manager: BrowserManager) -> None:
        self.manager = manager

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, _SessionInput)
        try:
            session = self.manager.get(validated.session_id)
            page = session.current
        except BrowserError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(ok=True, output=_page_output(session, page))


class BrowserLinksTool(Tool):
    name = "browser.links"
    description = (
        "List the current page's links (id + text + URL) for browser.click."
    )
    category = "browser"
    risk_level = PermissionLevel.SAFE
    input_model = _SessionInput

    def __init__(self, manager: BrowserManager) -> None:
        self.manager = manager

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, _SessionInput)
        try:
            session = self.manager.get(validated.session_id)
            page = session.current
        except BrowserError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(
            ok=True,
            output={
                "session_id": session.id,
                "url": page.url,
                "links": [
                    {"id": link.id, "text": link.text, "url": link.url}
                    for link in page.links
                ],
            },
        )


class BrowserClickTool(Tool):
    name = "browser.click"
    description = (
        "Follow a page link by id (from browser.links) or open a URL. "
        "Needs approval: link destinations can trigger side effects."
    )
    category = "browser"
    risk_level = PermissionLevel.MEDIUM
    requires_confirmation = True
    input_model = ClickInput

    def __init__(
        self,
        ctx: WebContext,
        manager: BrowserManager,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.ctx = ctx
        self.manager = manager
        self.timeout_seconds = timeout_seconds

    def assess(self, validated: BaseModel) -> PermissionLevel:
        assert isinstance(validated, ClickInput)
        target = validated.target.strip()
        if not target.lstrip("-").isdigit():
            from app.web.ssrf import check_url

            try:
                check_url(target, dns_resolver=self.ctx.resolve_host)
            except UnsafeDestinationError as exc:
                raise ToolDenied(str(exc)) from exc
        return self.risk_level

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, ClickInput)
        session = self.manager.open(validated.session_id)
        target = validated.target.strip()
        try:
            if target.lstrip("-").isdigit():
                page = session.click(int(target))
            else:
                page = session.navigate(target)
        except ElementNotFoundError as exc:
            return ToolResult(ok=False, error=str(exc))
        except NoPageError as exc:
            return ToolResult(ok=False, error=str(exc))
        except UnsafeDestinationError as exc:
            return ToolResult(ok=False, error=f"Refused: {exc}")
        except (WebError, BrowserError) as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:
            return ToolResult(
                ok=False, error=f"Navigation failed: {type(exc).__name__}."
            )
        return ToolResult(ok=True, output=_page_output(session, page))


class BrowserBackTool(Tool):
    name = "browser.back"
    description = "Go back to the previous page in a session (cached)."
    category = "browser"
    risk_level = PermissionLevel.SAFE
    input_model = _SessionInput

    def __init__(self, manager: BrowserManager) -> None:
        self.manager = manager

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, _SessionInput)
        try:
            session = self.manager.get(validated.session_id)
            page = session.back()
        except BrowserError as exc:
            return ToolResult(ok=False, error=str(exc))
        return ToolResult(ok=True, output=_page_output(session, page))


class BrowserCloseTool(Tool):
    name = "browser.close"
    description = "Close a browser session and release its history."
    category = "browser"
    risk_level = PermissionLevel.SAFE
    input_model = _SessionInput

    def __init__(self, manager: BrowserManager) -> None:
        self.manager = manager

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, _SessionInput)
        closed = self.manager.close(validated.session_id)
        if not closed:
            return ToolResult(
                ok=False,
                error=f"Session '{validated.session_id}' does not exist.",
            )
        return ToolResult(ok=True, output={"closed": validated.session_id})
