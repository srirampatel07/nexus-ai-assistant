"""Reader browser session + tool tests (Phase 6). HTTP mocked throughout."""

import httpx
import pytest

from app.browser.errors import BrowserUnavailableError
from app.browser.session import BrowserManager
from app.core.config import Settings
from app.security.audit import AuditLogger
from app.security.policy import PermissionLevel
from app.tools import build_default_registry
from app.tools.browser import (
    BrowserClickTool,
    BrowserNavigateTool,
    BrowserReadTool,
)
from app.tools.executor import ToolExecutor
from app.web.client import WebContext

_HOME = """<html><head><title>Home</title></head><body>
<h1>Welcome</h1><p>Start here.</p>
<p><a href="/about">About us</a> <a href="https://other.test/x">Other</a></p>
</body></html>"""

_ABOUT = """<html><head><title>About</title></head><body>
<h1>About</h1><p>We are a test page.</p>
</body></html>"""


def _public_dns(host):
    return [(2, 1, 6, "", ("93.184.216.34", 0))]


def _ctx(handler, **over):
    transport = httpx.MockTransport(handler)
    kwargs = dict(
        fetch_timeout=5.0,
        fetch_max_bytes=1_048_576,
        fetch_max_redirects=5,
        text_max_chars=20000,
        dns_resolver=_public_dns,
    )
    kwargs.update(over)
    return WebContext(
        client_factory=lambda timeout: httpx.Client(
            transport=transport, timeout=timeout
        ),
        **kwargs,
    )


def _site(handler_pages):
    def handler(request: httpx.Request) -> httpx.Response:
        body = handler_pages.get(request.url.path)
        if body is None:
            return httpx.Response(404, text="missing")
        return httpx.Response(
            200, headers={"content-type": "text/html"}, text=body
        )

    return handler


def _manager(pages=None):
    return BrowserManager(_ctx(_site(pages or {"/": _HOME, "/about": _ABOUT})))


def test_navigate_and_read():
    mgr = _manager()
    page = mgr.open("s1").navigate("https://example.com/")
    assert page.title == "Home"
    assert "Welcome" in page.text
    assert len(page.links) == 2
    assert mgr.open("s1").current.url == "https://example.com/"


def test_links_have_stable_ids():
    session = _manager().open()
    session.navigate("https://example.com/")
    links = session.current.links
    assert [link.id for link in links] == [1, 2]
    assert links[0].url == "https://example.com/about"


def test_click_follows_link_and_back_returns():
    session = _manager().open()
    session.navigate("https://example.com/")
    page = session.click(1)
    assert page.url == "https://example.com/about"
    assert "test page" in page.text
    back = session.back()
    assert back.url == "https://example.com/"


def test_click_unknown_id_is_honest():
    session = _manager().open()
    session.navigate("https://example.com/")
    from app.browser.errors import ElementNotFoundError

    with pytest.raises(ElementNotFoundError) as exc:
        session.click(99)
    assert "99" in str(exc.value)


def test_read_without_page_is_honest():
    mgr = _manager()
    from app.browser.errors import NoPageError

    with pytest.raises(NoPageError):
        mgr.open("empty").current
    assert mgr.open("empty").history() == []


def test_back_at_first_page_is_honest():
    from app.browser.errors import NoPageError

    session = _manager().open()
    session.navigate("https://example.com/")
    with pytest.raises(NoPageError):
        session.back()


def test_close_releases_session():
    mgr = _manager()
    mgr.open("tmp").navigate("https://example.com/")
    assert "tmp" in mgr.session_ids()
    assert mgr.close("tmp") is True
    assert mgr.close("tmp") is False
    with pytest.raises(BrowserUnavailableError):
        mgr.get("tmp")


def test_navigation_failure_is_honest():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="down")

    mgr = BrowserManager(_ctx(handler))
    from app.web.errors import WebError

    with pytest.raises(WebError) as exc:
        mgr.open().navigate("https://example.com/")
    assert "500" in str(exc.value)


def test_session_ids_are_labels_not_paths():
    mgr = _manager()
    mgr.open("a")
    mgr.open("b")
    assert mgr.session_ids() == ["a", "b"]


def test_browser_tools_registered_with_risk_levels():
    registry = build_default_registry(Settings(ai_provider="echo"))
    assert registry.get("browser.navigate").risk_level == PermissionLevel.LOW
    assert registry.get("browser.read").risk_level == PermissionLevel.SAFE
    assert registry.get("browser.links").risk_level == PermissionLevel.SAFE
    click = registry.get("browser.click")
    assert click.risk_level == PermissionLevel.MEDIUM
    assert click.requires_confirmation is True
    assert isinstance(click, BrowserClickTool)
    assert isinstance(registry.get("browser.navigate"), BrowserNavigateTool)
    assert isinstance(registry.get("browser.read"), BrowserReadTool)


@pytest.mark.asyncio
async def test_click_requires_approval(tmp_path):
    registry = build_default_registry(Settings(ai_provider="echo"))
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(tmp_path / "a.jsonl")
    )
    mock = _ctx(_site({"/": _HOME, "/about": _ABOUT}))
    navigate_tool = registry.get("browser.navigate")
    click_tool = registry.get("browser.click")
    navigate_tool.ctx = mock
    click_tool.ctx = mock
    # Both tools share one manager: point its HTTP layer at the mock too.
    navigate_tool.manager._ctx = mock
    nav = await executor.execute(
        "browser.navigate", {"url": "https://example.com/"}
    )
    assert nav.ok is True
    # Default executor auto-denies: click must not run without approval.
    denied = await executor.execute("browser.click", {"target": "1"})
    assert denied.ok is False
    assert "approval" in (denied.error or "").lower()
    # Explicit approval lets the same click through.
    from app.security.confirmation import auto_approve

    approved = await executor.execute(
        "browser.click", {"target": "1"}, confirm=auto_approve
    )
    assert approved.ok is True
    assert approved.output["url"] == "https://example.com/about"


@pytest.mark.asyncio
async def test_click_unsafe_url_denied_by_policy(tmp_path):
    registry = build_default_registry(Settings(ai_provider="echo"))
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(tmp_path / "a.jsonl")
    )
    mock = _ctx(_site({"/": _HOME}))
    navigate_tool = registry.get("browser.navigate")
    click_tool = registry.get("browser.click")
    navigate_tool.ctx = mock
    click_tool.ctx = mock
    navigate_tool.manager._ctx = mock
    await executor.execute("browser.navigate", {"url": "https://example.com/"})
    result = await executor.execute(
        "browser.click", {"target": "http://169.254.169.254/"}
    )
    assert result.ok is False


def test_forms_and_scripts_are_unsupported_not_faked():
    session = _manager().open()
    session.navigate("https://example.com/")
    assert not hasattr(session, "fill_form")
    assert not hasattr(session, "run_script")
    assert not hasattr(session, "download")
