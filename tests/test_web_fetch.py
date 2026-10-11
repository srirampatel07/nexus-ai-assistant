"""Page-retrieval + extraction tests (Phase 6). HTTP mocked throughout."""

import httpx
import pytest

from app.tools.web import WebFetchTool
from app.web.client import WebContext
from app.web.errors import UnsafeDestinationError
from app.web.extract import extract_readable
from app.web.ssrf import check_url


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


_PAGE = """<html><head><title>Example Domain</title>
<style>.x{display:none}</style><script>evil()</script></head>
<body><h1>Example Domain</h1>
<p>This domain is for use in documentation. <!-- secret comment --></p>
<p><a href="/more">More information</a> and
<a href="https://other.test/page">other</a></p>
</body></html>"""


def test_extract_title_text_links_and_drops_markup():
    page = extract_readable(_PAGE, "https://example.com/")
    assert page.title == "Example Domain"
    assert "Example Domain" in page.text
    assert "documentation" in page.text
    assert "evil()" not in page.text
    assert "display:none" not in page.text
    assert "secret comment" not in page.text
    urls = [link.url for link in page.links]
    assert "https://example.com/more" in urls  # relative resolved
    assert "https://other.test/page" in urls
    assert all(link.id >= 1 and link.text for link in page.links)


def test_extract_empty_and_entities():
    page = extract_readable("", "https://example.com/")
    assert page.title == "" and page.text == ""
    assert extract_readable("<p>a &amp; b</p>", "https://e.com/").text == "a & b"


def _html_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        text=_PAGE,
    )


def test_fetch_html_page():
    tool = WebFetchTool(_ctx(_html_handler))
    result = tool.execute(tool.validate({"url": "https://example.com/"}))
    assert result.ok is True
    out = result.output
    assert out["url"] == "https://example.com/"
    assert out["title"] == "Example Domain"
    assert "documentation" in out["text"]
    assert out["content_type"] == "text/html"
    assert out["redirects"] == 0
    assert len(out["links"]) == 2


def test_fetch_plain_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/plain"}, text="hello world"
        )

    result = WebFetchTool(_ctx(handler)).execute(
        WebFetchTool(_ctx(handler)).validate({"url": "https://example.com/r"})
    )
    assert result.ok is True
    assert "hello world" in result.output["text"]


def test_fetch_invalid_url_refused():
    tool = WebFetchTool(_ctx(_html_handler))
    result = tool.execute(tool.validate({"url": "not a url at all"}))
    assert result.ok is False
    result = tool.execute(tool.validate({"url": "ftp://example.com/f"}))
    assert result.ok is False


def test_fetch_assess_denies_restricted_initial_url():
    from app.core.exceptions import ToolDenied

    tool = WebFetchTool(_ctx(_html_handler))

    with pytest.raises(ToolDenied):
        tool.assess(tool.validate({"url": "http://169.254.169.254/latest"}))


def test_fetch_404_and_500_are_honest():
    def handler(request: httpx.Request, status=404) -> httpx.Response:
        return httpx.Response(status, text="nope")

    for status, needle in ((404, "404"), (500, "500")):
        def h(request: httpx.Request, s=status) -> httpx.Response:
            return httpx.Response(s, text="nope")

        result = WebFetchTool(_ctx(h)).execute(
            WebFetchTool(_ctx(h)).validate({"url": "https://example.com/x"})
        )
        assert result.ok is False
        assert needle in result.error


def test_fetch_unsupported_content_type():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            content=b"\x89PNG" + b"\x00" * 100,
        )

    result = WebFetchTool(_ctx(handler)).execute(
        WebFetchTool(_ctx(handler)).validate({"url": "https://example.com/i.png"})
    )
    assert result.ok is False
    assert "content type" in result.error.lower()
    assert "\x89PNG" not in (result.error or "")


def test_fetch_oversized_response_refused():
    big = b"x" * 3000

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/plain"}, content=big
        )

    result = WebFetchTool(_ctx(handler, fetch_max_bytes=1024)).execute(
        WebFetchTool(_ctx(handler, fetch_max_bytes=1024)).validate(
            {"url": "https://example.com/big"}
        )
    )
    assert result.ok is False
    assert "size limit" in result.error.lower()


def test_fetch_redirect_chain():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/a":
            return httpx.Response(302, headers={"location": "/b"})
        return httpx.Response(
            200, headers={"content-type": "text/plain"}, text="final page"
        )

    result = WebFetchTool(_ctx(handler)).execute(
        WebFetchTool(_ctx(handler)).validate({"url": "https://example.com/a"})
    )
    assert result.ok is True
    assert result.output["url"] == "https://example.com/b"
    assert "final page" in result.output["text"]
    assert result.output["redirects"] == 1


def test_fetch_redirect_to_restricted_blocked():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/"})

    result = WebFetchTool(_ctx(handler)).execute(
        WebFetchTool(_ctx(handler)).validate({"url": "https://example.com/a"})
    )
    assert result.ok is False
    assert "estricted" in result.error or "efused" in result.error


def test_fetch_redirect_loop_capped():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "/loop"})

    result = WebFetchTool(_ctx(handler, fetch_max_redirects=2)).execute(
        WebFetchTool(_ctx(handler, fetch_max_redirects=2)).validate(
            {"url": "https://example.com/loop"}
        )
    )
    assert result.ok is False
    assert "redirect" in result.error.lower()


def test_fetch_uses_final_url_as_source():
    from app.web.errors import UnsafeDestinationError

    assert check_url("https://example.com/", dns_resolver=_public_dns)
    with pytest.raises(UnsafeDestinationError):
        check_url("http://localhost:8000/", dns_resolver=_public_dns)
    with pytest.raises(UnsafeDestinationError):
        from app.web.client import fetch_url

        fetch_url(_ctx(_html_handler), "http://localhost/")


def test_fetch_timeout_is_honest():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("too slow")

    result = WebFetchTool(_ctx(handler)).execute(
        WebFetchTool(_ctx(handler)).validate({"url": "https://example.com/"})
    )
    assert result.ok is False
    assert "nreachable" in result.error or "imeout" in result.error
