"""Web-search tool tests (Phase 6). HTTP mocked; no live network."""

import httpx
import pytest

from app.core.config import Settings
from app.security.audit import AuditLogger
from app.security.policy import PermissionLevel
from app.tools import build_default_registry
from app.tools.executor import ToolExecutor
from app.tools.web import WebSearchTool
from app.web.client import WebContext, search_ddg
from app.web.errors import WebError


def _public_dns(host):
    return [(2, 1, 6, "", ("93.184.216.34", 0))]


def _ctx(handler, **over):
    transport = httpx.MockTransport(handler)
    kwargs = dict(
        search_url="https://search.test/",
        search_timeout=5.0,
        search_max_results=8,
        dns_resolver=_public_dns,
    )
    kwargs.update(over)
    return WebContext(
        client_factory=lambda timeout: httpx.Client(
            transport=transport, timeout=timeout
        ),
        **kwargs,
    )


_DDG_OK = {
    "Heading": "Example",
    "AbstractText": "Example domains are reserved for documentation.",
    "AbstractURL": "https://en.wikipedia.org/wiki/Example",
    "AbstractSource": "Wikipedia",
    "RelatedTopics": [
        {"Text": "Example.com docs entry", "FirstURL": "https://example.com/"},
        {
            "Name": "group",
            "Topics": [
                {"Text": "Nested topic page", "FirstURL": "https://example.org/n"},
                "not-a-dict",
            ],
        },
        {"Text": "", "FirstURL": ""},
    ],
}


def _ok_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=_DDG_OK)


def test_search_returns_structured_results():
    found = search_ddg(_ctx(_ok_handler), "example domains")
    assert found.query == "example domains"
    assert "reserved for documentation" in found.answer
    assert found.answer_url == "https://en.wikipedia.org/wiki/Example"
    urls = [r.url for r in found.results]
    assert "https://en.wikipedia.org/wiki/Example" in urls
    assert "https://example.com/" in urls
    assert "https://example.org/n" in urls  # nested Topics flattened
    assert all(r.title and r.snippet for r in found.results)


def test_search_respects_max_results():
    found = search_ddg(_ctx(_ok_handler, search_max_results=1), "example")
    assert len(found.results) == 1


def test_search_uses_configured_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["host"] = request.url.host
        return httpx.Response(200, json={})

    search_ddg(_ctx(handler, search_url="https://custom.test/api/"), "q")
    assert seen["host"] == "custom.test"


def test_search_empty_query_rejected():
    with pytest.raises(WebError):
        search_ddg(_ctx(_ok_handler), "   ")
    tool = WebSearchTool(_ctx(_ok_handler))
    result = tool.execute(tool.validate({"query": "   "}))
    assert result.ok is False


def test_search_blank_query_fails_validation():
    from app.core.exceptions import ToolValidationError

    tool = WebSearchTool(_ctx(_ok_handler))
    with pytest.raises(ToolValidationError):
        tool.validate({"query": ""})


def test_search_rate_limited():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={})

    tool = WebSearchTool(_ctx(handler))
    result = tool.execute(tool.validate({"query": "news"}))
    assert result.ok is False
    assert "429" in result.error


def test_search_server_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    result = WebSearchTool(_ctx(handler)).execute(
        WebSearchTool(_ctx(handler)).validate({"query": "x"})
    )
    assert result.ok is False
    assert "500" in result.error


def test_search_malformed_response():
    """Garbage HTML parses to zero results honestly — never fabricated."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/html"}, text="not html{{{"
        )

    tool = WebSearchTool(_ctx(handler))
    result = tool.execute(tool.validate({"query": "x"}))
    assert result.ok is True
    assert result.output["count"] == 0


def test_search_malformed_json_still_rejected():
    """Legacy JSON endpoints keep strict malformed-response errors."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json{{{")
    ctx = _ctx(handler, search_url="https://api.duckduckgo.com/")
    with pytest.raises(WebError):
        search_ddg(ctx, "x")


_HTML_RESULTS = """<html><body>
<div class="result"><h2 class="result__title">
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2Fdownloads%2F&rut=abc">Download Python | Python.org</a>
</h2>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2Fdownloads%2F&rut=abc">The official home of the <b>Python</b> language.</a></div>
<div class="result"><h2 class="result__title">
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F&rut=def">Python docs</a>
</h2></div>
<div class="result"><h2 class="result__title">
<a class="result__a" href="https://direct.test/page">Direct link</a>
</h2>
<a class="result__snippet" href="https://direct.test/page">Served directly.</a></div>
<div class="result"><h2 class="result__title">
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2Fdownloads%2F&rut=xyz">Download Python mirror</a>
</h2>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.python.org%2Fdownloads%2F&rut=xyz">duplicate target</a></div>
<div class="nav"><a class="result__a" href="javascript:void(0)">bad scheme</a></div>
<div class="nav"><a class="result__a" href="/internal">relative nav</a></div>
<div class="result"><h2 class="result__title">
<a class="result__a" href="//duckduckgo.com/l/?uddg=&rut=empty">No target</a>
</h2></div>
<div class="result"><h2 class="result__title">
<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2F&rut=e"></a>
</h2></div>
<a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Forphan.test%2F&rut=o">orphan snippet, no title</a>
</body></html>"""


def _html_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, headers={"content-type": "text/html"}, text=_HTML_RESULTS
    )


def test_html_results_parsed_and_cleaned():
    tool = WebSearchTool(_ctx(_html_handler))
    result = tool.execute(tool.validate({"query": "python release"}))
    assert result.ok is True
    out = result.output
    assert out["query"] == "python release"
    assert out["count"] == 3
    first, docs, direct = out["results"]
    assert first["title"] == "Download Python | Python.org"
    assert first["url"] == "https://www.python.org/downloads/"
    assert "official home" in first["snippet"] and "<b>" not in first["snippet"]
    assert docs["url"] == "https://docs.python.org/"
    assert docs["snippet"] == ""
    assert direct["url"] == "https://direct.test/page"
    urls = [r["url"] for r in out["results"]]
    assert len(set(urls)) == len(urls)  # duplicate target excluded
    assert all(u.startswith("https://") for u in urls)
    assert not any("orphan" in r["snippet"] for r in out["results"])


def test_html_results_respect_limit():
    tool = WebSearchTool(_ctx(_html_handler, search_max_results=2))
    result = tool.execute(tool.validate({"query": "python"}))
    assert result.ok is True
    assert result.output["count"] == 2


def test_html_empty_page_is_honest():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text="<html><body><p>no results here</p></body></html>",
        )

    result = WebSearchTool(_ctx(handler)).execute(
        WebSearchTool(_ctx(handler)).validate({"query": "obscure thing"})
    )
    assert result.ok is True
    assert result.output["count"] == 0
    assert result.output["results"] == []


def test_html_oversized_body_is_honest():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html"},
            content=b"x" * 5000,
        )

    ctx = _ctx(handler, fetch_max_bytes=100)
    result = WebSearchTool(ctx).execute(
        WebSearchTool(ctx).validate({"query": "x"})
    )
    assert result.ok is False


def test_search_connection_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("dns down")

    with pytest.raises(WebError) as exc:
        search_ddg(_ctx(handler), "x")
    assert "unreachable" in str(exc.value)


def test_search_no_results_is_honest():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"RelatedTopics": []})

    tool = WebSearchTool(_ctx(handler))
    result = tool.execute(tool.validate({"query": "obscure thing"}))
    assert result.ok is True
    assert result.output["count"] == 0
    assert result.output["results"] == []


def test_search_tool_risk_and_registration():
    registry = build_default_registry(Settings(ai_provider="echo"))
    tool = registry.get("web.search")
    assert isinstance(tool, WebSearchTool)
    assert tool.risk_level == PermissionLevel.LOW
    assert tool.requires_confirmation is False
    assert tool.category == "web"


@pytest.mark.asyncio
async def test_search_executes_without_confirmation(tmp_path):
    registry = build_default_registry(Settings(ai_provider="echo"))
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(tmp_path / "a.jsonl")
    )
    tool = registry.get("web.search")
    tool.ctx = _ctx(_ok_handler)  # point at the mock, keep policy wiring
    result = await executor.execute("web.search", {"query": "example"})
    assert result.ok is True
    assert result.output["count"] >= 1
