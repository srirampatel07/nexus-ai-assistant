"""Web search + page retrieval tools (Phase 6).

Both tools are read-only (LOW risk, no confirmation) and go through the
standard executor pipeline like every other tool. They never fabricate
results: provider failures, empty answers, and retrieval failures are
returned honestly so the brain can report them instead of guessing.

- `web.search`: DuckDuckGo Instant Answer API (keyless, configured URL).
- `web.fetch`: single-page retrieval with SSRF checks, redirect/size
  caps, readable-type allowlist, and markup-stripped extraction.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.exceptions import ToolDenied
from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult
from app.web.client import WebContext, fetch_url, search_ddg
from app.web.errors import UnsafeDestinationError, WebError
from app.web.search_session import SearchSession


class SearchInput(BaseModel):
    query: str = Field(description="Search query (non-empty).", min_length=1)
    max_results: int = Field(default=0, ge=0, le=20)


class FetchInput(BaseModel):
    url: str = Field(description="Page URL to retrieve (http/https only).")


class WebSearchTool(Tool):
    name = "web.search"
    description = (
        "Search the web for current information. Returns real result "
        "titles, URLs, and snippets (keyless provider)."
    )
    category = "web"
    risk_level = PermissionLevel.LOW
    input_model = SearchInput

    def __init__(
        self,
        ctx: WebContext,
        timeout_seconds: float = 30.0,
        session: SearchSession | None = None,
    ) -> None:
        self.ctx = ctx
        self.timeout_seconds = timeout_seconds
        # Shared current-search session: successful searches are recorded so
        # follow-up references ("open result 2") resolve to exact real URLs.
        # NB: `or` is wrong here — an empty session is falsy via __len__.
        self.session = session if session is not None else SearchSession()

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, SearchInput)
        query = validated.query.strip()
        if not query:
            return ToolResult(ok=False, error="Empty search query.")
        try:
            found = search_ddg(self.ctx, query)
        except WebError as exc:
            # A failed search must not leave stale results behind as the
            # "current" session — references afterwards report honestly
            # that there is nothing to open yet.
            self.session.clear()
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:  # never leak internals
            self.session.clear()
            return ToolResult(ok=False, error=f"Search failed: {type(exc).__name__}.")
        limit = validated.max_results or self.ctx.search_max_results
        results = [
            {"title": r.title, "url": r.url, "snippet": r.snippet, "source": r.source}
            for r in found.results[: max(1, limit)]
        ]
        # Record real results only (failed searches leave the session intact
        # only when they return nothing new — an empty search clears it so
        # stale results can never be opened by accident).
        self.session.update(query, results)
        return ToolResult(
            ok=True,
            output={
                "query": found.query,
                "answer": found.answer,
                "answer_url": found.answer_url,
                "count": len(results),
                "results": results,
            },
        )


class WebFetchTool(Tool):
    name = "web.fetch"
    description = (
        "Retrieve one web page as readable text with its source URL. "
        "Restricted destinations are refused; unsupported pages are "
        "reported honestly, never fabricated."
    )
    category = "web"
    risk_level = PermissionLevel.LOW
    input_model = FetchInput

    def __init__(self, ctx: WebContext, timeout_seconds: float = 30.0) -> None:
        self.ctx = ctx
        self.timeout_seconds = timeout_seconds

    def assess(self, validated: BaseModel) -> PermissionLevel:
        """Refuse restricted destinations at policy level (audited deny)."""
        assert isinstance(validated, FetchInput)
        from app.web.ssrf import check_url

        try:
            check_url(validated.url.strip(), dns_resolver=self.ctx.resolve_host)
        except UnsafeDestinationError as exc:
            raise ToolDenied(str(exc)) from exc
        return self.risk_level

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, FetchInput)
        url = validated.url.strip()
        if not url:
            return ToolResult(ok=False, error="Empty URL.")
        try:
            page = fetch_url(self.ctx, url)
        except UnsafeDestinationError as exc:
            return ToolResult(ok=False, error=f"Refused: {exc}")
        except WebError as exc:
            return ToolResult(ok=False, error=str(exc))
        except Exception as exc:  # never leak internals
            return ToolResult(
                ok=False, error=f"Page retrieval failed: {type(exc).__name__}."
            )
        return ToolResult(
            ok=True,
            output={
                "url": page.final_url,
                "title": page.title,
                "text": page.text,
                "links": [
                    {"id": link.id, "text": link.text, "url": link.url}
                    for link in page.links
                ],
                "content_type": page.content_type,
                "redirects": page.redirects,
            },
        )
