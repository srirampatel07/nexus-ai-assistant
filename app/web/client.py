"""HTTP web client for NEXUS search and retrieval (Phase 6).

All outbound traffic flows through here:

- `search_ddg()`: DuckDuckGo Instant Answer API
  (`https://api.duckduckgo.com/?q=...&format=json`, verified live; keyless).
  Returns abstract + related topics as structured results.
- `fetch_url()`: page retrieval with per-hop SSRF validation (initial URL
  AND every redirect target), redirect cap, response byte cap, and an
  allowlist of readable content types.

The HTTP client factory and DNS resolver are injectable so tests use
`httpx.MockTransport` and fake DNS — the suite never touches live sites.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlencode, urljoin, urlsplit

import httpx

from app.core.config import Settings
from app.web.errors import UnsafeDestinationError, WebError
from app.web.extract import PageLink, extract_readable
from app.web.ssrf import AddrInfo, check_url, default_resolver

USER_AGENT = "NEXUS-personal-assistant/6.0 (+local research tool)"

# Readable content only: markup we can extract, plain text, or JSON.
_READABLE_TYPES = {
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "application/json",
    "application/rss+xml",
    "application/atom+xml",
}


def _is_readable(content_type: str) -> bool:
    base = (content_type or "").split(";")[0].strip().lower()
    return base.startswith("text/") or base in _READABLE_TYPES


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str = ""


@dataclass
class SearchResults:
    query: str
    answer: str = ""
    answer_url: str = ""
    results: list[SearchResult] = field(default_factory=list)


@dataclass
class FetchedPage:
    url: str  # final URL after redirects (citable source)
    final_url: str = ""
    title: str = ""
    text: str = ""
    links: list[PageLink] = field(default_factory=list)
    content_type: str = ""
    bytes_read: int = 0
    redirects: int = 0

    def __post_init__(self) -> None:
        if not self.final_url:
            self.final_url = self.url


@dataclass
class WebContext:
    """Injectable web dependencies (mirrors FilesystemContext pattern)."""

    search_provider: str = "duckduckgo"
    search_url: str = "https://api.duckduckgo.com/"
    search_timeout: float = 15.0
    search_max_results: int = 8
    fetch_timeout: float = 20.0
    fetch_max_bytes: int = 1_048_576
    fetch_max_redirects: int = 5
    text_max_chars: int = 20000
    user_agent: str = USER_AGENT
    client_factory: Callable[..., httpx.Client] | None = None
    dns_resolver: Callable[[str], Sequence[AddrInfo]] | None = None

    @classmethod
    def from_settings(cls, settings: Settings) -> WebContext:
        return cls(
            search_provider=settings.web_search_provider,
            search_url=settings.web_search_url,
            search_timeout=settings.web_search_timeout,
            search_max_results=settings.web_search_max_results,
            fetch_timeout=settings.web_fetch_timeout,
            fetch_max_bytes=settings.web_fetch_max_bytes,
            fetch_max_redirects=settings.web_fetch_max_redirects,
            text_max_chars=settings.web_text_max_chars,
        )

    def _client(self, timeout: float) -> httpx.Client:
        if self.client_factory is not None:
            return self.client_factory(timeout=timeout)
        return httpx.Client(
            timeout=timeout,
            follow_redirects=False,
            headers={"User-Agent": self.user_agent},
        )

    def resolve_host(self, host: str) -> Sequence[AddrInfo]:
        """Resolve `host` (injectable in tests; OS resolver by default)."""
        if self.dns_resolver is not None:
            return self.dns_resolver(host)
        return default_resolver(host)

    def _resolve(self, host: str) -> Sequence[AddrInfo]:
        """Internal alias kept for symmetry with `_client`."""
        return self.resolve_host(host)


def search_ddg(ctx: WebContext, query: str) -> SearchResults:
    """Run one search; raises WebError (never fabricates results).

    Backend is chosen by the configured endpoint: the keyless
    DuckDuckGo HTML results page (default) is parsed for real result
    links; a legacy Instant Answer JSON endpoint still parses as before.
    """
    question = (query or "").strip()
    if not question:
        raise WebError("Empty search query.")
    base = (ctx.search_url or "").rstrip("/") + "/"
    legacy = "api.duckduckgo.com" in base
    if legacy:
        params = {"q": question, "format": "json", "no_html": "1", "skip_disambig": "1"}
    else:
        params = {"q": question}
    url = base + "?" + urlencode(params)
    try:
        with ctx._client(ctx.search_timeout) as client:
            resp = client.get(url, headers={"User-Agent": ctx.user_agent})
    except (httpx.ConnectError, httpx.TimeoutException) as exc:
        raise WebError(f"Search service unreachable: {type(exc).__name__}.") from exc
    except httpx.HTTPError as exc:
        raise WebError(f"Search request failed: {type(exc).__name__}.") from exc
    if resp.status_code == 429:
        raise WebError("Search service rate-limited us (429); try again later.")
    if resp.status_code != 200:
        raise WebError(f"Search service error (HTTP {resp.status_code}).")
    content_type = resp.headers.get("content-type", "")
    if "json" in content_type.lower() or legacy:
        try:
            data = resp.json()
        except ValueError as exc:
            raise WebError("Search service returned a malformed response.") from exc
        if not isinstance(data, dict):
            raise WebError("Search service returned a malformed response.")
        return _parse_ddg(question, data, ctx.search_max_results)
    try:
        raw = _read_capped(resp, ctx.fetch_max_bytes)
    except WebError:
        raise WebError(
            "Search response exceeds the size limit; try a narrower query."
        )
    try:
        body = raw.decode(resp.encoding or "utf-8", errors="replace")
    except (LookupError, ValueError):
        body = raw.decode("utf-8", errors="replace")
    return _parse_ddg_html(question, body, ctx.search_max_results)


def _parse_ddg_html(query: str, html_text: str, limit: int) -> SearchResults:
    """Map an HTML results page to structured results (best-effort).

    Unparseable pages yield zero results honestly — never fabricated.
    """
    from app.web.extract import extract_search_hits

    out = SearchResults(query=query)
    seen: set[str] = set()
    for hit in extract_search_hits(html_text):
        if len(out.results) >= max(1, limit):
            break
        if hit.url in seen:
            continue
        seen.add(hit.url)
        out.results.append(
            SearchResult(
                title=hit.title[:200],
                url=hit.url,
                snippet=hit.snippet[:500],
                source="",
            )
        )
    return out


def _parse_ddg(query: str, data: Mapping[str, Any], limit: int) -> SearchResults:
    """Map the Instant Answer payload to structured results (best-effort)."""
    out = SearchResults(query=query)
    abstract = str(data.get("AbstractText") or "").strip()
    if abstract:
        out.answer = abstract[:2000]
        out.answer_url = str(data.get("AbstractURL") or "")
        out.results.append(
            SearchResult(
                title=str(data.get("Heading") or abstract[:80]),
                url=out.answer_url,
                snippet=abstract[:500],
                source=str(data.get("AbstractSource") or ""),
            )
        )
    for topic in _flatten_topics(data.get("RelatedTopics")):
        if len(out.results) >= max(1, limit):
            break
        text = str(topic.get("Text") or "").strip()
        url = str(topic.get("FirstURL") or "").strip()
        if not text or not url:
            continue
        out.results.append(
            SearchResult(title=text[:120], url=url, snippet=text[:500], source="")
        )
    return out


def _flatten_topics(raw: Any) -> list[Mapping[str, Any]]:
    """RelatedTopics may nest grouped Topics lists; flatten one level."""
    flat: list[Mapping[str, Any]] = []
    if not isinstance(raw, list):
        return flat
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        nested = entry.get("Topics")
        if isinstance(nested, list):
            flat.extend(t for t in nested if isinstance(t, dict))
        else:
            flat.append(entry)
    return flat


def fetch_url(ctx: WebContext, raw_url: str) -> FetchedPage:
    """Retrieve one page with per-hop SSRF checks; raises WebError."""
    current = _checked(raw_url, ctx)
    redirects = 0
    try:
        with ctx._client(ctx.fetch_timeout) as client:
            while True:
                try:
                    resp = client.get(current, headers={"User-Agent": ctx.user_agent})
                except (httpx.ConnectError, httpx.TimeoutException) as exc:
                    raise WebError(
                        f"Page unreachable: {type(exc).__name__}."
                    ) from exc
                except httpx.HTTPError as exc:
                    raise WebError(
                        f"Page request failed: {type(exc).__name__}."
                    ) from exc
                if resp.status_code in (301, 302, 303, 307, 308):
                    if redirects >= ctx.fetch_max_redirects:
                        raise WebError("Too many redirects; refused.")
                    location = resp.headers.get("location", "")
                    if not location:
                        raise WebError("Redirect without a target; refused.")
                    redirects += 1
                    current = _checked(urljoin(current, location), ctx)
                    continue
                return _read_response(ctx, current, resp, redirects)
    except (WebError, UnsafeDestinationError):
        raise
    except Exception as exc:  # never leak internals or bodies
        raise WebError(f"Page retrieval failed: {type(exc).__name__}.") from exc


def _checked(raw_url: str, ctx: WebContext) -> str:
    """SSRF-check one hop; surfaces refusals as plain WebError for callers."""
    return check_url(raw_url, dns_resolver=ctx.resolve_host)


def _read_response(
    ctx: WebContext, url: str, resp: httpx.Response, redirects: int
) -> FetchedPage:
    if resp.status_code == 429:
        raise WebError("Page rate-limited us (429); try again later.")
    if resp.status_code == 404:
        raise WebError(f"Page not found (404): {urlsplit(url).hostname or 'host'}.")
    if resp.status_code != 200:
        raise WebError(f"Page error (HTTP {resp.status_code}).")
    content_type = resp.headers.get("content-type", "")
    if not _is_readable(content_type):
        base = content_type.split(";")[0].strip() or "(unknown)"
        raise WebError(f"Unsupported content type '{base}'; not retrieved.")
    raw = _read_capped(resp, ctx.fetch_max_bytes)
    encoding = resp.encoding or "utf-8"
    try:
        text = raw.decode(encoding, errors="replace")
    except (LookupError, ValueError):
        text = raw.decode("utf-8", errors="replace")
    base_type = content_type.split(";")[0].strip().lower()
    if base_type in ("text/plain", "application/json"):
        body = " ".join(text.split())[: ctx.text_max_chars]
        return FetchedPage(
            url=url,
            title="",
            text=body,
            links=[],
            content_type=base_type,
            bytes_read=len(raw),
            redirects=redirects,
        )
    page = extract_readable(text, url)
    return FetchedPage(
        url=url,
        title=page.title[:300],
        text=page.text[: ctx.text_max_chars],
        links=page.links,
        content_type=base_type,
        bytes_read=len(raw),
        redirects=redirects,
    )


def _read_capped(resp: httpx.Response, limit: int) -> bytes:
    """Read the body up to limit+1 bytes (proves oversize without keeping it)."""
    chunks: list[bytes] = []
    total = 0
    for chunk in resp.iter_bytes(chunk_size=65536):
        chunks.append(chunk)
        total += len(chunk)
        if total > limit:
            raise WebError(
                f"Page exceeds the size limit ({limit} bytes); not retrieved."
            )
    return b"".join(chunks)


# Re-export for callers that only need the socket default indirectly.
__all__ = [
    "FetchedPage",
    "SearchResult",
    "SearchResults",
    "WebContext",
    "fetch_url",
    "search_ddg",
]
