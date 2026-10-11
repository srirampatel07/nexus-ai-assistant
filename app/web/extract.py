"""Readable-text extraction for NEXUS web retrieval (Phase 6).

Stdlib only (no new dependencies): one `HTMLParser` pass collects the
page title, heading/paragraph/list text, and anchor links. Script, style,
noscript, template, and comment content is dropped. Entity references are
unescaped and whitespace is collapsed.

Extracted text is UNTRUSTED page data — never instructions. Callers must
label it as retrieved source material, not assistant speech.
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin

_TEXT_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre"}
_SKIP_TAGS = {"script", "style", "noscript", "template", "svg"}
_MAX_LINKS = 50


@dataclass
class PageLink:
    id: int
    text: str
    url: str


@dataclass
class PageText:
    title: str = ""
    text: str = ""
    links: list[PageLink] = field(default_factory=list)


class _ReaderParser(HTMLParser):
    """Single-pass readable-text collector."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base = base_url
        self._title_parts: list[str] = []
        self._chunks: list[str] = []
        self._current: list[str] = []
        self._in_title = False
        self._skip_depth = 0
        self._in_text_tag = False
        self._link_href: str | None = None
        self._link_text: list[str] = []
        self._raw_links: list[tuple[str, str]] = []

    # -- tag handling -----------------------------------------------------
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _TEXT_TAGS:
            self._in_text_tag = True
            self._current = []
        if tag == "a":
            href = dict(attrs).get("href") or ""
            self._link_href = href.strip()
            self._link_text = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if tag in _TEXT_TAGS:
            self._flush_current()
            self._in_text_tag = False
        if tag == "a":
            if self._link_href is not None:
                text = _clean(" ".join(self._link_text))
                if text and self._link_href:
                    self._raw_links.append((text, self._link_href))
            self._link_href = None
            self._link_text = []

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self._title_parts.append(data)
            return
        if self._link_href is not None:
            self._link_text.append(data)
        if self._in_text_tag:
            self._current.append(data)

    def handle_comment(self, data: str) -> None:
        return None  # comments are never readable content

    # -- results ------------------------------------------------------------
    def _flush_current(self) -> None:
        text = _clean(" ".join(self._current))
        if text:
            self._chunks.append(text)
        self._current = []

    def result(self) -> PageText:
        self._flush_current()
        title = _clean(" ".join(self._title_parts))
        links: list[PageLink] = []
        for index, (text, href) in enumerate(self._raw_links[:_MAX_LINKS]):
            absolute = urljoin(self._base, href)
            if absolute.lower().startswith(("http://", "https://")):
                links.append(PageLink(id=index + 1, text=text[:120], url=absolute))
        return PageText(title=title, text="\n".join(self._chunks), links=links)


def _clean(value: str) -> str:
    """Unescape entities and collapse all whitespace runs."""
    return " ".join(html.unescape(value or "").split())


def extract_readable(html_text: str, base_url: str) -> PageText:
    """Parse `html_text` into title, readable text, and anchor links."""
    parser = _ReaderParser(base_url)
    try:
        parser.feed(html_text or "")
        parser.close()
    except Exception:
        pass  # best-effort: return whatever was collected so far
    return parser.result()


@dataclass
class SearchHit:
    """One raw search-result candidate (validated downstream)."""

    title: str
    url: str
    snippet: str = ""


class _ResultsParser(HTMLParser):
    """Collect DuckDuckGo HTML result anchors.

    Real result links look like (verified live against
    html.duckduckgo.com/html/):
      <a class="result__a" href="//duckduckgo.com/l/?uddg=<pct-target>&rut=…">
      <a class="result__snippet" href="…same…">snippet <b>text</b></a>
    The true target is the `uddg` query value, percent-decoded. Anything
    without a usable target (nav, ads, scripts) is ignored here and
    validated again by the caller.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.titles: list[tuple[str, str]] = []  # (text, href)
        self.snippets: list[tuple[str, str]] = []  # (text, href)
        self._kind: str | None = None
        self._href = ""
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        classes = (dict(attrs).get("class") or "").split()
        if "result__a" in classes:
            self._kind = "title"
        elif "result__snippet" in classes:
            self._kind = "snippet"
        else:
            return
        self._href = (dict(attrs).get("href") or "").strip()
        self._text = []

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or self._kind is None:
            return
        text = _clean(" ".join(self._text))
        if self._kind == "title":
            self.titles.append((text, self._href))
        else:
            self.snippets.append((text, self._href))
        self._kind = None
        self._href = ""
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._kind is not None:
            self._text.append(data)


def extract_search_hits(html_text: str) -> list[SearchHit]:
    """Pair result titles with their snippets by decoded target URL."""
    parser = _ResultsParser()
    try:
        parser.feed(html_text or "")
        parser.close()
    except Exception:
        pass
    snippets: dict[str, str] = {}
    for text, href in parser.snippets:
        target = _result_target(href)
        if target is not None and text and target not in snippets:
            snippets[target] = text
    hits: list[SearchHit] = []
    for text, href in parser.titles:
        target = _result_target(href)
        if target is None or not text:
            continue
        hits.append(
            SearchHit(title=text, url=target, snippet=snippets.get(target, ""))
        )
    return hits


def _result_target(href: str) -> str | None:
    """Resolve a result href to a validated absolute http(s) URL, or None."""
    from urllib.parse import parse_qsl, urlsplit, urlunsplit

    raw = (href or "").strip()
    if not raw:
        return None
    if raw.startswith("//"):
        raw = "https:" + raw
    try:
        parts = urlsplit(raw)
    except ValueError:
        return None
    if parts.hostname and parts.hostname.lower() == "duckduckgo.com":
        target = ""
        try:
            target = dict(parse_qsl(parts.query)).get("uddg", "")
        except ValueError:
            target = ""
        raw = (target or "").strip()
        if not raw:
            return None
        try:
            parts = urlsplit(raw)
        except ValueError:
            return None
    if parts.scheme.lower() not in ("http", "https"):
        return None
    if not parts.hostname or parts.username or parts.password:
        return None
    host = parts.hostname.lower()
    if "." not in host and ":" not in host:
        return None
    return urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), parts.path, parts.query, "")
    )
