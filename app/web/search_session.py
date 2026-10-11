"""Current-search session for result selection (Phase 6).

`web.search` records every successful search here; `brave.open` resolves
references such as "result 2", "the second result", or "first" against it.
This keeps destination URLs exact (taken from real search output, never
reconstructed or guessed) and scoped to the current search session.

Vague references ("that website", "the page you found") resolve only when
the session holds exactly one result — otherwise the caller must ask the
user which result they mean. Nothing is ever fabricated: unresolvable
references return None and the tool reports that honestly.
"""

from __future__ import annotations

import re
import threading

_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
}

# "result 2", "result #3", "2", "2nd", "second", "the second result", ...
_REFERENCE_RE = re.compile(
    r"^(?:the\s+)?"
    r"(?:result\s*#?\s*(?P<num>\d+)"
    r"|#(?P<hashnum>\d+)"
    r"|(?P<plain>\d+)(?:st|nd|rd|th)?(?:\s+result)?"
    r"|(?P<word>first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|last)(?:\s+result)?)"
    r"\s*$",
    re.IGNORECASE,
)

# Vague but common follow-ups; only meaningful with exactly one result.
_VAGUE_REFERENCES = frozenset(
    {
        "that website",
        "that site",
        "that page",
        "that article",
        "that link",
        "that result",
        "the website",
        "the page you found",
        "the page",
        "the article",
        "the link",
        "it",
        "that one",
    }
)


class SearchSession:
    """Latest successful `web.search` results for reference resolution."""

    def __init__(self, max_stored: int = 20) -> None:
        self._lock = threading.Lock()
        self._query = ""
        self._results: list[dict] = []
        self._max_stored = max(1, max_stored)

    def update(self, query: str, results: list[dict]) -> None:
        """Remember a successful search (replaces the previous session)."""
        with self._lock:
            self._query = query
            self._results = [
                {"title": r.get("title", ""), "url": r.get("url", ""),
                 "snippet": r.get("snippet", "")}
                for r in results[: self._max_stored]
                if r.get("url")
            ]

    def clear(self) -> None:
        with self._lock:
            self._query = ""
            self._results = []

    @property
    def query(self) -> str:
        with self._lock:
            return self._query

    @property
    def results(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._results]

    def __len__(self) -> int:
        with self._lock:
            return len(self._results)

    def url_at(self, index: int) -> str | None:
        """Zero-based URL lookup; None when out of range."""
        with self._lock:
            if 0 <= index < len(self._results):
                return self._results[index]["url"]
            return None

    def resolve(self, reference: str) -> str | None:
        """Map a result reference to its exact destination URL.

        Returns None when the reference is unresolvable (empty session,
        out-of-range number, or vague reference with != 1 result).
        """
        text = (reference or "").strip().lower()
        if not text:
            return None
        with self._lock:
            if not self._results:
                return None
            match = _REFERENCE_RE.match(text)
            if match:
                if match.group("word") == "last":
                    return self._results[-1]["url"]
                word = match.group("word")
                if word:
                    number = _ORDINALS[word]
                else:
                    raw = (
                        match.group("num") or match.group("hashnum")
                        or match.group("plain")
                    )
                    try:
                        number = int(raw)
                    except (TypeError, ValueError):
                        return None
                if 1 <= number <= len(self._results):
                    return self._results[number - 1]["url"]
                return None
            if text in _VAGUE_REFERENCES and len(self._results) == 1:
                return self._results[0]["url"]
            return None
