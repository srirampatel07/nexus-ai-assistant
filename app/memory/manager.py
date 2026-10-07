"""Public memory API for NEXUS (Phase 3).

`MemoryManager` is the ONLY entry point the assistant and CLI may use —
SQLAlchemy models never leave this package (Pydantic `MemoryRecord`s do).
Retrieval is deterministic: text match, then importance, then recency,
with id as final tiebreak. No embeddings, no vector store (v1).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine

from app.core.exceptions import MemoryRejected
from app.memory.categories import (
    ALL_CATEGORIES,
    MemoryCategory,
    normalize_category,
)
from app.memory.database import create_memory_engine, init_db
from app.memory.models import MemoryEntry
from app.memory.redaction import evaluate
from app.memory.repository import MemoryRepository

# Global-knowledge categories injected across conversations (never scoped).
GLOBAL_CATEGORIES: tuple[str, ...] = (
    MemoryCategory.PREFERENCE.value,
    MemoryCategory.FACT.value,
    MemoryCategory.PROJECT.value,
    MemoryCategory.SETTING.value,
)


class MemoryRecord(BaseModel):
    """JSON-serializable view of one memory entry (public DTO)."""

    id: int
    conversation_id: str = ""
    category: str = ""
    content: str = ""
    importance: float = 0.5
    source: str = ""
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_accessed_at: datetime | None = None
    is_deleted: bool = False

    @classmethod
    def from_entry(cls, entry: MemoryEntry) -> MemoryRecord:
        # SQLite returns naive datetimes: interpret as UTC for a stable API.
        def aware(value: datetime | None) -> datetime | None:
            if value is not None and value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value

        return cls(
            id=entry.id,
            conversation_id=entry.conversation_id or "",
            category=entry.category,
            content=entry.content,
            importance=entry.importance,
            source=entry.source,
            created_at=aware(entry.created_at),
            updated_at=aware(entry.updated_at),
            last_accessed_at=aware(entry.last_accessed_at),
            is_deleted=entry.is_deleted,
        )


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _query_variants(query: str) -> list[str]:
    """Full query plus significant words (len>=4, max 6).

    v1 has no embeddings: exact-substring LIKE on a whole sentence rarely
    matches, so word variants give deterministic keyword recall.
    """
    words: list[str] = []
    for token in query.strip().split():
        word = "".join(ch for ch in token if ch.isalnum()).lower()
        if len(word) >= 4 and word not in words:
            words.append(word)
    variants = ([query.strip()] if query.strip() else []) + words[:6]
    return variants or [""]


class MemoryManager:
    """Persistent memory: remember, search, recall, forget."""

    def __init__(
        self,
        engine: Engine,
        *,
        context_limit: int = 10,
        recent_limit: int = 10,
        context_max_chars: int = 4000,
        default_category: str = MemoryCategory.CONVERSATION.value,
        retention_days: int = 0,
    ) -> None:
        init_db(engine)
        self.engine = engine
        self.repo = MemoryRepository(engine)
        self.context_limit = max(1, context_limit)
        self.recent_limit = max(1, recent_limit)
        self.context_max_chars = max(256, context_max_chars)
        self.default_category = normalize_category(default_category)
        self.retention_days = max(0, retention_days)

    @classmethod
    def from_settings(cls, settings) -> MemoryManager:  # type: ignore[no-untyped-def]
        """Build from app Settings (wiring helper for main.py)."""
        return cls(
            create_memory_engine(settings.memory_database_path),
            context_limit=settings.memory_context_limit,
            recent_limit=settings.memory_recent_limit,
            context_max_chars=settings.memory_context_max_chars,
            default_category=settings.memory_default_category,
            retention_days=settings.memory_retention_days,
        )

    # -- write --------------------------------------------------------------
    def remember(
        self,
        content: str,
        *,
        category: str | None = None,
        conversation_id: str = "",
        importance: float = 0.5,
        source: str = "chat",
        explicit_tool_history: bool = False,
    ) -> MemoryRecord:
        """Store one memory. Raises ValueError (bad input) or
        MemoryRejected (secret material — nothing is stored)."""
        text = (content or "").strip()
        if not text:
            raise ValueError("Memory content must not be empty.")
        resolved_category = normalize_category(category or self.default_category)
        if (
            resolved_category == MemoryCategory.TOOL_HISTORY.value
            and not explicit_tool_history
        ):
            raise ValueError(
                "tool_history memories require an explicit request "
                "(explicit_tool_history=True)."
            )
        verdict = evaluate(text)
        if verdict.decision == "reject":
            raise MemoryRejected(
                "I can't store that — it looks like secret material "
                f"({', '.join(verdict.reasons)}). Secrets are never kept in memory."
            )
        if self.retention_days:
            self.purge_expired()
        entry = self.repo.add(
            MemoryEntry(
                conversation_id=conversation_id,
                category=resolved_category,
                content=verdict.text,
                importance=min(1.0, max(0.0, importance)),
                source=source,
            )
        )
        return MemoryRecord.from_entry(entry)

    # -- read -----------------------------------------------------------------
    def get(self, entry_id: int) -> MemoryRecord | None:
        """Fetch one live entry (updates last-accessed). None if missing/deleted."""
        entry = self.repo.get_by_id(entry_id)
        if entry is None:
            return None
        self.repo.touch(entry_id, utcnow())
        entry.last_accessed_at = utcnow()
        return MemoryRecord.from_entry(entry)

    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        conversation_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryRecord]:
        """Keyword search (empty query = importance/recency listing)."""
        resolved = normalize_category(category) if category else None
        entries = self.repo.search(
            query.strip(),
            category=resolved,
            conversation_id=conversation_id,
            limit=max(1, limit),
        )
        return [MemoryRecord.from_entry(e) for e in entries]

    def get_recent(
        self, conversation_id: str, *, limit: int | None = None
    ) -> list[MemoryRecord]:
        """Newest entries of one conversation (any category)."""
        entries = self.repo.recent(
            conversation_id, limit=limit or self.recent_limit
        )
        return [MemoryRecord.from_entry(e) for e in entries]

    def get_context(
        self, conversation_id: str, *, query: str = "", limit: int | None = None
    ) -> list[MemoryRecord]:
        """Deterministic recall for prompt injection: recent same-conversation
        turns (chronological) + global knowledge matches (importance/recency)."""
        cap = limit or self.context_limit
        turns = self.repo.recent(conversation_id, limit=self.recent_limit)
        known: list[MemoryEntry] = []
        per_cat = max(1, cap // max(1, len(GLOBAL_CATEGORIES)))
        for variant in _query_variants(query):
            for cat in GLOBAL_CATEGORIES:
                known.extend(
                    self.repo.search(
                        variant, category=cat, limit=per_cat * 2
                    )
                )
        # Merge + deterministic sort, same-conversation turns excluded from
        # the global pool (already included chronologically below).
        seen_turns = {t.id for t in turns}
        deduped: dict[int, MemoryEntry] = {}
        for e in known:
            if e.id not in seen_turns:
                deduped.setdefault(e.id or 0, e)
        pool = list(deduped.values())
        pool.sort(
            key=lambda e: (
                -(e.importance or 0.0),
                -(e.updated_at.timestamp() if e.updated_at else 0.0),
                -(e.id or 0),
            )
        )
        records = [MemoryRecord.from_entry(t) for t in reversed(turns)]
        records.extend(MemoryRecord.from_entry(e) for e in pool[:cap])
        return records[: cap + len(turns)]

    def format_context(self, records: list[MemoryRecord]) -> str:
        """Render records as one bounded prompt section ('' when empty)."""
        if not records:
            return ""
        lines = ["Relevant remembered context:"]
        used = 0
        for rec in records:
            line = f"- [{rec.category}] {rec.content}"
            if used + len(line) > self.context_max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines) if len(lines) > 1 else ""

    def count(
        self, *, conversation_id: str | None = None, category: str | None = None
    ) -> int:
        resolved = normalize_category(category) if category else None
        return self.repo.count(conversation_id=conversation_id, category=resolved)

    # -- delete -----------------------------------------------------------------
    def forget(self, entry_id: int, *, conversation_id: str | None = None) -> bool:
        """Soft-delete one entry (optionally scoped to a conversation)."""
        return self.repo.soft_delete(entry_id, conversation_id=conversation_id)

    def clear_conversation(self, conversation_id: str) -> int:
        """Soft-delete every entry of one conversation. Returns count."""
        return self.repo.clear_conversation(conversation_id)

    def purge_expired(self) -> int:
        """Hard-delete rows older than the retention window (0 = disabled)."""
        if not self.retention_days:
            return 0
        cutoff = utcnow() - timedelta(days=self.retention_days)
        return self.repo.purge_older_than(cutoff)


__all__ = [
    "ALL_CATEGORIES",
    "GLOBAL_CATEGORIES",
    "MemoryCategory",
    "MemoryManager",
    "MemoryRecord",
    "normalize_category",
    "utcnow",
]
