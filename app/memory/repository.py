"""Memory persistence operations (Phase 3, internal).

Every method opens a short session and closes it before returning —
sessions are never shared between threads or callers. Soft deletion only
(`is_deleted`); nothing is hard-deleted except retention purges.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.engine import Engine

from app.memory.database import session_scope
from app.memory.models import MemoryEntry


def _escape_like(text: str) -> str:
    return text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _base_query(include_deleted: bool = False):  # type: ignore[no-untyped-def]
    stmt = select(MemoryEntry)
    if not include_deleted:
        stmt = stmt.where(MemoryEntry.is_deleted.is_(False))
    return stmt


class MemoryRepository:
    """CRUD + search over `MemoryEntry` rows."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    # -- create / read ------------------------------------------------------
    def add(self, entry: MemoryEntry) -> MemoryEntry:
        with session_scope(self.engine) as session:
            session.add(entry)
            session.flush()
            session.refresh(entry)
            return entry

    def get_by_id(self, entry_id: int) -> MemoryEntry | None:
        with session_scope(self.engine) as session:
            return session.scalar(
                _base_query().where(MemoryEntry.id == entry_id)
            )

    def touch(self, entry_id: int, when: datetime) -> None:
        with session_scope(self.engine) as session:
            session.execute(
                update(MemoryEntry)
                .where(MemoryEntry.id == entry_id)
                .values(last_accessed_at=when)
            )

    # -- search / list ------------------------------------------------------
    def search(
        self,
        query: str,
        *,
        category: str | None = None,
        conversation_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        stmt = _base_query()
        if query:
            stmt = stmt.where(
                MemoryEntry.content.ilike(f"%{_escape_like(query)}%", escape="\\")
            )
        if category:
            stmt = stmt.where(MemoryEntry.category == category)
        if conversation_id:
            stmt = stmt.where(MemoryEntry.conversation_id == conversation_id)
        stmt = stmt.order_by(
            MemoryEntry.importance.desc(),
            MemoryEntry.updated_at.desc(),
            MemoryEntry.id.desc(),
        ).limit(max(1, limit))
        with session_scope(self.engine) as session:
            return list(session.scalars(stmt).all())

    def recent(
        self, conversation_id: str, *, limit: int = 10
    ) -> list[MemoryEntry]:
        stmt = (
            _base_query()
            .where(MemoryEntry.conversation_id == conversation_id)
            .order_by(MemoryEntry.created_at.desc(), MemoryEntry.id.desc())
            .limit(max(1, limit))
        )
        with session_scope(self.engine) as session:
            return list(session.scalars(stmt).all())

    def count(
        self, *, conversation_id: str | None = None, category: str | None = None
    ) -> int:
        stmt = select(func.count()).select_from(MemoryEntry).where(
            MemoryEntry.is_deleted.is_(False)
        )
        if conversation_id:
            stmt = stmt.where(MemoryEntry.conversation_id == conversation_id)
        if category:
            stmt = stmt.where(MemoryEntry.category == category)
        with session_scope(self.engine) as session:
            return int(session.scalar(stmt) or 0)

    # -- delete ---------------------------------------------------------------
    def soft_delete(self, entry_id: int, *, conversation_id: str | None = None) -> bool:
        stmt = (
            update(MemoryEntry)
            .where(MemoryEntry.id == entry_id)
            .where(MemoryEntry.is_deleted.is_(False))
            .values(is_deleted=True)
        )
        if conversation_id:
            stmt = stmt.where(MemoryEntry.conversation_id == conversation_id)
        with session_scope(self.engine) as session:
            result = session.execute(stmt)
            return result.rowcount > 0

    def clear_conversation(self, conversation_id: str) -> int:
        with session_scope(self.engine) as session:
            result = session.execute(
                update(MemoryEntry)
                .where(MemoryEntry.conversation_id == conversation_id)
                .where(MemoryEntry.is_deleted.is_(False))
                .values(is_deleted=True)
            )
            return int(result.rowcount or 0)

    def purge_older_than(self, cutoff: datetime) -> int:
        """Hard-delete rows created before `cutoff` (retention enforcement)."""
        with session_scope(self.engine) as session:
            result = session.execute(
                delete(MemoryEntry).where(MemoryEntry.created_at < cutoff)
            )
            return int(result.rowcount or 0)
