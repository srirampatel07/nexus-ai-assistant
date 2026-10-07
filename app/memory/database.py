"""SQLite engine/session factory for NEXUS memory (Phase 3).

Session-per-operation: every repository call opens a short session and
closes it before returning. Sessions are never shared between threads
(`check_same_thread=False` + immediate close keeps this safe).
`init_db()` creates tables if missing and never touches existing data.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.memory.models import Base


def database_path_from_settings(db_path: str) -> Path:
    """Resolve the SQLite file path, creating parent directories as needed."""
    path = Path(db_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def create_memory_engine(db_path: str) -> Engine:
    """Create a SQLite engine for the given database file path."""
    path = database_path_from_settings(db_path)
    return create_engine(f"sqlite:///{path}", connect_args={"check_same_thread": False})


def init_db(engine: Engine) -> None:
    """Create memory tables if they do not exist. Never alters existing data."""
    Base.metadata.create_all(engine, checkfirst=True)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Yield one short-lived session: commit on success, rollback on error."""
    factory: sessionmaker[Session] = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
