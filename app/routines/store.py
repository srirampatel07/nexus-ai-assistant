"""Scheduled routines store (Jarvis jarvis.institute parity).

SQLite-backed, stdlib only (no new deps). Each routine is a plain-language
task + interval that the runner executes via the assistant.
Catch-up: overdue routines report as due on next poll.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Routine:
    id: int
    name: str
    prompt: str
    interval_seconds: int
    next_run_epoch: float
    enabled: bool
    last_status: str = ""


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS routines(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        prompt TEXT NOT NULL,
        interval_seconds INTEGER NOT NULL,
        next_run_epoch REAL NOT NULL,
        enabled INTEGER NOT NULL DEFAULT 1,
        last_status TEXT NOT NULL DEFAULT ''
        )"""
    )
    conn.commit()
    return conn


class RoutineStore:
    def __init__(self, db_path: str = "data/routines.db") -> None:
        self.db_path = db_path

    def add(self, name: str, prompt: str, interval_seconds: int) -> Routine:
        name = (name or "").strip() or "routine"
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("prompt must not be empty")
        interval = max(60, int(interval_seconds))
        nxt = time.time() + interval
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "INSERT INTO routines(name,prompt,interval_seconds,next_run_epoch,enabled)"
                " VALUES(?,?,?,?,1)",
                (name[:80], prompt[:4000], interval, nxt),
            )
            conn.commit()
            rid = int(cur.lastrowid)
        finally:
            conn.close()
        return Routine(id=rid, name=name[:80], prompt=prompt[:4000],
                       interval_seconds=interval, next_run_epoch=nxt, enabled=True)

    def list(self) -> list[Routine]:
        conn = _connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT id,name,prompt,interval_seconds,next_run_epoch,enabled,last_status"
                " FROM routines ORDER BY id"
            ).fetchall()
        finally:
            conn.close()
        return [Routine(id=r[0], name=r[1], prompt=r[2], interval_seconds=r[3],
                        next_run_epoch=r[4], enabled=bool(r[5]), last_status=r[6]) for r in rows]

    def remove(self, routine_id: int) -> bool:
        conn = _connect(self.db_path)
        try:
            cur = conn.execute("DELETE FROM routines WHERE id=?", (int(routine_id),))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    def due(self, now: float | None = None) -> list[Routine]:
        now = time.time() if now is None else float(now)
        return [r for r in self.list() if r.enabled and r.next_run_epoch <= now]

    def mark_ran(self, routine_id: int, status: str, *, reschedule: bool = True) -> None:
        conn = _connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT interval_seconds FROM routines WHERE id=?", (int(routine_id),)
            ).fetchall()
            if not rows:
                return
            interval = int(rows[0][0])
            nxt = time.time() + interval if reschedule else time.time() + interval
            conn.execute(
                "UPDATE routines SET last_status=?, next_run_epoch=? WHERE id=?",
                ((status or "")[:500], nxt, int(routine_id)),
            )
            conn.commit()
        finally:
            conn.close()
