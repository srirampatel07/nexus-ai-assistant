"""Reminders + local calendar tools (Jarvis ceo parity: reminders & calendar)."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


def _connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS reminders(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT NOT NULL,
        due TEXT NOT NULL DEFAULT '',
        done INTEGER NOT NULL DEFAULT 0,
        created TEXT NOT NULL DEFAULT ''
        )"""
    )
    conn.commit()
    return conn


class RemindersCreateInput(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    due: str = Field(default="", max_length=100)


class RemindersListInput(BaseModel):
    include_done: bool = False


class RemindersDoneInput(BaseModel):
    reminder_id: int = Field(ge=1)


class RemindersBase(Tool):
    category = "reminders"
    timeout_seconds = 15.0

    def __init__(self, db_path: str = "data/reminders.db") -> None:
        self.db_path = db_path


class RemindersCreateTool(RemindersBase):
    name = "reminders.create"
    description = "Create a reminder (text + optional due like 'tomorrow 9am' or ISO date)."
    risk_level = PermissionLevel.LOW
    input_model = RemindersCreateInput

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, RemindersCreateInput)
        conn = _connect(self.db_path)
        try:
            cur = conn.execute(
                "INSERT INTO reminders(text,due,created) VALUES(?,?,?)",
                (validated.text.strip(), validated.due.strip(),
                 datetime.now().isoformat(timespec="seconds")),
            )
            conn.commit()
            rid = int(cur.lastrowid)
        finally:
            conn.close()
        return ToolResult(ok=True, output={"id": rid, "text": validated.text.strip(),
                                           "due": validated.due.strip()})


class RemindersListTool(RemindersBase):
    name = "reminders.list"
    description = "List open reminders (or all with include_done=true)."
    risk_level = PermissionLevel.SAFE
    input_model = RemindersListInput

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, RemindersListInput)
        conn = _connect(self.db_path)
        try:
            if validated.include_done:
                rows = conn.execute(
                    "SELECT id,text,due,done,created FROM reminders ORDER BY id").fetchall()
            else:
                rows = conn.execute(
                    "SELECT id,text,due,done,created FROM reminders WHERE done=0 ORDER BY id").fetchall()
        finally:
            conn.close()
        return ToolResult(ok=True, output={"reminders": [
            {"id": r[0], "text": r[1], "due": r[2], "done": bool(r[3]), "created": r[4]}
            for r in rows]})
