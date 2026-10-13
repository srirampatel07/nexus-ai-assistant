"""Routine runner: execute due routines via the assistant (catch-up).

Used by `run.py --routines-run`: polls due routines, chats each prompt,
marks rescheduled with truncated status. Interactive confirmations
auto-deny (safe); use text mode to approve MEDIUM+ tools.
"""

from __future__ import annotations


def run_due_routines(assistant, *, limit: int = 5) -> dict:
    from app.routines.store import RoutineStore

    db = getattr(getattr(assistant, "settings", None), "routines_db_path", "data/routines.db")
    store = RoutineStore(db)
    due = store.due()[:max(1, limit)]
    ran: list[dict] = []
    for routine in due:
        try:
            reply = assistant.chat(f"[routine:{routine.name}] {routine.prompt}")
            status = (reply or "")[:400]
        except Exception as exc:
            status = f"failed: {type(exc).__name__}"
        store.mark_ran(routine.id, status)
        ran.append({"id": routine.id, "name": routine.name, "status": status[:200]})
    return {"ran": ran, "count": len(ran)}
