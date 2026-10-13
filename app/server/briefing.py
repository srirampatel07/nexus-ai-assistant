"""Daily briefing / Command Center data (Jarvis institute parity).

No server dependency: builds a JSON-able dict from memory + routines +
reminders + system. Served by run.py --briefing or future HTTP API.
"""

from __future__ import annotations


def build_briefing(settings, *, conversation_id: str = "") -> dict:
    from app.main import get_nexus_info

    info = get_nexus_info(settings)
    out: dict = {"nexus": info, "persona": getattr(settings, "nexus_persona", "assistant"),
                 "tone": getattr(settings, "nexus_tone", "calm"),
                 "privacy_mode": getattr(settings, "nexus_privacy_mode", "guarded")}
    # Memory counts (never content dumps — Command Center shows counts).
    try:
        if bool(getattr(settings, "memory_enabled", True)):
            from app.memory.manager import MemoryManager
            mem = MemoryManager.from_settings(settings)
            out["memory"] = {"total": mem.count(),
                             "conversation": mem.count(conversation_id=conversation_id) if conversation_id else 0}
        else:
            out["memory"] = {"total": 0}
    except Exception as exc:
        out["memory"] = {"error": str(exc)[:200]}
    # Routines due (catch-up surface).
    try:
        from app.routines.store import RoutineStore
        store = RoutineStore(getattr(settings, "routines_db_path", "data/routines.db"))
        routines = store.list()
        due = store.due()
        out["routines"] = {"total": len(routines),
                           "due": [{"id": r.id, "name": r.name} for r in due],
                           "all": [{"id": r.id, "name": r.name,
                                    "enabled": r.enabled,
                                    "last_status": (r.last_status or "")[:120]} for r in routines[:20]]}
    except Exception as exc:
        out["routines"] = {"error": str(exc)[:200]}
    # Reminders open count + sample.
    try:
        import sqlite3
        from pathlib import Path
        db = getattr(settings, "reminders_db_path", "data/reminders.db")
        if Path(db).exists():
            conn = sqlite3.connect(db)
            try:
                n = conn.execute("SELECT COUNT(*) FROM reminders WHERE done=0").fetchone()[0]
                rows = conn.execute(
                    "SELECT id,text,due FROM reminders WHERE done=0 ORDER BY id DESC LIMIT 10").fetchall()
            finally:
                conn.close()
            out["reminders"] = {"open": int(n), "items": [
                {"id": r[0], "text": r[1][:120], "due": r[2]} for r in rows]}
        else:
            out["reminders"] = {"open": 0, "items": []}
    except Exception:
        out["reminders"] = {"open": 0, "items": []}
    # Connectors configured count (no values).
    try:
        from app.connectors.base import connector_status
        st = connector_status()
        out["connectors"] = {"configured": sum(1 for c in st if c["configured"]),
                             "total": len(st),
                             "items": [{"key": c["key"], "label": c["label"],
                                        "configured": c["configured"]} for c in st]}
    except Exception:
        out["connectors"] = {"configured": 0}
    # Tool count (registry size, no definitions dumped).
    try:
        from app.tools import build_default_registry
        out["tools"] = {"count": len(build_default_registry(settings))}
    except Exception:
        out["tools"] = {"count": 0}
    # Server identity: detects two-servers-one-port split-brain.
    try:
        from app.server.api import server_identity
        out["server"] = server_identity()
    except Exception:
        pass
    return out
