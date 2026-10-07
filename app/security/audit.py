"""Audit logging for NEXUS tool execution (Phase 2).

Every tool execution produces one JSON Lines event in `logs/tools.jsonl`.
Secrets are sanitized BEFORE logging: API keys, tokens, passwords,
authorization headers, private keys, and `.env` contents must never be stored.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.logger import get_logger

log = get_logger("nexus.audit")

_SENSITIVE_KEYS = {
    "ai_api_key",
    "api_key",
    "apikey",
    "authorization",
    "password",
    "passwd",
    "pwd",
    "token",
    "secret",
    "client_secret",
    "private_key",
    "bearer",
}

_BEARER_RE = re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/=]+", re.IGNORECASE)
_LONG_TOKEN_RE = re.compile(r"\b(gsk_|sk-|xox[bap]-)[A-Za-z0-9\-_]{8,}")


def sanitize(value: Any) -> Any:
    """Recursively redact sensitive data from arbitrary structures."""
    if isinstance(value, dict):
        return {
            str(k): ("***" if str(k).lower() in _SENSITIVE_KEYS else sanitize(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        redacted = _BEARER_RE.sub("Bearer ***", value)
        return _LONG_TOKEN_RE.sub("***", redacted)
    return value


class AuditLogger:
    """Append-only JSON Lines audit log for tool executions."""

    def __init__(self, path: str | Path = "logs/tools.jsonl") -> None:
        self.path = Path(path)

    def record(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        risk_level: str,
        permission: str,
        confirmation: str,
        status: str,
        duration_ms: int,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Write one audit event. Returns the sanitized event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "arguments": sanitize(arguments),
            "risk_level": risk_level,
            "permission": permission,
            "confirmation": confirmation,
            "status": status,
            "duration_ms": duration_ms,
            "error": error,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
        except OSError as exc:
            log.warning("audit write failed: %s", exc)
        return event
