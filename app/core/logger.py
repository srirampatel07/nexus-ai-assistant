"""Structured logging for NEXUS (Phase 1).

Levels: DEBUG, INFO, WARNING, ERROR, SECURITY (custom, numeric 35).
Never log secrets — callers must redact API keys / passwords / tokens.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

SECURITY_LEVEL = 35
logging.addLevelName(SECURITY_LEVEL, "SECURITY")

_configured = False


def _security(self: logging.Logger, msg: object, *args: object, **kwargs: object) -> None:
    if self.isEnabledFor(SECURITY_LEVEL):
        self._log(SECURITY_LEVEL, msg, args, **kwargs)


# Attach `logger.security(...)` to all loggers (idempotent).
if not hasattr(logging.Logger, "security"):
    logging.Logger.security = _security  # type: ignore[attr-defined]


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root logging once. Safe to call multiple times."""
    global _configured
    if _configured:
        return
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=numeric, format=fmt, handlers=handlers, force=True)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger with an attached `.security()` method."""
    return logging.getLogger(name)


def reset_logging_for_tests() -> None:
    """Reset logging configuration (tests only)."""
    global _configured
    _configured = False
