"""Memory categories (Phase 3).

Controlled vocabulary — memory entries may only use these categories,
never arbitrary strings. `tool_history` is write-gated: only the memory
manager (explicit request or deliberate selection) may store it, never
automatic per-tool-result persistence.
"""

from __future__ import annotations

from enum import Enum


class MemoryCategory(str, Enum):
    CONVERSATION = "conversation"
    PREFERENCE = "preference"
    PROJECT = "project"
    FACT = "fact"
    SETTING = "setting"
    TOOL_HISTORY = "tool_history"


ALL_CATEGORIES: tuple[str, ...] = tuple(c.value for c in MemoryCategory)


def normalize_category(value: str) -> str:
    """Validate and normalize a category string. Raises ValueError if unknown."""
    normalized = value.strip().lower()
    if normalized not in ALL_CATEGORIES:
        raise ValueError(
            f"Unknown memory category '{value}'. Allowed: {', '.join(ALL_CATEGORIES)}."
        )
    return normalized
