"""In-process event bus for NEXUS (Phase 1).

Phase 4 (API/WebSocket) will bridge these events to `/ws`.
Phase 1 keeps it dependency-free: sync pub/sub only.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable


class EventType(str, Enum):
    USER_MESSAGE = "USER_MESSAGE"
    ASSISTANT_MESSAGE = "ASSISTANT_MESSAGE"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    TOOL_FAILED = "TOOL_FAILED"
    SYSTEM_STATUS = "SYSTEM_STATUS"
    VOICE_STARTED = "VOICE_STARTED"
    VOICE_STOPPED = "VOICE_STOPPED"


@dataclass
class NexusEvent:
    type: EventType
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


Handler = Callable[[NexusEvent], None]


class EventBus:
    """Tiny synchronous event bus."""

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Handler]] = defaultdict(list)

    def subscribe(self, event_type: EventType, handler: Handler) -> None:
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: Handler) -> None:
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def publish(self, event: NexusEvent) -> int:
        """Publish to all subscribers. Returns number of handlers called."""
        called = 0
        for handler in list(self._handlers[event.type]):
            handler(event)
            called += 1
        return called


# Shared process-wide bus (Phase 1 convenience; API layer may scope its own).
event_bus = EventBus()
