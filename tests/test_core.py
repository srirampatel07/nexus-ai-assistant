"""Logger + events tests (Phase 1)."""

import logging

from app.core.events import EventBus, EventType, NexusEvent
from app.core.logger import SECURITY_LEVEL, get_logger, setup_logging


def test_security_level_exists():
    assert logging.getLevelName(SECURITY_LEVEL) == "SECURITY"
    logger = get_logger("nexus.test")
    assert hasattr(logger, "security")


def test_setup_logging_idempotent(tmp_path):
    setup_logging("DEBUG")
    setup_logging("DEBUG", log_file=tmp_path / "n.log")
    assert get_logger("nexus.test") is not None


def test_event_bus_pub_sub():
    bus = EventBus()
    seen: list[NexusEvent] = []
    bus.subscribe(EventType.USER_MESSAGE, seen.append)
    n = bus.publish(NexusEvent(type=EventType.USER_MESSAGE, payload={"text": "hi"}))
    assert n == 1
    assert seen[0].payload["text"] == "hi"


def test_event_bus_unsubscribe():
    bus = EventBus()
    seen: list = []
    bus.subscribe(EventType.SYSTEM_STATUS, seen.append)
    bus.unsubscribe(EventType.SYSTEM_STATUS, seen.append)
    assert bus.publish(NexusEvent(type=EventType.SYSTEM_STATUS)) == 0
