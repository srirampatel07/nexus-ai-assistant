"""App factory + info helpers (Phase 2: tools wired)."""

from __future__ import annotations

from app import __version__
from app.brain.assistant import NexusAssistant
from app.brain.provider import create_provider
from app.core.config import Settings, get_settings
from app.memory.manager import MemoryManager
from app.tools import build_default_executor, build_default_registry


def build_assistant(
    settings: Settings | None = None, conversation_id: str | None = None
) -> NexusAssistant:
    """Build the assistant: provider, tools, and persistent memory."""
    resolved = settings or get_settings()
    provider = create_provider(resolved)
    registry = build_default_registry(resolved)
    executor = build_default_executor(resolved, registry)
    memory = MemoryManager.from_settings(resolved) if resolved.memory_enabled else None
    return NexusAssistant(
        settings=resolved,
        provider=provider,
        registry=registry,
        executor=executor,
        conversation_id=conversation_id,
        memory=memory,
    )


def get_nexus_info(settings: Settings | None = None) -> dict:
    """Return status info (safe: secrets never included)."""
    resolved = settings or get_settings()
    return {
        "name": resolved.app_name,
        "version": __version__,
        "phase": 5,
        "environment": resolved.environment,
        "ai_provider": resolved.ai_provider,
        "ai_model": resolved.ai_model,
        "memory_enabled": resolved.memory_enabled,
        "memory_database": resolved.memory_database_path,
        "vision_enabled": resolved.vision_enabled,
        "vision_model": resolved.vision_model,
    }
