"""App factory + info helpers (Phase 2: tools wired)."""

from __future__ import annotations

from app import __version__
from app.brain.assistant import NexusAssistant
from app.brain.provider import create_provider
from app.core.config import Settings, get_settings
from app.tools import build_default_executor, build_default_registry


def build_assistant(settings: Settings | None = None) -> NexusAssistant:
    """Build the assistant: provider via factory, tools via default registry."""
    resolved = settings or get_settings()
    provider = create_provider(resolved)
    registry = build_default_registry(resolved)
    executor = build_default_executor(resolved, registry)
    return NexusAssistant(
        settings=resolved, provider=provider, registry=registry, executor=executor
    )


def get_nexus_info(settings: Settings | None = None) -> dict:
    """Return status info (safe: secrets never included)."""
    resolved = settings or get_settings()
    return {
        "name": resolved.app_name,
        "version": __version__,
        "phase": 2,
        "environment": resolved.environment,
        "ai_provider": resolved.ai_provider,
        "ai_model": resolved.ai_model,
    }
