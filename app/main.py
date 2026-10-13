"""App factory + info helpers (NVIDIA cloud; tools wired)."""

from __future__ import annotations

from app import __version__
from app.brain.assistant import NexusAssistant
from app.brain.provider import create_provider
from app.core.config import Settings, get_settings
from app.memory.manager import MemoryManager
from app.tools import build_default_executor, build_default_registry


def build_assistant(
    settings: Settings | None = None,
    conversation_id: str | None = None,
) -> NexusAssistant:
    """Build the assistant: NVIDIA provider, tools, and persistent memory."""
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
    ai_model = resolved.ai_model
    ai_base_url = resolved.ai_base_url
    vision_model = resolved.vision_model
    vision_base_url = resolved.vision_base_url
    # In NVIDIA cloud mode show the resolved hosted values, not local
    # leftovers (e.g. a GGUF path or :8080/:8081 URL still in AI_*/VISION_*
    # fields).
    if resolved.ai_provider.strip().lower() in (
        "nvidia",
        "nvidia-nemotron",
        "nvidia_nemotron",
        "nemotron",
    ):
        try:
            from app.brain.provider import (
                resolve_nvidia_settings,
                resolve_nvidia_vision_settings,
            )

            base_url, model, _ = resolve_nvidia_settings(resolved)
            ai_model = model
            ai_base_url = base_url
            v_base, v_model, _ = resolve_nvidia_vision_settings(resolved)
            vision_model = v_model
            vision_base_url = v_base
        except Exception:
            pass
    return {
        "name": resolved.app_name,
        "version": __version__,
        "phase": 8,
        "environment": resolved.environment,
        "ai_provider": resolved.ai_provider,
        "ai_model": ai_model,
        "ai_base_url": ai_base_url,
        "memory_enabled": resolved.memory_enabled,
        "memory_database": resolved.memory_database_path,
        "vision_enabled": resolved.vision_enabled,
        "vision_provider": resolved.vision_provider,
        "vision_model": vision_model,
        "vision_base_url": vision_base_url,
        "web_enabled": resolved.web_enabled,
        "web_search_provider": resolved.web_search_provider,
    }
