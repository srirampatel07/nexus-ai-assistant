"""Vision backend factories (Phase 5.0; local mode added later).

One decision point: `create_describer(settings)`.

- `vision_provider=local` -> fully local vision. Requires an explicit
  loopback VISION_BASE_URL (anything else is refused so images can never
  reach a hosted service) and a VISION_MODEL id. The key stays empty.
  Missing pieces raise explicit errors — never a silent downgrade.
- Otherwise (default `openai-compatible`) -> hosted behavior, unchanged:
  echo provider, missing key/URL -> MetadataDescriber (offline, honest);
  else OpenRouterVisionDescriber with VISION_MODEL and VISION_API_KEY or
  AI_API_KEY.
- Vision disabled -> MetadataDescriber would be misleading, so callers
  check `settings.vision_enabled` first and refuse with guidance.

VISION_MODEL is always independent of AI_MODEL.
"""

from __future__ import annotations

from urllib.parse import urlsplit

from app.core.config import Settings
from app.vision.describer import (
    MetadataDescriber,
    OpenRouterVisionDescriber,
    VisionDescriber,
)
from app.vision.errors import VisionUnavailableError

__all__ = ["create_describer", "is_local_url"]

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def is_local_url(raw: str) -> bool:
    """True only for loopback endpoints (local vision servers)."""
    try:
        host = (urlsplit((raw or "").strip()).hostname or "").lower()
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


def create_describer(settings: Settings) -> VisionDescriber:
    """Build the configured vision describer (offline fallback included)."""
    mode = settings.vision_provider.strip().lower()
    if mode == "local":
        return _create_local_describer(settings)
    if mode in ("openai-compatible", "openai", "openai_compatible"):
        return _create_hosted_describer(settings)
    raise VisionUnavailableError(
        f"Unknown VISION_PROVIDER '{settings.vision_provider}'. "
        "Supported: 'local', 'openai-compatible'."
    )


def _create_local_describer(settings: Settings) -> VisionDescriber:
    """Local-only vision: explicit loopback endpoint, empty key, no fallback."""
    base_url = settings.vision_base_url.strip()
    if not base_url:
        raise VisionUnavailableError(
            "Local vision needs VISION_BASE_URL "
            "(e.g. http://127.0.0.1:8081/v1). It is not set."
        )
    if not is_local_url(base_url):
        raise VisionUnavailableError(
            f"Local vision refuses non-loopback endpoint {base_url!r}: "
            "images must never be sent to a hosted service in local mode."
        )
    model = settings.vision_model.strip()
    if not model:
        raise VisionUnavailableError(
            "Local vision needs VISION_MODEL (the local model id)."
        )
    return OpenRouterVisionDescriber(
        base_url=base_url,
        model=model,
        api_key=settings.vision_api_key.get_secret_value(),
        timeout=settings.ai_timeout_seconds,
    )


def _create_hosted_describer(settings: Settings) -> VisionDescriber:
    """Hosted vision behavior (unchanged from before local mode existed)."""
    provider = settings.ai_provider.strip().lower()
    key = (
        settings.vision_api_key.get_secret_value()
        or settings.ai_api_key.get_secret_value()
    )
    base_url = (
        settings.vision_base_url.strip() or settings.ai_base_url.strip()
    )
    if provider == "echo" or not key or not base_url:
        return MetadataDescriber()
    return OpenRouterVisionDescriber(
        base_url=base_url,
        model=settings.vision_model,
        api_key=key,
        timeout=settings.ai_timeout_seconds,
    )
