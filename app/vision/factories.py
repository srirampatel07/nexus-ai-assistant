"""Vision backend factories (NVIDIA-only: cloud inference, no local models).

One decision point: `create_describer(settings)`.

- `AI_PROVIDER=nvidia` -> NVIDIA-hosted vision, always:
  `resolve_nvidia_vision_settings()` (VISION_API_KEY or NVIDIA_API_KEY,
  NVIDIA base URL, NVIDIA_VISION_MODEL). Loopback endpoints are never
  used; missing credentials raise explicit errors — never a silent
  fallback to another provider.
- `vision_provider=nvidia` -> same NVIDIA-hosted vision (explicit opt-in
  even when the text provider is not NVIDIA).
- `vision_provider=local` -> refused. Local model inference was removed
  under the NVIDIA-only architecture; loopback vision endpoints are never
  contacted.
- Otherwise (`openai-compatible`) -> generic hosted behavior: echo /
  missing key/URL -> MetadataDescriber (offline, honest); else a hosted
  OpenAI-compatible vision endpoint with VISION_MODEL and VISION_API_KEY
  or AI_API_KEY.
- Vision disabled -> callers check `settings.vision_enabled` first.

VISION_MODEL is always independent of AI_MODEL; in cloud mode the vision
model defaults to NVIDIA_VISION_MODEL (a vision-capable model — the
Nemotron text model does not accept images).
"""

from __future__ import annotations

from urllib.parse import urlsplit

from app.core.config import Settings
from app.core.logger import get_logger
from app.vision.describer import (
    MetadataDescriber,
    OpenRouterVisionDescriber,
    VisionDescriber,
)
from app.vision.errors import VisionUnavailableError

__all__ = ["create_describer", "is_cloud_vision", "is_local_url"]

log = get_logger("nexus.vision")

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})


def is_local_url(raw: str) -> bool:
    """True only for loopback endpoints (refused by NVIDIA cloud vision)."""
    try:
        host = (urlsplit((raw or "").strip()).hostname or "").lower()
    except ValueError:
        return False
    return host in _LOOPBACK_HOSTS


def create_describer(settings: Settings) -> VisionDescriber:
    """Build the configured vision describer (offline fallback included)."""
    if is_cloud_vision(settings):
        return _create_nvidia_describer(settings)
    mode = settings.vision_provider.strip().lower()
    if mode == "local":
        raise VisionUnavailableError(
            "Local vision was removed: NEXUS is NVIDIA-only for AI "
            "inference and never contacts loopback model endpoints. "
            "Set AI_PROVIDER=nvidia (or VISION_PROVIDER=nvidia) to use "
            "NVIDIA cloud vision."
        )
    if mode in ("openai-compatible", "openai", "openai_compatible"):
        return _create_hosted_describer(settings)
    raise VisionUnavailableError(
        f"Unknown VISION_PROVIDER '{settings.vision_provider}'. "
        "Supported: 'nvidia', 'openai-compatible'."
    )


def is_cloud_vision(settings: Settings) -> bool:
    """True when vision must use NVIDIA-hosted inference (cloud-only)."""
    if settings.vision_provider.strip().lower() in (
        "nvidia",
        "nvidia-nemotron",
        "nvidia_nemotron",
        "nemotron",
    ):
        return True
    return settings.ai_provider.strip().lower() in (
        "nvidia",
        "nvidia-nemotron",
        "nvidia_nemotron",
        "nemotron",
    )


def _create_nvidia_describer(settings: Settings) -> VisionDescriber:
    """NVIDIA-hosted vision: verified vision model, shared key/base.

    Never routes to a loopback endpoint, never falls back to local
    inference or offline metadata. Missing credentials/configuration
    raise explicit errors so callers report them honestly.
    """
    from app.brain.provider import (
        NVIDIA_DEFAULT_BASE_URL,
        NVIDIA_DEFAULT_VISION_MODEL,
        nvidia_vision_stale_override,
        resolve_nvidia_vision_settings,
    )

    stale = nvidia_vision_stale_override(settings)
    if stale:
        log.warning(
            "stale vision model override %r is retired (HTTP 410); "
            "using current default %s instead — remove the stale line "
            "from .env",
            stale,
            NVIDIA_DEFAULT_VISION_MODEL,
        )
    base_url, model, api_key = resolve_nvidia_vision_settings(settings)
    if not api_key:
        raise VisionUnavailableError(
            "NVIDIA vision needs an API key: set NVIDIA_API_KEY in .env "
            "(VISION_API_KEY or AI_API_KEY fallbacks also empty)."
        )
    if not base_url or is_local_url(base_url):
        raise VisionUnavailableError(
            f"NVIDIA vision refuses loopback endpoint {base_url!r}: "
            "cloud vision must use https://integrate.api.nvidia.com/v1."
        )
    if not model:
        raise VisionUnavailableError(
            f"NVIDIA vision model is not set (expected default "
            f"{NVIDIA_DEFAULT_VISION_MODEL})."
        )
    if not base_url:
        raise VisionUnavailableError(
            f"NVIDIA vision base URL is not set (expected default "
            f"{NVIDIA_DEFAULT_BASE_URL})."
        )
    return OpenRouterVisionDescriber(
        base_url=base_url,
        model=model,
        api_key=api_key,
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
