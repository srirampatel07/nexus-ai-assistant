"""Vision backend factories (Phase 5.0).

One decision point: `create_describer(settings)`.

- Vision disabled -> MetadataDescriber would be misleading, so callers
  check `settings.vision_enabled` first and refuse with guidance.
- Echo provider, missing key/URL -> MetadataDescriber (offline, honest).
- Otherwise -> GroqVisionDescriber with VISION_MODEL + shared credentials.

No new secrets: the vision describer reuses AI_BASE_URL + AI_API_KEY.
"""

from __future__ import annotations

from app.core.config import Settings
from app.vision.describer import (
    GroqVisionDescriber,
    MetadataDescriber,
    VisionDescriber,
)

__all__ = ["create_describer"]


def create_describer(settings: Settings) -> VisionDescriber:
    """Build the configured vision describer (offline fallback included)."""
    provider = settings.ai_provider.strip().lower()
    key = settings.ai_api_key.get_secret_value()
    if provider == "echo" or not key or not settings.ai_base_url.strip():
        return MetadataDescriber()
    return GroqVisionDescriber(
        base_url=settings.ai_base_url,
        model=settings.vision_model,
        api_key=key,
        timeout=settings.ai_timeout_seconds,
    )
