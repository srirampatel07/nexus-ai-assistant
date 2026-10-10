"""Vision exception hierarchy (Phase 5).

All vision errors derive from `VisionError` (a `NexusError`). Messages never
carry API keys, credentials, image bytes, or base64 data — only safe
diagnostics (basename, dimensions, sizes).
"""

from __future__ import annotations

from app.core.exceptions import NexusError


class VisionError(NexusError):
    """Base class for all vision-interface errors."""


class VisionUnavailableError(VisionError):
    """Vision cannot run (disabled, Pillow missing, or no vision backend)."""


class ImageRejectedError(VisionError):
    """Image refused: outside allowed roots, bad type/size, or unverifiable."""
