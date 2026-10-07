"""Centralized exception hierarchy for NEXUS.

Phase 1: foundation only. Tool/voice/vision/IoT errors are stubbed here
so later phases can raise specific errors without rewriting imports.
"""


class NexusError(Exception):
    """Base class for all NEXUS errors."""


class ConfigError(NexusError):
    """Raised when configuration is missing or invalid."""


class ProviderError(NexusError):
    """Base class for AI provider failures."""


class ProviderUnavailable(ProviderError):
    """The AI provider could not be reached (network, timeout, 5xx)."""


class ProviderAuthError(ProviderError):
    """Missing or invalid credentials for the AI provider."""


class AssistantError(NexusError):
    """The assistant core failed to produce a response."""


class ToolError(NexusError):
    """Base class for tool failures (Phase 2+)."""


class ToolNotAvailable(ToolError):
    """Requested tool is not registered (expected in Phase 1)."""


class PermissionDenied(ToolError):
    """Permission system denied the operation (Phase 2+)."""


class UnknownTool(ToolError):
    """Requested tool name is not registered (Phase 2+)."""


class ToolValidationError(ToolError):
    """Tool arguments failed input validation (Phase 2+)."""


class ToolDenied(ToolError):
    """Tool refused to run: blocked by policy (Phase 2+)."""


class ToolTimeout(ToolError):
    """Tool exceeded its execution timeout (Phase 2+)."""


class MemoryError(NexusError):  # noqa: A001 - intentional domain name
    """Memory subsystem failure (Phase 3+)."""


class MemoryRejected(MemoryError):
    """A remember() request was refused (secret material). Nothing stored."""


class VoiceNotAvailable(NexusError):
    """Audio dependencies or devices unavailable (Phase 5+)."""
