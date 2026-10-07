"""Voice exception hierarchy (Phase 4).

All voice errors derive from `VoiceError` (a `NexusError`). Messages never
carry API keys, credentials, or raw audio — only safe diagnostics.
"""

from __future__ import annotations

from app.core.exceptions import NexusError


class VoiceError(NexusError):
    """Base class for all voice-interface errors."""


class VoiceUnavailableError(VoiceError):
    """Voice mode cannot start (backend missing, disabled, or no devices)."""


class MicrophoneUnavailableError(VoiceUnavailableError):
    """No usable microphone / capture device found or permitted."""


class STTUnavailableError(VoiceUnavailableError):
    """Speech-to-text backend cannot be used (missing engine/model)."""


class STTError(VoiceError):
    """Transcription failed for one utterance (session continues)."""


class TTSUnavailableError(VoiceUnavailableError):
    """Text-to-speech backend cannot be used."""


class TTSError(VoiceError):
    """Speaking one reply failed (session continues)."""


class VoiceCancelledError(VoiceError):
    """Session stop requested (stop phrase, Ctrl+C, script exhausted)."""
