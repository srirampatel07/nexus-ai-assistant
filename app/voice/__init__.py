"""NEXUS voice interface package (Phase 4).

Backends are replaceable: see `factories.create_stt/create_tts`.
"""

from app.voice.factories import (
    create_audio_config,
    create_recorder,
    create_stt,
    create_tts,
)
from app.voice.session import VoiceSession

__all__ = [
    "VoiceSession",
    "create_audio_config",
    "create_recorder",
    "create_stt",
    "create_tts",
]
