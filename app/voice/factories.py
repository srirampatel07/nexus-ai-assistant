"""Voice backend factories.

Maps configuration to the selected voice backends.
Backends are loaded lazily where possible.
"""

from __future__ import annotations

from app.core.config import Settings
from app.voice.audio import AudioConfig, AudioRecorder, SoundDeviceRecorder
from app.voice.errors import VoiceUnavailableError
from app.voice.session import VoiceSession
from app.voice.stt import (
    ScriptedSTT,
    SpeechToText,
    VoskSTT,
    WhisperSTT,
)
from app.voice.tts import NullTTS, SapiTTS, TextToSpeech

__all__ = [
    "AudioConfig",
    "VoiceSession",
    "create_audio_config",
    "create_recorder",
    "create_stt",
    "create_tts",
]


def create_audio_config(settings: Settings) -> AudioConfig:
    return AudioConfig(
        sample_rate=settings.voice_sample_rate,
        channels=settings.voice_channels,
        silence_timeout_seconds=settings.voice_silence_timeout_seconds,
        max_utterance_seconds=settings.voice_max_utterance_seconds,
    )


def create_recorder(settings: Settings) -> AudioRecorder:
    return SoundDeviceRecorder(create_audio_config(settings))


def create_stt(
    settings: Settings,
    script: list[str] | None = None,
) -> SpeechToText:
    """Build the configured STT backend.

    `script` always forces headless scripted mode.
    """

    if script is not None:
        return ScriptedSTT(script)

    backend = settings.voice_stt_backend.strip().lower()

    if backend == "whisper":
        stt = WhisperSTT(
            model_path=settings.voice_stt_model_path,
            language=settings.voice_language,
            device=settings.voice_stt_device,
            compute_type=settings.voice_stt_compute_type,
        )

    elif backend == "vosk":
        stt = VoskSTT(
            settings.voice_stt_model_path,
            settings.voice_language,
        )

    else:
        raise VoiceUnavailableError(
            f"Unknown VOICE_STT_BACKEND "
            f"'{settings.voice_stt_backend}'. "
            "Supported: 'whisper', 'vosk'. "
            "(Or run with --script for headless mode.)"
        )

    ok, detail = stt.is_available()

    if not ok:
        raise VoiceUnavailableError(
            f"Speech recognition unavailable: {detail}"
        )

    return stt


def create_tts(settings: Settings) -> TextToSpeech:
    """Build the TTS backend ('null' = silent sink for headless runs)."""

    backend = settings.voice_tts_backend.strip().lower()

    if backend == "null":
        return NullTTS()

    if backend == "sapi":
        return SapiTTS(
            rate=settings.voice_tts_rate,
            volume=settings.voice_tts_volume,
        )

    raise VoiceUnavailableError(
        f"Unknown VOICE_TTS_BACKEND "
        f"'{settings.voice_tts_backend}'. "
        "Supported: 'sapi', 'null'."
    )