"""Voice backend factories.

Maps configuration to the selected voice backends. Backends are loaded
lazily where possible. No local AI speech models are constructed here:
live STT is NVIDIA cloud Parakeet, live TTS is local SAPI by default
with an NVIDIA cloud Magpie option.
"""

from __future__ import annotations

from app.core.config import Settings
from app.voice.audio import AudioConfig, AudioRecorder, SoundDeviceRecorder
from app.voice.errors import VoiceUnavailableError
from app.voice.session import VoiceSession
from app.voice.stt import (
    DEFAULT_STT_FUNCTION_ID,
    DEFAULT_STT_MODEL,
    NvidiaSTT,
    ScriptedSTT,
    SpeechToText,
    default_stt_base_url,
)
from app.voice.tts import (
    DEFAULT_TTS_FUNCTION_ID,
    DEFAULT_TTS_VOICE,
    NvidiaTTS,
    NullTTS,
    SapiTTS,
    TextToSpeech,
    default_tts_base_url,
)

__all__ = [
    "AudioConfig",
    "VoiceSession",
    "create_audio_config",
    "create_recorder",
    "create_stt",
    "create_tts",
    "resolve_nvidia_stt_settings",
    "resolve_nvidia_tts_settings",
]

_STT_NVIDIA_ALIASES = frozenset(
    {
        "nvidia",
        "parakeet",
        "parakeet-ctc",
        "parakeet-ctc-1_1b-asr",
        "parakeet-ctc-1.1b-asr",
        "nvidia-parakeet",
    }
)
_STT_REMOVED_LOCAL = frozenset({"whisper", "faster-whisper", "vosk"})
_TTS_NVIDIA_ALIASES = frozenset(
    {"nvidia", "magpie", "magpie-tts", "magpie-tts-multilingual", "nvidia-magpie"}
)


def resolve_nvidia_stt_settings(settings: Settings) -> tuple[str, str, str, float]:
    """Resolve NVIDIA STT endpoint/key/language/timeout (never logs secrets).

    Returns (base_url, api_key, language, timeout). Key precedence:
    NVIDIA_API_KEY, else AI_API_KEY. Base URL: NVIDIA_STT_BASE_URL when
    explicitly set, else derived from the function ID.
    """
    api_key = (
        settings.nvidia_api_key.get_secret_value()
        or settings.ai_api_key.get_secret_value()
    )
    base = settings.nvidia_stt_base_url.strip().rstrip("/")
    if not base:
        fid = (
            settings.nvidia_stt_function_id.strip() or DEFAULT_STT_FUNCTION_ID
        )
        base = default_stt_base_url(fid)
    language = (settings.voice_stt_language or "en-US").strip() or "en-US"
    timeout = float(settings.voice_stt_timeout_seconds or 30.0)
    return base, api_key, language, timeout


def resolve_nvidia_tts_settings(
    settings: Settings,
) -> tuple[str, str, str, str, int, float]:
    """Resolve NVIDIA TTS endpoint/key/voice settings (never logs secrets).

    Returns (base_url, api_key, language, voice, sample_rate_hz, timeout).
    """
    api_key = (
        settings.nvidia_api_key.get_secret_value()
        or settings.ai_api_key.get_secret_value()
    )
    base = settings.nvidia_tts_base_url.strip().rstrip("/")
    if not base:
        fid = (
            settings.nvidia_tts_function_id.strip() or DEFAULT_TTS_FUNCTION_ID
        )
        base = default_tts_base_url(fid)
    language = (settings.voice_tts_language or "en-US").strip() or "en-US"
    voice = (settings.voice_tts_voice or DEFAULT_TTS_VOICE).strip() or DEFAULT_TTS_VOICE
    try:
        rate = int(settings.voice_tts_sample_rate_hz or 44100)
    except (TypeError, ValueError):
        rate = 44100
    timeout = float(settings.voice_tts_timeout_seconds or 30.0)
    return base, api_key, language, voice, rate, timeout


def create_audio_config(settings: Settings) -> AudioConfig:
    try:
        interrupt_threshold = float(settings.voice_interrupt_rms_threshold)
    except (TypeError, ValueError, AttributeError):
        interrupt_threshold = 500.0
    return AudioConfig(
        sample_rate=settings.voice_sample_rate,
        channels=settings.voice_channels,
        silence_timeout_seconds=settings.voice_silence_timeout_seconds,
        max_utterance_seconds=settings.voice_max_utterance_seconds,
        interrupt_rms_threshold=interrupt_threshold,
    )


def create_recorder(settings: Settings) -> AudioRecorder:
    return SoundDeviceRecorder(create_audio_config(settings))


def create_stt(
    settings: Settings,
    script: list[str] | None = None,
) -> SpeechToText:
    """Build the speech-to-text backend.

    `script` forces headless scripted mode (canned transcripts, no
    microphone, no model, no network). Otherwise `VOICE_STT_BACKEND`
    selects the live backend: `nvidia` (plus parakeet aliases) builds the
    verified NVIDIA cloud Parakeet STT. Legacy local-AI names
    (whisper/vosk) and unknown names raise `VoiceUnavailableError` with
    migration guidance — they are never silently mapped to a local model.
    Missing NVIDIA credentials also raise (voice refuses to start with
    guidance toward text mode instead of looping on failures).
    """

    if script is not None:
        return ScriptedSTT(script)

    backend = settings.voice_stt_backend.strip().lower()
    if backend in _STT_NVIDIA_ALIASES:
        base_url, api_key, language, timeout = resolve_nvidia_stt_settings(
            settings
        )
        if not api_key:
            raise VoiceUnavailableError(
                "Speech recognition is unavailable: NVIDIA_API_KEY is not "
                "set (AI_API_KEY fallback also empty). Set NVIDIA_API_KEY "
                "(nvapi-... from build.nvidia.com) in .env to use live "
                f"voice mode ({DEFAULT_STT_MODEL} cloud STT). "
                "Use text mode (python run.py --text) or a headless script "
                "check (python run.py --voice --script \"hello | status\"). "
                "See docs/VOICE.md."
            )
        return NvidiaSTT(
            api_key=api_key,
            base_url=base_url,
            function_id=(
                settings.nvidia_stt_function_id.strip()
                or DEFAULT_STT_FUNCTION_ID
            ),
            language=language,
            timeout_seconds=timeout,
        )

    if backend in _STT_REMOVED_LOCAL:
        raise VoiceUnavailableError(
            f"Speech recognition backend '{settings.voice_stt_backend}' was "
            "removed: NEXUS must not run local AI models. Use "
            "VOICE_STT_BACKEND=nvidia (verified NVIDIA cloud Parakeet STT, "
            "set NVIDIA_API_KEY in .env), text mode "
            "(python run.py --text), or a headless script check "
            "(python run.py --voice --script \"hello | status\"). "
            "See docs/VOICE.md."
        )

    raise VoiceUnavailableError(
        f"Unknown VOICE_STT_BACKEND '{settings.voice_stt_backend}'. "
        "Supported: 'nvidia' (live cloud STT). "
        "Use text mode (python run.py --text) or a headless script check "
        "(python run.py --voice --script \"hello | status\"). "
        "See docs/VOICE.md."
    )


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

    if backend in _TTS_NVIDIA_ALIASES:
        (
            base_url,
            api_key,
            language,
            voice,
            rate,
            timeout,
        ) = resolve_nvidia_tts_settings(settings)
        if not api_key:
            raise VoiceUnavailableError(
                "Cloud speech synthesis is unavailable: NVIDIA_API_KEY is "
                "not set (AI_API_KEY fallback also empty). Set "
                "NVIDIA_API_KEY in .env, or use VOICE_TTS_BACKEND=sapi "
                "(local Windows OS synthesizer) or 'null' (silent). "
                "See docs/VOICE.md."
            )
        return NvidiaTTS(
            api_key=api_key,
            base_url=base_url,
            function_id=(
                settings.nvidia_tts_function_id.strip()
                or DEFAULT_TTS_FUNCTION_ID
            ),
            language=language,
            voice=voice,
            sample_rate_hz=rate,
            timeout_seconds=timeout,
        )

    raise VoiceUnavailableError(
        f"Unknown VOICE_TTS_BACKEND "
        f"'{settings.voice_tts_backend}'. "
        "Supported: 'sapi' (local Windows OS synthesizer), 'nvidia' "
        "(NVIDIA cloud Magpie TTS), 'null' (silent)."
    )
