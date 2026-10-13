"""STT/TTS backend + factory + config tests (Phase 8, NVIDIA cloud speech).

Live microphone speech-to-text uses verified NVIDIA cloud Parakeet ASR;
local AI speech models (Whisper/Vosk) remain removed. All network is
faked: no credentials, no mic, no audio devices.
"""

import pytest

from app.core.config import Settings
from app.voice.errors import (
    STTError,
    TTSError,
    VoiceCancelledError,
    VoiceUnavailableError,
)
from app.voice.factories import (
    create_audio_config,
    create_recorder,
    create_stt,
    create_tts,
)
from app.voice.stt import NvidiaSTT, ScriptedSTT, TranscriptionResult
from app.voice.tts import NullTTS


def test_transcription_result_defaults():
    r = TranscriptionResult()
    assert r.text == "" and r.confidence is None and r.duration_seconds is None


def test_scripted_stt_replays_in_order():
    stt = ScriptedSTT(["hello", "status"])
    assert stt.is_available()[0] is True
    assert stt.transcribe(b"xxx", 16000).text == "hello"
    assert stt.transcribe(b"xxx", 16000).text == "status"


def test_scripted_stt_exhaustion_ends_session():
    stt = ScriptedSTT([])
    with pytest.raises(VoiceCancelledError):
        stt.transcribe(b"xxx", 16000)


def test_local_ai_stt_classes_are_removed():
    """Whisper/Vosk model-backed STT must not exist to be (mis)used."""
    import app.voice.stt as stt_mod

    assert not hasattr(stt_mod, "WhisperSTT")
    assert not hasattr(stt_mod, "VoskSTT")
    assert not hasattr(stt_mod, "VOSK_MODEL_DOWNLOAD")


def test_removed_local_backend_refused_whisper():
    with pytest.raises(VoiceUnavailableError) as exc_info:
        create_stt(_settings(voice_stt_backend="whisper"))
    detail = str(exc_info.value).lower()
    assert "removed" in detail or "nvidia" in detail


def test_removed_local_backend_refused_vosk():
    with pytest.raises(VoiceUnavailableError) as exc_info:
        create_stt(_settings(voice_stt_backend="vosk"))
    assert "nvidia" in str(exc_info.value).lower()


def test_unknown_stt_backend_rejected_with_guidance():
    with pytest.raises(VoiceUnavailableError) as exc_info:
        create_stt(_settings(voice_stt_backend="watson-x"))
    detail = str(exc_info.value)
    assert "nvidia" in detail.lower()


def test_nvidia_stt_missing_credentials_refused():
    with pytest.raises(VoiceUnavailableError) as exc_info:
        create_stt(
            _settings(
                voice_stt_backend="nvidia",
                nvidia_api_key="",
                ai_api_key="",
            )
        )
    detail = str(exc_info.value)
    assert "NVIDIA_API_KEY" in detail
    assert "--text" in detail and "--script" in detail


def test_nvidia_stt_builds_with_key():
    stt = create_stt(
        _settings(
            voice_stt_backend="nvidia",
            nvidia_api_key="test-nvapi-key",
        )
    )
    assert isinstance(stt, NvidiaSTT)
    ok, detail = stt.is_available()
    assert ok is True
    assert "nvidia" in detail.lower()


def test_nvidia_stt_alias_parakeet():
    stt = create_stt(
        _settings(
            voice_stt_backend="parakeet",
            nvidia_api_key="test-nvapi-key",
        )
    )
    assert isinstance(stt, NvidiaSTT)


def test_null_tts_records_and_stops():
    tts = NullTTS()
    assert tts.is_available()[0] is True

    tts.speak("hi there")

    assert tts.last_spoken == "hi there"

    tts.stop()

    assert tts.stopped is True


def test_tts_error_type_exists():
    assert issubclass(TTSError, Exception)
    assert issubclass(STTError, Exception)


def _settings(**overrides):
    base = dict(
        ai_provider="echo",
        nvidia_api_key="",
        ai_api_key="",
    )
    base.update(overrides)
    return Settings(**base)


def test_create_stt_script_forces_scripted():
    assert isinstance(
        create_stt(_settings(), script=["hi"]),
        ScriptedSTT,
    )


def test_create_stt_live_backend_never_touches_microphone(tmp_path):
    """Refusal happens before any device is opened (no mic side effects)."""
    with pytest.raises(VoiceUnavailableError):
        create_stt(
            _settings(
                voice_stt_backend="whisper",
                voice_stt_model_path=str(tmp_path / "missing"),
            )
        )


def test_create_tts_null_sapi_unknown():
    assert isinstance(
        create_tts(_settings(voice_tts_backend="null")),
        NullTTS,
    )

    with pytest.raises(VoiceUnavailableError):
        create_tts(_settings(voice_tts_backend="watson-x"))

    with pytest.raises(VoiceUnavailableError):
        create_tts(
            _settings(voice_tts_backend="nvidia", nvidia_api_key="", ai_api_key="")
        )


def test_create_tts_nvidia_builds_with_key():
    from app.voice.tts import NvidiaTTS

    assert isinstance(
        create_tts(
            _settings(voice_tts_backend="nvidia", nvidia_api_key="test-nvapi-key")
        ),
        NvidiaTTS,
    )


def test_create_audio_config_maps_settings():
    cfg = create_audio_config(
        _settings(voice_sample_rate=8000)
    )

    assert cfg.sample_rate == 8000
    assert cfg.channels == 1


def test_create_recorder_type():
    from app.voice.audio import SoundDeviceRecorder

    assert isinstance(
        create_recorder(_settings()),
        SoundDeviceRecorder,
    )


def test_voice_config_defaults():
    s = Settings(
        ai_provider="echo",
        nvidia_api_key="",
        ai_api_key="",
    )

    assert s.voice_enabled is True
    assert s.voice_stt_backend == "nvidia"
    assert s.voice_tts_backend == "sapi"
    assert s.voice_sample_rate == 16000
    assert s.voice_language == "en"
    assert s.voice_stt_language == "en-US"
    assert s.voice_tts_voice
    assert s.voice_stop_phrases
    assert s.voice_tts_rate > 0
