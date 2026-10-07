"""STT/TTS backend + factory + config tests (Phase 4). All offline fakes."""

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
from app.voice.stt import ScriptedSTT, TranscriptionResult, VoskSTT
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


def test_vosk_missing_model_reports_setup(tmp_path):
    stt = VoskSTT(str(tmp_path / "no-model-here"))
    ok, detail = stt.is_available()
    assert ok is False
    assert "not found" in detail and "download" in detail.lower()


def test_vosk_transcribe_without_model_raises_unavailable(tmp_path):
    from app.voice.errors import STTUnavailableError

    with pytest.raises(STTUnavailableError):
        VoskSTT(str(tmp_path / "no-model-here")).transcribe(
            b"\x01\x02" * 800,
            16000,
        )


def test_vosk_empty_pcm_short_circuits(tmp_path):
    stt = VoskSTT(str(tmp_path / "no-model-here"))
    assert stt.transcribe(b"", 16000).text == ""


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
    base = dict(ai_provider="echo")
    base.update(overrides)
    return Settings(**base)


def test_create_stt_script_forces_scripted():
    assert isinstance(
        create_stt(_settings(), script=["hi"]),
        ScriptedSTT,
    )


def test_create_stt_unknown_backend_rejected():
    with pytest.raises(VoiceUnavailableError):
        create_stt(_settings(voice_stt_backend="watson-x"))


def test_create_stt_missing_model_rejected(tmp_path):
    with pytest.raises(VoiceUnavailableError) as exc_info:
        create_stt(
            _settings(
                voice_stt_model_path=str(tmp_path / "missing"),
            )
        )

    assert "Cannot load Whisper model" in str(exc_info.value)


def test_create_tts_null_and_unknown():
    assert isinstance(
        create_tts(_settings(voice_tts_backend="null")),
        NullTTS,
    )

    with pytest.raises(VoiceUnavailableError):
        create_tts(_settings(voice_tts_backend="watson-x"))


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
    s = Settings(ai_provider="echo")

    assert s.voice_enabled is True
    assert s.voice_stt_backend == "whisper"
    assert s.voice_tts_backend == "sapi"
    assert s.voice_sample_rate == 16000
    assert s.voice_language == "en"
    assert s.voice_stop_phrases
    assert s.voice_tts_rate > 0