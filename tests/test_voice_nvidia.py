"""NVIDIA cloud speech tests (Phase 8). All mocked: no network, no mic.

Covers the task-required cases: successful transcription, assistant
handoff, spoken output, missing credentials, provider timeouts,
malformed audio, and no local-model fallback.
"""

import io
import struct
import wave

import pytest

from app.core.config import Settings
from app.voice.audio import AudioConfig, pcm_to_wav_bytes
from app.voice.errors import STTError, TTSError, VoiceUnavailableError
from app.voice.factories import create_stt, create_tts
from app.voice.session import VoiceSession
from app.voice.stt import NvidiaSTT, ScriptedSTT
from app.voice.tts import NvidiaTTS, NullTTS


def _pcm(amplitude: int = 3000, frames: int = 1600) -> bytes:
    return struct.pack(f"<{frames}h", *([amplitude] * frames))


def _nvidia_settings(**over):
    base = dict(
        ai_provider="echo",
        nvidia_api_key="test-nvapi-key",
        ai_api_key="",
        voice_stt_backend="nvidia",
        voice_tts_backend="null",
    )
    base.update(over)
    return Settings(**base)


class _FakeResponse:
    def __init__(self, status=200, json_data=None, text="", content=b""):
        self.status_code = status
        self._json = json_data
        self.text = text
        self.content = content

    def json(self):
        if isinstance(self._json, Exception):
            raise self._json
        return self._json


class _FakeClient:
    """Minimal httpx.Client double (context manager with .post/.get)."""

    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error
        self.requests = []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, **kwargs):
        self.requests.append((url, kwargs))
        if self._error is not None:
            raise self._error
        return self._response


def _stt_factory(response=None, error=None):
    holder = {}

    def make(timeout=None):
        client = _FakeClient(response=response, error=error)
        holder["client"] = client
        return client

    make.holder = holder  # type: ignore[attr-defined]
    return make


# -- transcription -----------------------------------------------------------

def test_nvidia_transcription_success():
    factory = _stt_factory(
        response=_FakeResponse(status=200, json_data={"text": "hello nexus"})
    )
    stt = NvidiaSTT(
        api_key="test-key",
        base_url="https://stt.example",
        http_client_factory=factory,
    )
    result = stt.transcribe(_pcm(), 16000)
    assert result.text == "hello nexus"
    url, kwargs = factory.holder["client"].requests[0]  # type: ignore[attr-defined]
    assert url.endswith("/v1/audio/transcriptions")
    assert kwargs["data"]["language"] == "en-US"
    assert kwargs["files"]["file"][0] == "utterance.wav"
    # Auth header present, key never in URL or logged detail.
    assert kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert "test-key" not in url


def test_nvidia_transcription_silence_empty_text():
    """Verified live behavior: silence -> {"text": ""} (session skips)."""
    factory = _stt_factory(response=_FakeResponse(status=200, json_data={"text": ""}))
    stt = NvidiaSTT(
        api_key="k", base_url="https://stt.example", http_client_factory=factory
    )
    assert stt.transcribe(_pcm(0), 16000).text == ""


def test_nvidia_transcription_sends_wav_container():
    factory = _stt_factory(response=_FakeResponse(status=200, json_data={"text": "x"}))
    stt = NvidiaSTT(
        api_key="k", base_url="https://stt.example", http_client_factory=factory
    )
    stt.transcribe(_pcm(), 16000)
    wav = factory.holder["client"].requests[0][1]["files"]["file"][1]  # type: ignore[attr-defined]
    assert wav[:4] == b"RIFF"
    with wave.open(io.BytesIO(wav), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2


def test_transcribe_missing_credentials_raises_stt_error():
    stt = NvidiaSTT(api_key="", base_url="https://stt.example")
    with pytest.raises(STTError, match="credentials"):
        stt.transcribe(_pcm(), 16000)


def test_transcribe_malformed_audio_empty():
    factory = _stt_factory(response=_FakeResponse(status=200, json_data={"text": "x"}))
    stt = NvidiaSTT(
        api_key="k", base_url="https://stt.example", http_client_factory=factory
    )
    with pytest.raises(STTError, match="[Ee]mpty"):
        stt.transcribe(b"", 16000)
    with pytest.raises(STTError):
        stt.transcribe(b"\x00", 16000)
    # No HTTP call on malformed audio.
    assert "client" not in factory.holder  # type: ignore[attr-defined]


def test_transcribe_malformed_audio_bad_rate():
    stt = NvidiaSTT(api_key="k", base_url="https://stt.example")
    with pytest.raises(STTError, match="[Ss]ample"):
        stt.transcribe(_pcm(), 0)


def test_pcm_to_wav_rejects_empty():
    with pytest.raises(ValueError, match="[Ee]mpty"):
        pcm_to_wav_bytes(b"", 16000)
    with pytest.raises(ValueError):
        pcm_to_wav_bytes(b"\x00", 16000)


def test_nvidia_auth_failure_maps_to_stt_error():
    factory = _stt_factory(response=_FakeResponse(status=401, text="invalid key"))
    stt = NvidiaSTT(
        api_key="bad", base_url="https://stt.example", http_client_factory=factory
    )
    with pytest.raises(STTError, match="authentication"):
        stt.transcribe(_pcm(), 16000)


def test_nvidia_rate_limit_maps_to_stt_error():
    factory = _stt_factory(response=_FakeResponse(status=429, text="slow down"))
    stt = NvidiaSTT(
        api_key="k", base_url="https://stt.example", http_client_factory=factory
    )
    with pytest.raises(STTError, match="rate-limited"):
        stt.transcribe(_pcm(), 16000)


def test_nvidia_server_error_maps_to_stt_error():
    factory = _stt_factory(response=_FakeResponse(status=500, text="boom"))
    stt = NvidiaSTT(
        api_key="k", base_url="https://stt.example", http_client_factory=factory
    )
    with pytest.raises(STTError, match="upstream"):
        stt.transcribe(_pcm(), 16000)


def test_nvidia_timeout_maps_to_stt_error():
    class FakeTimeout(Exception):
        pass

    FakeTimeout.__name__ = "ReadTimeout"
    factory = _stt_factory(error=FakeTimeout("slow"))
    stt = NvidiaSTT(
        api_key="k", base_url="https://stt.example", http_client_factory=factory
    )
    with pytest.raises(STTError, match="timed out"):
        stt.transcribe(_pcm(), 16000)


def test_nvidia_network_error_maps_to_stt_error():
    factory = _stt_factory(error=ConnectionError("dns down"))
    stt = NvidiaSTT(
        api_key="k", base_url="https://stt.example", http_client_factory=factory
    )
    with pytest.raises(STTError, match="unreachable"):
        stt.transcribe(_pcm(), 16000)


def test_nvidia_malformed_response_raises():
    factory = _stt_factory(
        response=_FakeResponse(status=200, json_data=["not", "a", "dict"])
    )
    stt = NvidiaSTT(
        api_key="k", base_url="https://stt.example", http_client_factory=factory
    )
    with pytest.raises(STTError, match="[Mm]alformed"):
        stt.transcribe(_pcm(), 16000)


def test_no_local_model_fallback():
    """Creating/failing NVIDIA STT must never construct a local model."""
    import app.voice.stt as stt_mod

    assert not hasattr(stt_mod, "WhisperSTT")
    assert not hasattr(stt_mod, "VoskSTT")
    with pytest.raises(VoiceUnavailableError):
        create_stt(_nvidia_settings(voice_stt_backend="whisper"))
    with pytest.raises(VoiceUnavailableError):
        create_stt(_nvidia_settings(voice_stt_backend="vosk"))


def test_is_available_reports_missing_credentials_without_network():
    stt = NvidiaSTT(api_key="", base_url="https://stt.example")
    ok, detail = stt.is_available()
    assert ok is False
    assert "NVIDIA_API_KEY" in detail


# -- assistant handoff + spoken output ---------------------------------------

class _FakeAssistant:
    def __init__(self, reply="done"):
        self.heard: list[str] = []
        self._reply = reply

    def chat(self, text: str) -> str:
        self.heard.append(text)
        return self._reply


class _FakeRecorder:
    def open(self):
        pass

    def read_chunk(self):
        return b"\x00\x00" * 1600

    def close(self):
        pass


def test_voice_turn_hands_transcript_to_assistant_and_speaks():
    factory = _stt_factory(
        response=_FakeResponse(status=200, json_data={"text": "what time is it"})
    )
    stt = NvidiaSTT(
        api_key="k", base_url="https://stt.example", http_client_factory=factory
    )
    tts = NullTTS()
    assistant = _FakeAssistant(reply="It is noon.")
    session = VoiceSession(
        assistant=assistant,  # type: ignore[arg-type]
        stt=stt,
        tts=tts,
        recorder=_FakeRecorder(),  # type: ignore[arg-type]
        config=AudioConfig(),
        stop_phrases=["stop listening"],
    )
    # Bypass microphone capture: inject PCM directly.
    session._ensure_open = lambda: None  # type: ignore[method-assign]
    import app.voice.session as session_mod

    real_capture = session_mod.capture_utterance
    session_mod.capture_utterance = lambda *a, **k: _pcm()
    try:
        assert session.run_once() is True
    finally:
        session_mod.capture_utterance = real_capture
    assert assistant.heard == ["what time is it"]
    assert tts.last_spoken == "It is noon."
    assert session.turns == 1


def test_nvidia_tts_synthesize_success_and_speak_calls_player():
    wav = pcm_to_wav_bytes(_pcm(), 16000)
    played = {}

    factory = _stt_factory(response=_FakeResponse(status=200, content=wav))
    tts = NvidiaTTS(
        api_key="k",
        base_url="https://tts.example",
        http_client_factory=factory,
        audio_player=lambda b: played.setdefault("bytes", b),
    )
    out = tts.synthesize("Hello there.")
    assert out[:4] == b"RIFF"
    tts.speak("Hello there.")
    assert played["bytes"][:4] == b"RIFF"
    url, kwargs = factory.holder["client"].requests[0]  # type: ignore[attr-defined]
    assert url.endswith("/v1/audio/synthesize")
    assert kwargs["data"]["voice"]
    assert kwargs["headers"]["Authorization"] == "Bearer k"


def test_nvidia_tts_missing_credentials():
    tts = NvidiaTTS(api_key="", base_url="https://tts.example")
    ok, detail = tts.is_available()
    assert ok is False
    assert "NVIDIA_API_KEY" in detail
    with pytest.raises(TTSError, match="credentials"):
        tts.synthesize("hi")


def test_nvidia_tts_timeout_maps_to_tts_error():
    class FakeTimeout(Exception):
        pass

    FakeTimeout.__name__ = "ReadTimeout"
    factory = _stt_factory(error=FakeTimeout("slow"))
    tts = NvidiaTTS(
        api_key="k", base_url="https://tts.example", http_client_factory=factory
    )
    with pytest.raises(TTSError, match="timed out"):
        tts.synthesize("hello")


def test_nvidia_tts_auth_failure():
    factory = _stt_factory(response=_FakeResponse(status=403, text="bad key"))
    tts = NvidiaTTS(
        api_key="bad", base_url="https://tts.example", http_client_factory=factory
    )
    with pytest.raises(TTSError, match="authentication"):
        tts.synthesize("hello")


def test_nvidia_tts_invalid_audio_rejected():
    factory = _stt_factory(response=_FakeResponse(status=200, content=b"not-a-wav"))
    tts = NvidiaTTS(
        api_key="k", base_url="https://tts.example", http_client_factory=factory
    )
    with pytest.raises(TTSError, match="invalid audio"):
        tts.synthesize("hello")


def test_create_tts_nvidia_missing_credentials_refused():
    with pytest.raises(VoiceUnavailableError, match="NVIDIA_API_KEY"):
        create_tts(_nvidia_settings(voice_tts_backend="nvidia", nvidia_api_key=""))


def test_scripted_path_needs_no_credentials():
    stt = create_stt(_nvidia_settings(nvidia_api_key=""), script=["hi"])
    assert isinstance(stt, ScriptedSTT)
