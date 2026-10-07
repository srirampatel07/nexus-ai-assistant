"""Audio layer tests (Phase 4). No microphone: synthetic PCM fakes only."""

import struct
import threading
import time

import pytest

import app.voice.audio as audio_mod
from app.voice.audio import (
    AudioConfig,
    SoundDeviceRecorder,
    capture_utterance,
    check_microphone,
    chunk_rms,
)
from app.voice.errors import MicrophoneUnavailableError


def _chunk(amplitude: int, frames: int = 1600) -> bytes:
    return struct.pack(f"<{frames}h", *([amplitude] * frames))


class FakeRecorder:
    """ Scripted chunk source with open/close tracking.

    `delay` simulates wall-clock pacing: VAD timeouts are wall-time based,
    so instant fakes would spin thousands of chunks per timeout window.
    """

    def __init__(self, chunks: list[bytes], delay: float = 0.02) -> None:
        self._chunks = list(chunks)
        self._delay = delay
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read_chunk(self) -> bytes:
        time.sleep(self._delay)
        if self._chunks:
            return self._chunks.pop(0)
        return _chunk(0)

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def config():
    return AudioConfig(
        sample_rate=16000, channels=1, silence_timeout_seconds=0.2,
        max_utterance_seconds=5.0,
    )


@pytest.fixture
def fast_wait(monkeypatch):
    monkeypatch.setattr(audio_mod, "MAX_WAIT_FOR_SPEECH_SECONDS", 0.3)


def test_chunk_rms_silence():
    assert chunk_rms(_chunk(0)) == 0.0
    assert chunk_rms(b"") == 0.0


def test_chunk_rms_loud():
    assert chunk_rms(_chunk(2000)) > audio_mod.SPEECH_RMS_THRESHOLD


def test_capture_utterance_speech_then_silence(config):
    rec = FakeRecorder([_chunk(0), _chunk(3000), _chunk(3000)] + [_chunk(0)] * 10)
    pcm = capture_utterance(rec, config)
    assert pcm is not None
    assert len(pcm) >= 2 * 3200  # speech chunks kept


def test_capture_utterance_leading_silence_skipped(config):
    rec = FakeRecorder([_chunk(0)] * 3 + [_chunk(3000)] + [_chunk(0)] * 10)
    pcm = capture_utterance(rec, config)
    assert pcm is not None
    assert len(pcm) < 14 * 3200  # leading silence dropped


def test_capture_utterance_all_silence_returns_none(config, fast_wait):
    rec = FakeRecorder([_chunk(0)] * 50)
    assert capture_utterance(rec, config) is None


def test_capture_utterance_cancel_returns_none(config):
    cancel = threading.Event()
    cancel.set()
    rec = FakeRecorder([_chunk(3000)] * 50)
    assert capture_utterance(rec, config, cancel) is None


def test_capture_utterance_max_length_bounded():
    cfg = AudioConfig(max_utterance_seconds=0.25, silence_timeout_seconds=5.0)
    rec = FakeRecorder([_chunk(3000)] * 100)
    pcm = capture_utterance(rec, cfg)
    assert pcm is not None
    # Must stop near the 0.25s cap, far before exhausting 100 scripted chunks.
    assert len(pcm) <= 20 * 3200


def test_capture_utterance_propagates_device_errors(config):
    class Broken:
        def read_chunk(self):
            raise MicrophoneUnavailableError("gone")

    with pytest.raises(MicrophoneUnavailableError):
        capture_utterance(Broken(), config)  # type: ignore[arg-type]


def test_recorder_close_is_idempotent():
    rec = SoundDeviceRecorder(AudioConfig())
    rec.close()
    rec.close()


def test_recorder_read_without_open_fails():
    rec = SoundDeviceRecorder(AudioConfig())
    with pytest.raises(MicrophoneUnavailableError):
        rec.read_chunk()


def test_check_microphone_returns_tuple():
    ok, detail = check_microphone()
    assert isinstance(ok, bool)
    assert isinstance(detail, str) and detail
