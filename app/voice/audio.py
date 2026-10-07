"""Microphone capture abstraction (Phase 4).

`AudioRecorder` is a protocol: production uses `SoundDeviceRecorder`
(PortAudio), tests inject fakes serving synthetic PCM. Nothing here imports
audio libraries at module load — native backends load lazily so text mode
and tests never break when they are missing.

Privacy: captured PCM lives in memory only, is handed to STT, then dropped.
Nothing is written to disk or logged.
"""

from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.voice.errors import MicrophoneUnavailableError

# 16-bit mono PCM chunk length (seconds) and speech-detection threshold.
CHUNK_SECONDS = 0.1
SPEECH_RMS_THRESHOLD = 500.0
MAX_WAIT_FOR_SPEECH_SECONDS = 10.0


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    silence_timeout_seconds: float = 1.2
    max_utterance_seconds: float = 15.0

    @property
    def frames_per_chunk(self) -> int:
        return int(self.sample_rate * CHUNK_SECONDS)


@runtime_checkable
class AudioRecorder(Protocol):
    """One utterance-oriented PCM source (16-bit mono little-endian)."""

    def open(self) -> None:
        """Acquire the device. Raises MicrophoneUnavailableError."""
        ...

    def read_chunk(self) -> bytes:
        """Return one CHUNK_SECONDS chunk. Raises MicrophoneUnavailableError."""
        ...

    def close(self) -> None:
        """Release the device. Never raises."""
        ...


def chunk_rms(pcm: bytes) -> float:
    """Root-mean-square energy of an int16 mono chunk (stdlib only)."""
    if len(pcm) < 2:
        return 0.0
    count = len(pcm) // 2
    samples = struct.unpack(f"<{count}h", pcm[: count * 2])
    return math.sqrt(sum(s * s for s in samples) / count)


class SoundDeviceRecorder:
    """PortAudio capture via `sounddevice` (lazy import)."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._stream = None

    def open(self) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise MicrophoneUnavailableError(
                "sounddevice is not installed. Install it with: "
                "pip install sounddevice numpy"
            ) from exc
        try:
            self._stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="int16",
                blocksize=self.config.frames_per_chunk,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise MicrophoneUnavailableError(
                f"Microphone unavailable: {exc}"
            ) from exc

    def read_chunk(self) -> bytes:
        if self._stream is None:
            raise MicrophoneUnavailableError("Recorder is not open.")
        try:
            data, _overflow = self._stream.read(self.config.frames_per_chunk)
            return bytes(data)
        except Exception as exc:
            raise MicrophoneUnavailableError(f"Microphone read failed: {exc}") from exc

    def close(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        finally:
            self._stream = None


def check_microphone(sample_rate: int = 16000) -> tuple[bool, str]:
    """Probe for a usable input device without opening a stream."""
    try:
        import sounddevice as sd
    except ImportError:
        return False, "sounddevice is not installed (pip install sounddevice numpy)"
    try:
        devices = sd.query_devices()
        inputs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        if not inputs:
            return False, "no input devices found"
        default = sd.default.device[0] if isinstance(sd.default.device, tuple) else None
        name = next(
            (d["name"] for d in devices if d.get("index") == default),
            inputs[0]["name"],
        )
        return True, f"input device: {name}"
    except Exception as exc:
        return False, f"device query failed: {exc}"


def capture_utterance(
    recorder: AudioRecorder,
    config: AudioConfig,
    cancel: threading.Event | None = None,
) -> bytes | None:
    """Record one utterance: skip leading silence, stop after trailing silence.

    Returns raw PCM, or None when no speech arrived (silence/cancel/timeout).
    Bounded by MAX_WAIT_FOR_SPEECH_SECONDS + max_utterance_seconds.
    """
    stop = cancel or threading.Event()
    deadline = time.monotonic() + MAX_WAIT_FOR_SPEECH_SECONDS
    recording = bytearray()
    in_speech = False
    quiet_since: float | None = None
    started = time.monotonic()

    while not stop.is_set():
        if not in_speech and time.monotonic() > deadline:
            return None
        if in_speech and (time.monotonic() - started) > config.max_utterance_seconds:
            break
        try:
            chunk = recorder.read_chunk()
        except MicrophoneUnavailableError:
            raise
        except Exception as exc:
            raise MicrophoneUnavailableError(f"Microphone read failed: {exc}") from exc
        loud = chunk_rms(chunk) >= SPEECH_RMS_THRESHOLD
        if not in_speech:
            if loud:
                in_speech = True
                started = time.monotonic()
                recording.extend(chunk)
                quiet_since = None
            continue
        recording.extend(chunk)
        now = time.monotonic()
        if loud:
            quiet_since = None
        else:
            quiet_since = quiet_since if quiet_since is not None else now
            if now - quiet_since >= config.silence_timeout_seconds:
                break
    if not in_speech or stop.is_set():
        return None
    return bytes(recording)
