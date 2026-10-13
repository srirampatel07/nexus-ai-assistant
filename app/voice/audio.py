"""Microphone capture abstraction (local device utility, not AI inference).

`AudioRecorder` is a protocol: production uses `SoundDeviceRecorder`
(PortAudio), tests inject fakes serving synthetic PCM. Nothing here imports
audio libraries at module load — native backends load lazily so text mode
and tests never break when they are missing.

Privacy: captured PCM lives in memory only. For live voice mode it is
encoded to WAV in memory and sent to the verified NVIDIA cloud STT
endpoint over HTTPS for transcription, then dropped. Nothing is written
to disk, logged, or sent anywhere else.
"""

from __future__ import annotations

import io
import math
import struct
import threading
import time
import wave
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.voice.errors import MicrophoneUnavailableError

# 16-bit mono PCM chunk length (seconds) and speech-detection threshold.
CHUNK_SECONDS = 0.1
SPEECH_RMS_THRESHOLD = 500.0
MAX_WAIT_FOR_SPEECH_SECONDS = 10.0
# Pre-threshold context kept so soft word onsets (e.g. the /w/ in "wait")
# are not clipped: number of 0.1s chunks prepended on trigger (~0.3s).
PRE_ROLL_CHUNKS = 3
# Sustained-speech gate for the interruption callback: consecutive loud
# chunks required before on_speech_start fires (blips don't stop playback).
SUSTAINED_SPEECH_CHUNKS = 2


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    silence_timeout_seconds: float = 1.2
    max_utterance_seconds: float = 15.0
    # Interruption loudness gate (RMS). Values <= 0 fall back to
    # SPEECH_RMS_THRESHOLD. Raise it if room speakers trip interruption;
    # a headset normally works at the default. Tuned via
    # VOICE_INTERRUPT_RMS_THRESHOLD.
    interrupt_rms_threshold: float = 500.0

    @property
    def frames_per_chunk(self) -> int:
        return int(self.sample_rate * CHUNK_SECONDS)


def vad_threshold(config: AudioConfig | None = None) -> float:
    """Effective loudness gate: per-session override or the default."""
    if config is not None:
        try:
            override = float(config.interrupt_rms_threshold)
        except (TypeError, ValueError, AttributeError):
            override = 0.0
        if override > 0:
            return override
    return SPEECH_RMS_THRESHOLD


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


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Encode 16-bit mono PCM as a WAV container in memory (stdlib only).

    The verified NVIDIA STT endpoint accepts WAV/OPUS/FLAC; NEXUS sends
    WAV because it is encodable without extra dependencies. Raises
    ValueError for empty audio or invalid parameters (callers map this
    to STTError, never to a local-model fallback).
    """
    if not pcm or len(pcm) < 2:
        raise ValueError("empty audio: no speech captured")
    if sample_rate <= 0:
        raise ValueError(f"invalid sample_rate {sample_rate}")
    if channels not in (1, 2):
        raise ValueError(f"invalid channels {channels}")
    if len(pcm) % 2 != 0:
        pcm = pcm[:-1]
        if len(pcm) < 2:
            raise ValueError("empty audio: no speech captured")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)
    return buf.getvalue()


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
    on_speech_start=None,
) -> bytes | None:
    """Record one utterance: skip leading silence, stop after trailing silence.

    Returns raw PCM, or None when no speech arrived (silence/cancel/timeout).
    Bounded by MAX_WAIT_FOR_SPEECH_SECONDS + max_utterance_seconds.

    A short pre-threshold context (~0.3s) is prepended so soft word onsets
    are not clipped. When `on_speech_start` is given, it is called once as
    soon as sustained loud speech is confirmed (a few consecutive loud
    chunks); it must be fast and never raise — callers use it to stop TTS
    playback promptly while capture of the full utterance continues.
    """
    from collections import deque

    stop = cancel or threading.Event()
    threshold = vad_threshold(config)
    deadline = time.monotonic() + MAX_WAIT_FOR_SPEECH_SECONDS
    recording = bytearray()
    pre_roll: deque[bytes] = deque(maxlen=PRE_ROLL_CHUNKS)
    in_speech = False
    loud_run = 0
    start_notified = False
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
        loud = chunk_rms(chunk) >= threshold
        if not in_speech:
            if loud:
                in_speech = True
                started = time.monotonic()
                recording.extend(b"".join(pre_roll))
                recording.extend(chunk)
                quiet_since = None
                loud_run = 1
            else:
                pre_roll.append(chunk)
                loud_run = 0
            continue
        recording.extend(chunk)
        now = time.monotonic()
        if loud:
            loud_run += 1
            quiet_since = None
        else:
            loud_run = 0
            quiet_since = quiet_since if quiet_since is not None else now
            if now - quiet_since >= config.silence_timeout_seconds:
                break
        if on_speech_start is not None and not start_notified and loud_run >= SUSTAINED_SPEECH_CHUNKS:
            start_notified = True
            try:
                on_speech_start()
            except Exception:
                pass
    if not in_speech or stop.is_set():
        return None
    return bytes(recording)
