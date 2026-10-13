"""Text-to-speech backends (local OS utility + NVIDIA cloud option).

`TextToSpeech` is a protocol with three implementations:

- `SapiTTS` (default): local Windows OS speech synthesizer (SAPI through
  pyttsx3) — an ordinary operating-system utility, not an AI model and
  not NVIDIA speech generation.
- `NvidiaTTS` (`VOICE_TTS_BACKEND=nvidia`): verified NVIDIA hosted
  Magpie TTS (`nvidia/magpie-tts-multilingual`) via NVCF HTTPS:

      POST https://<function-id>.invocation.api.nvcf.nvidia.com/v1/audio/synthesize
      Authorization: Bearer <NVIDIA_API_KEY>
      multipart: text + language=en-US + voice + encoding + sample_rate_hz

  Verified live (HTTP 200, Oct 2026, audio/wav). Returned WAV audio is
  played locally through the OS audio device (sounddevice); the text of
  every reply is always printed, so playback failure never loses the
  answer. Cloud TTS sends reply text to NVIDIA; SAPI sends nothing.
- `NullTTS`: silent sink for headless runs and tests.

The SAPI engine is created fresh for every utterance. This avoids a Windows
pyttsx3/SAPI issue where the first utterance succeeds but subsequent
utterances on the same engine instance may produce no audio.
"""

from __future__ import annotations

import io
import threading
import wave
from typing import Protocol, runtime_checkable

from app.core.logger import get_logger
from app.voice.errors import TTSError, TTSUnavailableError

log = get_logger("nexus.voice.tts")

# Verified NVIDIA TTS invocation (public routing ID, not a secret).
# Model: nvidia/magpie-tts-multilingual.
# Function page: https://build.nvidia.com/nvidia/magpie-tts-multilingual/api
DEFAULT_TTS_FUNCTION_ID = "877104f7-e885-42b9-8de8-f6e4c6303969"
DEFAULT_TTS_MODEL = "nvidia/magpie-tts-multilingual"
DEFAULT_TTS_VOICE = "Magpie-Multilingual.EN-US.Aria"
TTS_MAX_CHARS = 2000


def default_tts_base_url(function_id: str = DEFAULT_TTS_FUNCTION_ID) -> str:
    """Derive the NVCF invocation base URL for a speech function ID."""
    return f"https://{function_id}.invocation.api.nvcf.nvidia.com"


@runtime_checkable
class TextToSpeech(Protocol):
    """Spoken output sink."""

    def is_available(self) -> tuple[bool, str]:
        """Return (usable, detail)."""
        ...

    def speak(self, text: str) -> None:
        """Speak one reply (blocking). Raises TTSError on failure."""
        ...

    def stop(self) -> None:
        """Cancel pending speech. Never raises."""
        ...


class SapiTTS:
    """Local Windows OS speech synthesis (SAPI through pyttsx3).

    This is an operating-system utility, not AI inference: speech is
    generated locally by Windows and does not use a cloud API. It is not
    NVIDIA speech generation.

    A fresh pyttsx3 SAPI engine is created for every utterance because
    repeated calls on one engine instance can fail on some Windows/SAPI
    configurations. The active engine is tracked under a lock so stop()
    from a listener thread promptly interrupts an in-progress utterance
    (used by voice interruption); sequential use is unchanged.
    """

    def __init__(self, rate: int = 175, volume: float = 1.0) -> None:
        try:
            import pyttsx3
        except ImportError as exc:
            raise TTSUnavailableError(
                "pyttsx3 is not installed. Install it with: "
                "pip install pyttsx3"
            ) from exc

        self._pyttsx3 = pyttsx3
        self._rate = rate
        self._volume = min(1.0, max(0.0, volume))
        self._engine = None
        self._active_lock = threading.Lock()
        self._active_engine = None

        # Probe SAPI during initialization so startup can report a clear
        # availability error instead of failing on the first response.
        try:
            engine = self._pyttsx3.init("sapi5")
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            self._engine = engine
        except Exception as exc:
            raise TTSUnavailableError(
                f"Speech output unavailable: {exc}"
            ) from exc
        finally:
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    pass
                self._engine = None

    def is_available(self) -> tuple[bool, str]:
        return True, "sapi (local Windows OS synthesizer; not NVIDIA)"

    def speak(self, text: str) -> None:
        """Speak one utterance using a fresh SAPI engine."""
        if not text or not text.strip():
            return

        engine = None

        try:
            engine = self._pyttsx3.init("sapi5")
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            with self._active_lock:
                self._active_engine = engine

            engine.say(text)
            engine.runAndWait()

        except Exception as exc:
            raise TTSError(
                f"Speech output failed: {exc}"
            ) from exc

        finally:
            with self._active_lock:
                if self._active_engine is engine:
                    self._active_engine = None
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def stop(self) -> None:
        """Stop any currently active engine without raising."""
        try:
            with self._active_lock:
                active = self._active_engine
            if active is not None:
                try:
                    active.stop()
                except Exception:
                    pass
        except Exception:
            pass


class NvidiaTTS:
    """NVIDIA cloud speech synthesis (Magpie via NVCF HTTPS).

    Sends reply text to the verified `/v1/audio/synthesize` endpoint and
    plays the returned WAV audio on the local audio device. Construction
    with missing credentials raises TTSUnavailableError; per-utterance
    network/API failures raise TTSError (the session prints the reply
    text and keeps listening). The API key never appears in errors/logs.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        function_id: str = DEFAULT_TTS_FUNCTION_ID,
        language: str = "en-US",
        voice: str = DEFAULT_TTS_VOICE,
        sample_rate_hz: int = 44100,
        timeout_seconds: float = 30.0,
        http_client_factory=None,
        audio_player=None,
    ) -> None:
        self._api_key = api_key or ""
        fid = (function_id or DEFAULT_TTS_FUNCTION_ID).strip()
        self._function_id = fid
        base = (base_url or "").strip().rstrip("/")
        self._base_url = base or default_tts_base_url(fid)
        self._language = (language or "en-US").strip() or "en-US"
        self._voice = (voice or DEFAULT_TTS_VOICE).strip() or DEFAULT_TTS_VOICE
        try:
            self._sample_rate_hz = int(sample_rate_hz or 44100)
        except (TypeError, ValueError):
            self._sample_rate_hz = 44100
        self._timeout = max(1.0, float(timeout_seconds or 30.0))
        self._client_factory = http_client_factory
        self._player = audio_player
        self._stop = threading.Event()

    def is_available(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, (
                "missing credentials: set NVIDIA_API_KEY in .env "
                "(nvapi-... from build.nvidia.com; AI_API_KEY fallback "
                "also empty). Spoken replies fall back to local SAPI "
                "(`VOICE_TTS_BACKEND=sapi`) or silent (`null`)."
            )
        return True, (
            f"nvidia {DEFAULT_TTS_MODEL} voice={self._voice} "
            f"(cloud TTS; {self._base_url})"
        )

    def stop(self) -> None:
        """Request cancellation of current/future speech. Never raises."""
        try:
            self._stop.set()
            try:
                import sounddevice as sd  # type: ignore

                sd.stop()
            except Exception:
                pass
        except Exception:
            pass

    def _client(self):  # type: ignore[no-untyped-def]
        if self._client_factory is not None:
            return self._client_factory(timeout=self._timeout)
        import httpx

        return httpx.Client(timeout=self._timeout)

    def _play_wav(self, wav_bytes: bytes) -> None:
        if self._player is not None:
            self._player(wav_bytes)
            return
        try:
            import sounddevice as sd  # type: ignore
        except ImportError as exc:
            raise TTSError(
                "Cannot play cloud speech: sounddevice is not installed "
                "(pip install sounddevice numpy)."
            ) from exc
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise TTSError(
                "Cannot play cloud speech: numpy is not installed "
                "(pip install numpy)."
            ) from exc
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
                channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                frames = wf.readframes(wf.getnframes())
        except Exception as exc:
            raise TTSError(f"Cloud speech returned invalid audio: {exc}") from exc
        if sampwidth != 2:
            raise TTSError(
                f"Cloud speech returned unsupported audio ({sampwidth * 8}-bit)."
            )
        try:
            audio = np.frombuffer(frames, dtype=np.int16)
            if channels > 1:
                audio = audio.reshape(-1, channels)
            sd.play(audio, samplerate=framerate)
            sd.wait()
        except Exception as exc:
            if self._stop.is_set():
                return
            raise TTSError(f"Speech playback failed: {exc}") from exc

    def synthesize(self, text: str) -> bytes:
        """Fetch WAV audio bytes for `text` (raises TTSError on failure)."""
        clean = (text or "").strip()
        if not clean:
            return b""
        if len(clean) > TTS_MAX_CHARS:
            log.warning(
                "tts text truncated to %d chars (cloud limit)", TTS_MAX_CHARS
            )
            clean = clean[:TTS_MAX_CHARS]
        if not self._api_key:
            raise TTSError(
                "Speech synthesis credentials are missing: set "
                "NVIDIA_API_KEY in .env."
            )
        url = f"{self._base_url}/v1/audio/synthesize"
        try:
            with self._client() as client:
                response = client.post(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    data={
                        "text": clean,
                        "language": self._language,
                        "voice": self._voice,
                        "encoding": "LINEAR_PCM",
                        "sample_rate_hz": str(self._sample_rate_hz),
                    },
                )
        except Exception as exc:
            kind = type(exc).__name__
            if "Timeout" in kind or "timeout" in kind.lower():
                raise TTSError(
                    f"Speech synthesis timed out after {self._timeout:g}s."
                ) from exc
            raise TTSError(
                f"Speech synthesis unreachable ({kind})."
            ) from exc
        status = getattr(response, "status_code", 0)
        if status == 200:
            content = getattr(response, "content", b"") or b""
            if len(content) < 44 or not content.startswith(b"RIFF"):
                raise TTSError("Speech synthesis returned invalid audio.")
            return bytes(content)
        if status in (401, 403):
            raise TTSError(
                f"Speech synthesis authentication failed (HTTP {status}): "
                "check NVIDIA_API_KEY in .env."
            )
        if status == 429:
            raise TTSError("Speech synthesis rate-limited (HTTP 429).")
        if status == 402:
            raise TTSError("Speech synthesis quota/payment error (HTTP 402).")
        if status == 400:
            detail = ""
            try:
                detail = f" {(response.text or '')[:200]}"
            except Exception:
                pass
            raise TTSError(f"Speech synthesis rejected the text (HTTP 400).{detail}")
        raise TTSError(f"Speech synthesis error (HTTP {status}).")

    def speak(self, text: str) -> None:
        """Synthesize and play one reply (blocking)."""
        if not text or not text.strip():
            return
        if self._stop.is_set():
            return
        wav_bytes = self.synthesize(text)
        if not wav_bytes:
            return
        if self._stop.is_set():
            return
        self._play_wav(wav_bytes)


class NullTTS:
    """Silent sink for headless runs and tests."""

    def __init__(self) -> None:
        self.last_spoken: str | None = None
        self.stopped = False

    def is_available(self) -> tuple[bool, str]:
        return True, "null sink (silent)"

    def speak(self, text: str) -> None:
        self.last_spoken = text

    def stop(self) -> None:
        self.stopped = True
