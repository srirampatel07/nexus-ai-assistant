"""Text-to-speech abstraction (Phase 4).

TextToSpeech is a protocol. Production uses Windows SAPI through pyttsx3.
Headless/test runs can use NullTTS.

The SAPI engine is created fresh for every utterance. This avoids a Windows
pyttsx3/SAPI issue where the first utterance succeeds but subsequent
utterances on the same engine instance may produce no audio.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from app.voice.errors import TTSError, TTSUnavailableError


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
    """Windows Speech API through pyttsx3.

    Speech is generated locally and does not use a cloud API.

    A fresh pyttsx3 SAPI engine is created for every utterance because
    repeated calls on one engine instance can fail on some Windows/SAPI
    configurations.
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
        return True, "sapi (pyttsx3)"

    def speak(self, text: str) -> None:
        """Speak one utterance using a fresh SAPI engine."""
        if not text or not text.strip():
            return

        engine = None

        try:
            engine = self._pyttsx3.init("sapi5")
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)

            engine.say(text)
            engine.runAndWait()

        except Exception as exc:
            raise TTSError(
                f"Speech output failed: {exc}"
            ) from exc

        finally:
            if engine is not None:
                try:
                    engine.stop()
                except Exception:
                    pass

    def stop(self) -> None:
        """Stop any currently active engine without raising."""
        try:
            if self._engine is not None:
                self._engine.stop()
        except Exception:
            pass


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