"""Voice conversation session (Phase 4).

`VoiceSession` is a thin interface loop around the existing
`NexusAssistant` — it contains no assistant, memory, or tool logic.
Memory and tools work automatically because the same assistant instance
is used. Confirmation-required tools are auto-denied in voice mode (the
safe default); such requests are spoken back as denials, and the user can
approve them in text mode.
"""

from __future__ import annotations

import threading

from app.brain.assistant import NexusAssistant
from app.core.logger import get_logger
from app.voice.audio import AudioConfig, AudioRecorder, capture_utterance
from app.voice.errors import (
    MicrophoneUnavailableError,
    STTError,
    TTSError,
    VoiceCancelledError,
)
from app.voice.stt import ScriptedSTT, SpeechToText, TranscriptionResult
from app.voice.tts import TextToSpeech

log = get_logger("nexus.voice")


class VoiceSession:
    """One microphone-driven conversation. Not thread-safe; single owner."""

    def __init__(
        self,
        assistant: NexusAssistant,
        stt: SpeechToText,
        tts: TextToSpeech,
        recorder: AudioRecorder,
        config: AudioConfig,
        stop_phrases: list[str] | None = None,
    ) -> None:
        self.assistant = assistant
        self.stt = stt
        self.tts = tts
        self.recorder = recorder
        self.config = config
        self.stop_phrases = [p.strip().lower() for p in (stop_phrases or []) if p.strip()]
        self._cancel = threading.Event()
        self._recorder_open = False
        self.turns = 0

    # -- lifecycle ------------------------------------------------------------
    def _ensure_open(self) -> None:
        if not self._recorder_open and not isinstance(self.stt, ScriptedSTT):
            self.recorder.open()
            self._recorder_open = True

    def close(self) -> None:
        """Release microphone (idempotent, never raises)."""
        try:
            if self._recorder_open:
                self.recorder.close()
        except Exception:
            pass
        finally:
            self._recorder_open = False

    def stop(self) -> None:
        """Request a clean shutdown: no orphan mic streams."""
        self._cancel.set()
        try:
            self.tts.stop()
        except Exception:
            pass

    # -- single turn ------------------------------------------------------------
    def _is_stop_phrase(self, text: str) -> bool:
        normalized = " ".join(text.strip().lower().split())
        return any(phrase and phrase in normalized for phrase in self.stop_phrases)

    def _say(self, text: str) -> None:
        print(f"NEXUS (voice)> {text}")
        try:
            self.tts.speak(text)
        except TTSError as exc:
            log.warning("tts failed: %s", exc)

    def run_once(self) -> bool:
        """Run one listen->transcribe->answer->speak turn.

        Returns True to keep listening, False to end the session.
        """
        self._ensure_open()
        if isinstance(self.stt, ScriptedSTT):
            pcm = b""
        else:
            pcm = capture_utterance(self.recorder, self.config, self._cancel)
        if pcm is None:
            return not self._cancel.is_set()
        try:
            result: TranscriptionResult = self.stt.transcribe(
                pcm, self.config.sample_rate
            )
        except VoiceCancelledError:
            return False
        except STTError as exc:
            log.warning("stt failed: %s", exc)
            return not self._cancel.is_set()
        text = (result.text or "").strip()
        if not text:
            return not self._cancel.is_set()
        print(f"YOU (voice)> {text}")
        if self._is_stop_phrase(text):
            self._say("Goodbye.")
            return False
        try:
            reply = self.assistant.chat(text)
        except Exception as exc:  # never kill the session on assistant errors
            log.warning("assistant failed in voice mode: %s", exc)
            self._say("Sorry, something went wrong. Please try again.")
            return not self._cancel.is_set()
        self.turns += 1
        self._say(reply)
        return not self._cancel.is_set()

    # -- main loop --------------------------------------------------------------
    def run_forever(self) -> int:
        """Listen until stopped. Always releases resources.

        Returns 0 on clean end, 2 when the microphone fails (with guidance,
        never a traceback).
        """
        print("Listening... (say a stop phrase or press Ctrl+C to exit)")
        code = 0
        try:
            while not self._cancel.is_set():
                try:
                    if not self.run_once():
                        break
                except VoiceCancelledError:
                    break
        except KeyboardInterrupt:
            print("\nStopping voice mode.")
        except MicrophoneUnavailableError as exc:
            print(f"Microphone failed: {exc}")
            code = 2
        finally:
            self.stop()
            self.close()
        print("Voice session ended.")
        return code
