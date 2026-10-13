"""Voice conversation session (Phase 4).

`VoiceSession` is a thin interface loop around the existing
`NexusAssistant` — it contains no assistant, memory, or tool logic.
Memory and tools work automatically because the same assistant instance
is used. Confirmation-required tools are auto-denied in voice mode (the
safe default); such requests are spoken back as denials, and the user can
approve them in text mode.

Live microphone mode additionally supports a minimal English-only
interruption: while a reply is playing, saying "wait" stops playback and
pauses; "continue" resumes the unsaid sentences, a new question starts a
new turn immediately. Interruption uses the same already-open microphone
stream (playback uses the speaker device), energy VAD only — no local AI
models. Scripted/headless mode stays strictly sequential.
"""

from __future__ import annotations

import re
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

# Minimal English-only interruption controls (short utterances only; longer
# speech is treated as a new question so corrections are never swallowed).
_PAUSE_WORDS = ("wait", "hold on", "pause", "stop", "quiet")
_RESUME_WORDS = ("continue", "resume", "go on", "carry on", "keep going")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")


def split_reply_sentences(text: str, max_chars: int = 2000) -> list[str]:
    """Split a reply into speakable chunks without cutting words (English).

    Splits on sentence boundaries; over-long sentences split further on
    word boundaries to respect per-request synthesis limits.
    """
    clean = " ".join((text or "").split())
    if not clean:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(clean) if p.strip()]
    chunks: list[str] = []
    for part in parts:
        while len(part) > max_chars:
            cut = part.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        if part:
            chunks.append(part)
    return chunks


def match_interruption_control(text: str) -> str | None:
    """Return "pause"/"resume" for high-confidence English controls, else None.

    Only short utterances (at most 6 words) match, so a longer sentence
    that merely contains "wait" is handled as conversation, not a command.
    """
    normalized = " ".join((text or "").strip().lower().split())
    if not normalized or len(normalized.split()) > 6:
        return None
    for word in _PAUSE_WORDS:
        if word in normalized and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", normalized):
            return "pause"
    for word in _RESUME_WORDS:
        if word in normalized and re.search(rf"(?<!\w){re.escape(word)}(?!\w)", normalized):
            return "resume"
    return None


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
        # Interruption bookkeeping (live mic path only; scripted mode never
        # touches these). One playback worker at a time; the generation
        # counter invalidates stale audio after an interruption.
        self._play_cancel = threading.Event()
        self._phase_done = threading.Event()
        self._gen_lock = threading.Lock()
        self._generation = 0
        self._playback_thread: threading.Thread | None = None

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
        """Request a clean shutdown: no orphan mic streams or threads."""
        self._cancel.set()
        self._play_cancel.set()
        self._phase_done.set()
        self._next_generation()
        try:
            self.tts.stop()
        except Exception:
            pass
        self._join_playback()

    def _join_playback(self) -> None:
        thread = self._playback_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        self._playback_thread = None

    def _next_generation(self) -> int:
        with self._gen_lock:
            self._generation += 1
            return self._generation

    def _current_generation(self) -> int:
        with self._gen_lock:
            return self._generation

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

    # -- interruptible live playback (mic path only) ---------------------------
    def _play_sentences(
        self, sentences: list[str], generation: int, played: dict
    ) -> None:
        """Worker: speak sentences in order unless invalidated.

        Checks generation/cancel before every chunk so stale audio can
        never play after an interruption. Never raises.
        """
        count = 0
        try:
            for sentence in sentences:
                if self._cancel.is_set() or self._play_cancel.is_set():
                    break
                if generation != self._current_generation():
                    break
                print(f"NEXUS (voice)> {sentence}")
                try:
                    self.tts.speak(sentence)
                except (TTSError, Exception) as exc:
                    log.warning("tts chunk failed: %s", exc)
                    continue
                count += 1
        finally:
            played["count"] = count
            self._phase_done.set()

    def _stop_playback(self) -> None:
        """Halt active playback promptly; invalidate pending chunks."""
        self._play_cancel.set()
        self._next_generation()
        try:
            self.tts.stop()
        except Exception:
            pass
        self._join_playback()

    def _transcribe_pcm(self, pcm: bytes) -> str | None:
        """Transcribe captured audio; None on cancel, "" on skip/empty."""
        try:
            result: TranscriptionResult = self.stt.transcribe(
                pcm, self.config.sample_rate
            )
        except VoiceCancelledError:
            return None
        except STTError as exc:
            log.warning("stt failed: %s", exc)
            return ""
        return (result.text or "").strip()

    def _speak_live(self, reply: str) -> tuple[str, str | None]:
        """Speak a reply with "wait"/"continue"/new-question handling.

        Returns ("done", None) when fully spoken, ("end", None) to exit,
        or ("new_turn", text) to answer new transcribed speech at once.
        Resume replays from the interrupted sentence (never repeats fully
        played sentences; the cut-off sentence itself replays from its
        start — documented sentence granularity).
        """
        sentences = split_reply_sentences(reply)
        if not sentences:
            return "done", None
        remaining = list(sentences)
        while remaining:
            if self._cancel.is_set():
                return "end", None
            self._play_cancel.clear()
            self._phase_done.clear()
            generation = self._next_generation()
            played: dict = {}
            thread = threading.Thread(
                target=self._play_sentences,
                args=(remaining, generation, played),
                daemon=True,
            )
            self._playback_thread = thread
            thread.start()
            interrupted_pcm: bytes | None = None
            while thread.is_alive():
                try:
                    pcm = capture_utterance(
                        self.recorder, self.config, self._cancel
                    )
                except MicrophoneUnavailableError:
                    raise
                if pcm is None:
                    if not thread.is_alive():
                        break
                    continue
                interrupted_pcm = pcm
                break
            if interrupted_pcm is None:
                thread.join(timeout=5.0)
                self._playback_thread = None
                return "done", None
            # Speech while speaking: stop audio first, decide afterwards.
            self._stop_playback()
            done_count = int(played.get("count", 0))
            remaining = remaining[done_count:]
            text = self._transcribe_pcm(interrupted_pcm)
            if text is None:
                return "end", None
            if not text:
                continue  # noise remnant: resume remaining
            print(f"YOU (voice)> {text}")
            if self._is_stop_phrase(text):
                self._say("Goodbye.")
                return "end", None
            control = match_interruption_control(text)
            if control == "pause":
                outcome, pending = self._wait_paused()
                if outcome == "end":
                    return "end", None
                if outcome == "new_turn":
                    return "new_turn", pending
                continue  # resumed: replay remaining
            if control == "resume":
                continue  # nothing paused: keep speaking remaining
            return "new_turn", text
        return "done", None

    def _wait_paused(self) -> tuple[str, str | None]:
        """Paused wait: "continue" resumes, speech starts a turn, exit ends.

        Returns ("resumed", None), ("new_turn", text), or ("end", None).
        Bounded by the cancel event; Ctrl+C safe.
        """
        print("NEXUS (voice)> Paused. Say “continue” to resume, or ask away.")
        while not self._cancel.is_set():
            try:
                pcm = capture_utterance(self.recorder, self.config, self._cancel)
            except MicrophoneUnavailableError:
                raise
            if pcm is None:
                continue
            text = self._transcribe_pcm(pcm)
            if text is None:
                return "end", None
            if not text:
                continue
            print(f"YOU (voice)> {text}")
            if self._is_stop_phrase(text):
                self._say("Goodbye.")
                return "end", None
            control = match_interruption_control(text)
            if control == "resume":
                return "resumed", None
            if control == "pause":
                continue
            return "new_turn", text
        return "end", None

    def _answer_live(self, text: str) -> str | None:
        """Chat with the assistant; None only on fatal assistant errors.

        Returns the reply, or a spoken fallback notice on provider failure
        (session continues, matching run_once behavior).
        """
        try:
            return self.assistant.chat(text)
        except Exception as exc:  # never kill the session on assistant errors
            log.warning("assistant failed in voice mode: %s", exc)
            return None

    def _run_live_turn(self, first_text: str) -> bool:
        """One live-mic turn with interruptible speech. False ends session."""
        text: str | None = first_text
        while text is not None:
            if self._is_stop_phrase(text):
                self._say("Goodbye.")
                return False
            reply = self._answer_live(text)
            if reply is None:
                self._say("Sorry, something went wrong. Please try again.")
                return not self._cancel.is_set()
            self.turns += 1
            outcome, pending = self._speak_live(reply)
            if outcome == "end":
                return False
            if outcome == "new_turn" and pending:
                text = pending
                continue
            return not self._cancel.is_set()
        return not self._cancel.is_set()

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
        never a traceback). Scripted mode stays strictly sequential;
        live microphone mode speaks interruptibly ("wait"/"continue").
        """
        print("Listening... (say a stop phrase or press Ctrl+C to exit)")
        if not isinstance(self.stt, ScriptedSTT):
            print("You can interrupt me anytime — say “wait” to pause.")
        code = 0
        try:
            if isinstance(self.stt, ScriptedSTT):
                while not self._cancel.is_set():
                    try:
                        if not self.run_once():
                            break
                    except VoiceCancelledError:
                        break
            else:
                self._ensure_open()
                while not self._cancel.is_set():
                    try:
                        pcm = capture_utterance(
                            self.recorder, self.config, self._cancel
                        )
                    except VoiceCancelledError:
                        break
                    if pcm is None:
                        continue
                    text = self._transcribe_pcm(pcm)
                    if text is None:
                        break
                    if not text:
                        continue
                    print(f"YOU (voice)> {text}")
                    try:
                        if not self._run_live_turn(text):
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
