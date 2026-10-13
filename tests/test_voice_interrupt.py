"""English-only voice interruption tests (mocked; no mic, no network).

Covers: sentence chunking, control matching, prompt playback stop,
pause/resume without repeats, new-question abandonment, empty/STT-error
resume, stop-phrase exit, thread cleanup, uninterrupted speech, and
SapiTTS prompt stop. Live microphone interruption is NOT tested here.
"""

import struct
import threading
import time

import pytest

import app.voice.session as session_mod
from app.voice.audio import AudioConfig
from app.voice.errors import STTError, TTSError
from app.voice.session import (
    VoiceSession,
    match_interruption_control,
    split_reply_sentences,
)
from app.voice.stt import ScriptedSTT, TranscriptionResult
from app.voice.tts import NullTTS, SapiTTS


def _pcm(amplitude: int = 3000, frames: int = 1600) -> bytes:
    return struct.pack(f"<{frames}h", *([amplitude] * frames))


class FakeAssistant:
    def __init__(self, replies=None):
        self.replies = list(replies or ["ok"])
        self.heard: list[str] = []

    def chat(self, text: str) -> str:
        self.heard.append(text)
        return self.replies.pop(0) if self.replies else "ok"


class FakeRecorder:
    def __init__(self):
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read_chunk(self) -> bytes:
        return _pcm(0)

    def close(self) -> None:
        self.closed = True


class LiveSTT:
    """Non-scripted STT double with queued transcripts."""

    def __init__(self, texts):
        self._texts = list(texts)

    def is_available(self):
        return True, "live test double"

    def transcribe(self, pcm, rate):
        if not self._texts:
            raise STTError("test script exhausted")
        return TranscriptionResult(text=self._texts.pop(0))


class FakeTTS(NullTTS):
    """Recording TTS with optional blocking speak for stop tests."""

    def __init__(self, block: bool = False, block_first_n: int = 0):
        super().__init__()
        self.spoken: list[str] = []
        self.stop_calls = 0
        self._block = block
        self._block_first_n = block_first_n
        self._release = threading.Event()

    def speak(self, text: str) -> None:
        self.spoken.append(text)
        self.last_spoken = text
        if self._block or self._block_first_n > 0:
            if self._block_first_n > 0:
                self._block_first_n -= 1
            self._release.wait(timeout=10.0)

    def stop(self) -> None:
        self.stop_calls += 1
        self._release.set()


def _live_session(tts=None, stt=None, assistant=None, recorder=None, **kwargs):
    return VoiceSession(
        assistant=assistant or FakeAssistant(),
        stt=stt if stt is not None else LiveSTT(["hello"]),
        tts=tts or FakeTTS(),
        recorder=recorder or FakeRecorder(),
        config=AudioConfig(),
        stop_phrases=["stop listening"],
        **kwargs,
    )


@pytest.fixture
def scripted_capture(monkeypatch):
    """Patch session capture_utterance with a queued script."""
    calls = {"n": 0}
    queue: list = []

    def fake(recorder, config, cancel=None):
        calls["n"] += 1
        if queue:
            return queue.pop(0)
        return None

    monkeypatch.setattr(session_mod, "capture_utterance", fake)
    return queue


# -- chunking + control matching ------------------------------------------------

def test_split_sentences_basic():
    assert split_reply_sentences("Hello world. How are you? Fine!") == [
        "Hello world.",
        "How are you?",
        "Fine!",
    ]


def test_split_sentences_respects_limit_without_cutting_words():
    text = " ".join(["word"] * 1000)
    chunks = split_reply_sentences(text, max_chars=200)
    assert chunks
    assert all(len(c) <= 200 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_split_sentences_empty():
    assert split_reply_sentences("") == []
    assert split_reply_sentences("   ") == []


def test_match_control_pause_and_resume():
    assert match_interruption_control("wait") == "pause"
    assert match_interruption_control("please hold on a moment") == "pause"
    assert match_interruption_control("stop") == "pause"
    assert match_interruption_control("continue") == "resume"
    assert match_interruption_control("please resume now") == "resume"
    assert match_interruption_control("keep going") == "resume"


def test_match_control_ignores_long_sentences_and_noise():
    assert match_interruption_control("wait, what about tomorrow and the meeting") is None
    assert match_interruption_control("the weather is nice today") is None
    assert match_interruption_control("") is None
    assert match_interruption_control("   ") is None


# -- uninterrupted speech ---------------------------------------------------------

def test_uninterrupted_live_turn_speaks_all_sentences(scripted_capture):
    assistant = FakeAssistant(replies=["First sentence. Second sentence."])
    tts = FakeTTS()
    session = _live_session(tts=tts, assistant=assistant)
    # Empty capture script: no speech during playback -> None -> done.
    assert session._run_live_turn("hello") is True
    assert assistant.heard == ["hello"]
    assert tts.spoken == ["First sentence.", "Second sentence."]
    assert session.turns == 1
    assert tts.stop_calls == 0


# -- interruption: stop promptly ----------------------------------------------------

def test_wait_stops_playback_promptly(scripted_capture):
    tts = FakeTTS(block=True)  # speak blocks until stop() releases it
    session = _live_session(tts=tts, assistant=FakeAssistant(replies=["Answer here."]))
    scripted_capture.append(_pcm())  # interruption speech during playback
    outcome, pending = session._speak_live("Sentence one. Sentence two.")
    assert tts.stop_calls >= 1
    assert outcome in ("done", "new_turn", "end")
    # Playback thread must be gone: prompt stop, no orphan thread.
    assert session._playback_thread is None or not session._playback_thread.is_alive()


def test_wait_pauses_then_continue_resumes_tail_without_repeat(scripted_capture, capsys):
    # First speak call blocks so the interruption lands mid-playback.
    tts = FakeTTS(block_first_n=1)
    session = _live_session(
        tts=tts,
        stt=LiveSTT(["wait", "continue"]),
        assistant=FakeAssistant(replies=["One. Two. Three."]),
    )
    scripted_capture.append(_pcm())  # "wait" during playback
    scripted_capture.append(_pcm())  # "continue" while paused
    outcome, pending = session._speak_live("One. Two. Three.")
    assert outcome == "done"
    out = capsys.readouterr().out
    assert "Paused" in out
    # Every sentence spoken exactly once across stop + resume.
    assert sorted(tts.spoken) == ["One.", "Three.", "Two."]


def test_new_question_abandons_old_answer(scripted_capture):
    assistant = FakeAssistant(replies=["fresh answer."])
    # First speak blocks so the interruption lands before sentence two.
    tts = FakeTTS(block_first_n=1)
    session = _live_session(
        tts=tts,
        stt=LiveSTT(["actually, what about tomorrow"]),
        assistant=assistant,
    )
    scripted_capture.append(_pcm())  # interruption during playback
    outcome, pending = session._speak_live("Old one. Old two.")
    assert outcome == "new_turn"
    assert pending == "actually, what about tomorrow"
    assert tts.spoken == ["Old one."]  # second sentence never played


def test_empty_transcript_resumes_remaining(scripted_capture):
    tts = FakeTTS()
    session = _live_session(tts=tts, stt=LiveSTT(["   "]))
    scripted_capture.append(_pcm())  # noise blip during playback
    outcome, _ = session._speak_live("Alpha. Beta.")
    assert outcome == "done"
    assert tts.spoken == ["Alpha.", "Beta."]


def test_stt_error_during_interruption_resumes(scripted_capture):
    tts = FakeTTS()

    class BrokenSTT(LiveSTT):
        def transcribe(self, pcm, rate):
            raise STTError("boom")

    session = _live_session(tts=tts, stt=BrokenSTT([]))
    scripted_capture.append(_pcm())
    outcome, _ = session._speak_live("Alpha. Beta.")
    assert outcome == "done"
    assert tts.spoken == ["Alpha.", "Beta."]


def test_stop_phrase_during_playback_ends_session(scripted_capture):
    tts = FakeTTS(block_first_n=1)  # block so the stop lands mid-playback
    session = _live_session(tts=tts, stt=LiveSTT(["stop listening"]))
    scripted_capture.append(_pcm())
    assert session._run_live_turn("hello") is False


def test_full_live_run_ends_on_stop_phrase(scripted_capture):
    assistant = FakeAssistant(replies=["Hi there."])
    tts = FakeTTS()
    session = _live_session(
        tts=tts, stt=LiveSTT(["hello", "stop listening"]), assistant=assistant
    )
    scripted_capture.append(_pcm())  # turn 1 capture
    scripted_capture.append(None)  # no interruption during answer
    scripted_capture.append(_pcm())  # turn 2 capture
    assert session.run_forever() == 0
    assert assistant.heard == ["hello"]
    assert session.turns == 1


def test_cleanup_joins_threads_and_closes_recorder(scripted_capture):
    rec = FakeRecorder()
    tts = FakeTTS()
    session = _live_session(tts=tts, recorder=rec, stt=LiveSTT(["stop listening"]))
    scripted_capture.append(_pcm())
    assert session.run_forever() == 0
    assert rec.closed is True
    assert session._playback_thread is None or not session._playback_thread.is_alive()


def test_scripted_mode_unaffected_by_interruption_logic():
    assistant = FakeAssistant(replies=["Hello there."])
    tts = NullTTS()
    session = VoiceSession(
        assistant=assistant,
        stt=ScriptedSTT(["hello"]),
        tts=tts,
        recorder=FakeRecorder(),
        config=AudioConfig(),
        stop_phrases=["stop listening"],
    )
    assert session.run_once() is True
    assert tts.last_spoken == "Hello there."


# -- SapiTTS prompt stop --------------------------------------------------------------

class _BlockingEngine:
    """pyttsx3 engine double whose runAndWait blocks until stop()."""

    def __init__(self):
        self._stop = threading.Event()
        self.run_returned = False

    def setProperty(self, *a):
        pass

    def say(self, *a):
        pass

    def runAndWait(self):
        self._stop.wait(timeout=10.0)
        self.run_returned = True

    def stop(self):
        self._stop.set()


class _FakePyttsx3:
    def __init__(self):
        self.engines: list[_BlockingEngine] = []

    def init(self, *a):
        engine = _BlockingEngine()
        self.engines.append(engine)
        return engine


def test_sapi_stop_interrupts_blocking_playback():
    tts = SapiTTS.__new__(SapiTTS)
    tts._pyttsx3 = _FakePyttsx3()
    tts._rate = 175
    tts._volume = 1.0
    tts._engine = None
    import threading as _th

    tts._active_lock = _th.Lock()
    tts._active_engine = None

    done = threading.Event()

    def speak():
        tts.speak("hello there")
        done.set()

    thread = threading.Thread(target=speak, daemon=True)
    thread.start()
    time.sleep(0.3)  # let runAndWait block
    assert not done.is_set()
    tts.stop()  # must release runAndWait promptly
    assert done.wait(timeout=5.0), "stop() did not interrupt playback"
    thread.join(timeout=5.0)
    assert not thread.is_alive()
