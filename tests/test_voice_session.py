"""VoiceSession tests (Phase 4). Fakes only: no mic, STT engine, or Groq."""

import pytest

from app.brain.assistant import NexusAssistant
from app.brain.provider import EchoProvider
from app.core.config import Settings
from app.memory.database import create_memory_engine
from app.memory.manager import MemoryManager
from app.voice.audio import AudioConfig
from app.voice.errors import MicrophoneUnavailableError, STTError, TTSError
from app.voice.session import VoiceSession
from app.voice.stt import ScriptedSTT, TranscriptionResult
from app.voice.tts import NullTTS


class FakeAssistant:
    """Minimal assistant double: scripted replies or failures."""

    def __init__(self, replies=None, error=None):
        self.replies = list(replies or ["ok"])
        self.error = error
        self.heard: list[str] = []

    def chat(self, text: str) -> str:
        self.heard.append(text)
        if self.error is not None:
            raise self.error
        return self.replies.pop(0) if self.replies else "ok"


class FakeRecorder:
    def __init__(self):
        self.opened = False
        self.closed = False

    def open(self) -> None:
        self.opened = True

    def read_chunk(self) -> bytes:
        return b"\x00\x00" * 1600

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def config():
    return AudioConfig()


def _session(stt=None, tts=None, assistant=None, recorder=None, **kwargs):
    kwargs.setdefault("stop_phrases", ["stop listening"])
    return VoiceSession(
        assistant=assistant or FakeAssistant(),
        stt=stt or ScriptedSTT(["hello"]),
        tts=tts or NullTTS(),
        recorder=recorder or FakeRecorder(),
        config=AudioConfig(),
        **kwargs,
    )


def test_successful_voice_turn():
    tts = NullTTS()
    assistant = FakeAssistant(replies=["Hello there."])
    session = _session(stt=ScriptedSTT(["hello"]), tts=tts, assistant=assistant)
    assert session.run_once() is True
    assert assistant.heard == ["hello"]
    assert tts.last_spoken == "Hello there."
    assert session.turns == 1


def test_multiple_voice_turns():
    assistant = FakeAssistant(replies=["r1", "r2"])
    session = _session(stt=ScriptedSTT(["one", "two"]), assistant=assistant)
    assert session.run_once() is True
    assert session.run_once() is True
    assert assistant.heard == ["one", "two"]
    assert session.turns == 2


def test_empty_transcription_skips_assistant():
    assistant = FakeAssistant()
    session = _session(stt=ScriptedSTT(["   "]), assistant=assistant)
    assert session.run_once() is True
    assert assistant.heard == []
    assert session.turns == 0


def test_stop_phrase_ends_session_with_goodbye():
    tts = NullTTS()
    session = _session(stt=ScriptedSTT(["please stop listening now"]), tts=tts)
    assert session.run_once() is False
    assert tts.last_spoken == "Goodbye."


def test_script_exhaustion_ends_session():
    session = _session(stt=ScriptedSTT([]))
    assert session.run_once() is False


def test_stt_failure_continues_session():
    class BrokenSTT:
        def transcribe(self, pcm, rate):
            raise STTError("boom")

    assistant = FakeAssistant()
    session = _session(stt=BrokenSTT(), assistant=assistant)  # type: ignore[arg-type]
    assert session.run_once() is True
    assert assistant.heard == []


def test_tts_failure_does_not_lose_reply():
    class BrokenTTS(NullTTS):
        def speak(self, text):
            raise TTSError("no speaker")

    assistant = FakeAssistant(replies=["answer"])
    session = _session(stt=ScriptedSTT(["q"]), tts=BrokenTTS(), assistant=assistant)
    assert session.run_once() is True
    assert assistant.heard == ["q"]


def test_assistant_failure_is_spokengracefully():
    tts = NullTTS()
    session = _session(
        stt=ScriptedSTT(["q"]), tts=tts, assistant=FakeAssistant(error=RuntimeError("x"))
    )
    assert session.run_once() is True
    assert "wrong" in (tts.last_spoken or "")


def test_stop_then_close_idempotent():
    rec = FakeRecorder()
    session = _session(recorder=rec)
    session.stop()
    session.stop()
    session.close()
    session.close()
    assert rec.closed is False  # never opened in script mode


def test_run_forever_script_mode_returns_zero_and_closes():
    rec = FakeRecorder()
    session = _session(stt=ScriptedSTT(["hi"]), recorder=rec)
    assert session.run_forever() == 0
    assert session.turns == 1


def test_run_forever_releases_recorder_on_stop():
    class LoudRecorder(FakeRecorder):
        def open(self):
            super().open()

    rec = LoudRecorder()
    session = _session(stt=ScriptedSTT([]), recorder=rec)
    assert session.run_forever() == 0


def test_keyboard_interrupt_ends_cleanly():
    rec = FakeRecorder()
    rec.open()
    calls = {"n": 0}

    class InterruptingSession(VoiceSession):
        def run_once(self):
            calls["n"] += 1
            raise KeyboardInterrupt

    session = InterruptingSession(
        assistant=FakeAssistant(),
        stt=ScriptedSTT(["hi"]),
        tts=NullTTS(),
        recorder=rec,
        config=AudioConfig(),
    )
    session._recorder_open = True
    assert session.run_forever() == 0
    assert calls["n"] == 1
    assert rec.closed is True


def test_microphone_unavailable_surfaces_cleanly():
    class DeadRecorder:
        def open(self):
            raise MicrophoneUnavailableError("no mic")

        def close(self):
            pass

    from app.voice.stt import TranscriptionResult as _T  # noqa: F401
    from app.voice.tts import NullTTS as _N  # noqa: F401

    class LiveSTT:
        def transcribe(self, pcm, rate):
            return TranscriptionResult(text="hi")

    session = _session(stt=LiveSTT(), recorder=DeadRecorder())  # type: ignore[arg-type]
    assert session.run_forever() == 2


def test_memory_integration(tmp_path):
    settings = Settings(
        ai_provider="echo", memory_database_path=str(tmp_path / "v.db")
    )
    assistant = NexusAssistant(
        settings=settings,
        provider=EchoProvider(),
        conversation_id="voice-conv",
        memory=MemoryManager(create_memory_engine(str(tmp_path / "v.db"))),
    )
    session = _session(stt=ScriptedSTT(["remember that my favorite car is red"]), assistant=assistant)
    assert session.run_once() is True
    assert assistant.memory.search("red", category="preference")


def test_tool_integration(tmp_path):
    from app.security.audit import AuditLogger
    from app.tools import build_default_registry
    from app.tools.executor import ToolExecutor
    from app.tools.schemas import LlmToolCall
    from app.brain.provider import AIProvider, ChatMessage, ChatResponse

    class ToolCallingProvider(AIProvider):
        @property
        def name(self):
            return "scripted"

        async def send_message(self, messages):
            return ChatResponse(content="plain", model="s", provider="s")

        async def stream_message(self, messages):
            yield "plain"

        async def send_message_with_tools(self, messages, tools):
            if not any(m.role == "tool" for m in messages):
                return ChatResponse(
                    content="",
                    model="s",
                    provider="s",
                    tool_calls=[LlmToolCall(id="c1", name="system.info", arguments={})],
                )
            return ChatResponse(content="Windows box.", model="s", provider="s")

        async def health_check(self):
            return True

    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
        memory_database_path=str(tmp_path / "v.db"),
    )
    registry = build_default_registry(settings)
    executor = ToolExecutor(registry=registry, audit=AuditLogger(settings.tools_audit_file))
    assistant = NexusAssistant(
        settings=settings,
        provider=ToolCallingProvider(),
        registry=registry,
        executor=executor,
        conversation_id="voice-tools",
        memory=MemoryManager(create_memory_engine(str(tmp_path / "v.db"))),
    )
    tts = NullTTS()
    session = _session(stt=ScriptedSTT(["what os am i on"]), tts=tts, assistant=assistant)
    assert session.run_once() is True
    assert "Windows" in (tts.last_spoken or "")
    assert (tmp_path / "audit.jsonl").exists()
