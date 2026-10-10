"""Vision security-boundary regression tests (Phase 5.0 hardening).

Covers five invariants the earlier suite did not prove directly:

1. Decompression-bomb rejection before expensive decode/resize.
2. Symlink inside allowed roots resolving outside is rejected unread.
3. Missing Pillow -> safe VisionUnavailableError, text chat still imports.
4. VoiceSession.run_once() never touches the vision upload path.
5. Image bytes / base64 / data-URIs never reach audit, memory, or logs.

All offline and deterministic. No network, microphone, or real API keys.
"""

from __future__ import annotations

import base64
import logging
import sys

import pytest

pytest.importorskip("PIL")

from pathlib import Path

from PIL import Image

from app.brain.assistant import NexusAssistant
from app.brain.provider import EchoProvider
from app.core.config import Settings
from app.memory.database import create_memory_engine
from app.memory.manager import MemoryManager
from app.security.audit import AuditLogger
from app.tools import build_default_registry
from app.tools.executor import ToolExecutor
from app.vision.describer import VisualDescription
from app.vision.errors import ImageRejectedError, VisionUnavailableError
from app.vision.loader import MAX_IMAGE_PIXELS, load_validated_image
from app.voice.audio import AudioConfig
from app.voice.session import VoiceSession
from app.voice.stt import ScriptedSTT, TranscriptionResult
from app.voice.tts import NullTTS


def _make_png(path: Path, size=(48, 32), color=(20, 40, 60)) -> Path:
    Image.new("RGB", size, color).save(path, format="PNG")
    return path


def _loader_kwargs(tmp_path: Path, **over) -> dict:
    kwargs = dict(
        allowed_roots=[str(tmp_path)],
        max_bytes=10_485_760,
        max_dim=2048,
        jpeg_quality=85,
    )
    kwargs.update(over)
    return kwargs


# -- 1. Decompression-bomb rejection ----------------------------------------

_BOMB_W, _BOMB_H = 8000, 8000  # 64 MP > 50 MP guard; nothing allocated


class _BombImage:
    """Fake Pillow image: huge header dimensions, zero pixel allocation."""

    size = (_BOMB_W, _BOMB_H)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def verify(self) -> None:
        return None

    def load(self) -> None:
        return None


def test_oversized_dimensions_rejected_before_decode(tmp_path, monkeypatch):
    """Guard fires on header dimensions; no resize/encode is ever reached."""
    import PIL.Image

    src = _make_png(tmp_path / "bomb.png")  # tiny real file: stat passes
    opened = {"n": 0}

    def fake_open(target, *args, **kwargs):
        opened["n"] += 1
        return _BombImage()

    monkeypatch.setattr(PIL.Image, "open", fake_open)
    with pytest.raises(ImageRejectedError) as exc:
        load_validated_image(str(src), **_loader_kwargs(tmp_path))
    assert "too many pixels" in str(exc.value).lower()
    # Safe diagnostics only: dimensions are fine, contents/secrets never are.
    assert f"{_BOMB_W}x{_BOMB_H}" in str(exc.value)
    assert len(str(exc.value)) < 500
    # Both phases attempted (verify probe + processing open); the second
    # raised before any flatten/resize/encode could run.
    assert opened["n"] == 2
    assert MAX_IMAGE_PIXELS == 50_000_000


# -- 2. Symlink escape -------------------------------------------------------

def test_symlink_inside_resolving_outside_is_rejected(tmp_path):
    """A link placed inside roots but pointing outside must be refused unread."""
    pytest.importorskip("PIL")
    inner = tmp_path / "in"
    outer = tmp_path / "out"
    inner.mkdir()
    outer.mkdir()
    real = _make_png(outer / "secret.png")
    link = inner / "pic.png"
    try:
        link.symlink_to(real, target_is_directory=False)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")
    if not link.is_symlink():
        pytest.skip("symlink creation reported success but link is missing")
    with pytest.raises(ImageRejectedError) as excinfo:
        load_validated_image(str(link), **_loader_kwargs(tmp_path, allowed_roots=[str(inner)]))
    message = str(excinfo.value)
    assert "outside" in message.lower()
    # Basename-only diagnostics: the link name may appear, but the real
    # outside directory and file contents must not.
    assert str(outer) not in message
    assert str(real) not in message


def test_confine_rejects_symlink_style_resolved_outside(tmp_path, monkeypatch):
    """Deterministic symlink simulation: candidate resolves outside roots.

    Runs everywhere (no symlink privilege needed) and exercises the real
    `_confine` resolve-then-check path exactly as the OS would for a link.
    """
    from app.vision.loader import _confine

    inner = tmp_path / "in"
    inner.mkdir()
    (tmp_path / "out").mkdir()
    link = inner / "pic.png"
    link.touch()
    real_resolve = Path.resolve

    def fake_resolve(self, *args, **kwargs):
        resolved = real_resolve(self, *args, **kwargs)
        if resolved == real_resolve(link):
            return tmp_path / "out" / "secret.png"
        return resolved

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    with pytest.raises(ImageRejectedError) as excinfo:
        _confine(str(link), [str(inner)])
    assert "outside" in str(excinfo.value).lower()


# -- 3. Pillow missing --------------------------------------------------------

def test_missing_pillow_reports_safe_unavailable(monkeypatch):
    """Loader degrades to VisionUnavailableError, never raw ImportError."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(VisionUnavailableError) as excinfo:
        load_validated_image(
            "whatever.png",
            allowed_roots=["C:\\chanti\\nexus"],
            max_bytes=10_485_760,
            max_dim=2048,
        )
    assert "pillow" in str(excinfo.value).lower()


def test_text_chat_imports_without_pillow(monkeypatch):
    """Lazy-import design: core chat path imports even when PIL is blocked."""
    monkeypatch.setitem(sys.modules, "PIL", None)
    import importlib

    import app.brain.assistant as assistant_mod

    importlib.reload(assistant_mod)
    assert hasattr(assistant_mod, "NexusAssistant")
    # Echo text path needs no imaging at all.
    settings = Settings(ai_provider="echo", ai_model="test")
    assistant = assistant_mod.NexusAssistant(
        settings=settings, provider=EchoProvider()
    )
    assert "Hello" in assistant.chat("hello")


# -- 4. Voice never uploads ---------------------------------------------------

class _NeverUploadDescriber:
    """Fails loudly if the voice path ever touches vision."""

    def __init__(self):
        self.calls = 0

    async def describe(self, *args, **kwargs):
        self.calls += 1
        raise AssertionError("voice path must never call vision describe()")


class _FakeRecorder:
    def open(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_voice_run_once_uses_text_chat_only(tmp_path):
    """run_once() routes speech via chat(); vision upload count stays zero."""
    from app.voice.stt import ScriptedSTT

    settings = Settings(
        ai_provider="echo",
        ai_model="test",
        nexus_allowed_roots=[str(tmp_path)],
    )
    never = _NeverUploadDescriber()
    assistant = NexusAssistant(
        settings=settings, provider=EchoProvider(), vision_describer=never
    )
    # Belt and braces: even a direct method call from voice code would fail.
    assistant.chat_with_image = lambda *a, **k: (_ for _ in ()).throw(  # type: ignore[method-assign]
        AssertionError("voice path must never call chat_with_image()")
    )
    tts = NullTTS()
    session = VoiceSession(
        assistant=assistant,
        stt=ScriptedSTT(["hello"]),
        tts=tts,
        recorder=_FakeRecorder(),
        config=AudioConfig(),
        stop_phrases=["stop listening"],
    )
    assert session.run_once() is True
    assert never.calls == 0
    assert session.turns == 1
    assert tts.last_spoken  # reply came back through the normal TTS path


# -- 5. No base64 in audit / memory / logs ------------------------------------

class _CapturingDescriber:
    """Returns fixed text while capturing the exact uploaded bytes."""

    def __init__(self, text="a calm lake at dawn"):
        self.text = text
        self.uploaded: bytes = b""

    async def describe(self, image_data: bytes, mime: str, question: str):
        self.uploaded = bytes(image_data)
        return VisualDescription(text=self.text, model="fake-vision")


def test_image_bytes_never_reach_audit_memory_or_logs(tmp_path, caplog):
    """Raw bytes, base64, and data-URIs stay out of every persisted sink."""
    src = _make_png(tmp_path / "lake.png", size=(64, 48))
    engine = create_memory_engine(str(tmp_path / "mem.db"))
    memory = MemoryManager(engine)
    audit_path = tmp_path / "audit.jsonl"
    settings = Settings(
        ai_provider="echo",
        ai_model="test",
        nexus_allowed_roots=[str(tmp_path)],
    )
    registry = build_default_registry(settings)
    executor = ToolExecutor(registry=registry, audit=AuditLogger(audit_path))
    spy = _CapturingDescriber()
    assistant = NexusAssistant(
        settings=settings,
        provider=EchoProvider(),
        registry=registry,
        executor=executor,
        memory=memory,
        vision_describer=spy,
    )
    with caplog.at_level(logging.WARNING, logger="nexus"):
        reply = assistant.chat_with_image(str(src), "What is shown?")
    assert reply  # honest answer produced; safe text may persist
    assert spy.uploaded[:2] == b"\xff\xd8"  # upload really happened locally
    payload_b64 = base64.b64encode(spy.uploaded).decode("ascii")

    audit_text = audit_path.read_text(encoding="utf-8") if audit_path.exists() else ""
    memory_text = "\n".join(
        rec.content for rec in memory.search("", limit=50)
    )
    history_text = "\n".join(msg.content for msg in assistant.get_history())

    for sink_name, sink in (
        ("audit", audit_text),
        ("memory", memory_text),
        ("history", history_text),
        ("logs", caplog.text),
    ):
        assert payload_b64 not in sink, f"base64 leaked into {sink_name}"
        assert "data:image" not in sink, f"data-URI leaked into {sink_name}"
    # Raw JPEG bytes are not valid UTF-8 text; check a decodable fragment
    # cannot round-trip either: the first 12 raw bytes' latin-1 rendering
    # must not appear verbatim in any text sink.
    fragment = spy.uploaded[:12].decode("latin-1")
    assert fragment not in audit_text + memory_text + caplog.text
    # Safe textual information is still stored where intended.
    assert "lake.png" in memory_text
    assert reply.strip() in memory_text
