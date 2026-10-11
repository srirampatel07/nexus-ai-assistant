"""Vision -> text handoff tests (mocked; no servers, no inference).

The handoff runs at most once per /vision call: stop the owned vision
server, start text, synthesize — never concurrently, never recursively.
"""

import pytest

pytest.importorskip("PIL")

from pathlib import Path

from PIL import Image

from app.brain.assistant import PROVIDER_ERROR_REPLY, NexusAssistant
from app.brain.provider import ChatResponse, EchoProvider
from app.core.config import Settings
from app.core.exceptions import ProviderUnavailable
from app.launcher import servers as S
from app.launcher.handoff import make_text_handoff
from app.vision.describer import VisualDescription


def _make_png(path: Path) -> Path:
    Image.new("RGB", (48, 32), (20, 40, 60)).save(path, format="PNG")
    return path


class FakeVision:
    def __init__(self, text="a red barn"):
        self.text = text
        self.calls = 0

    async def describe(self, image_data, mime, question):
        self.calls += 1
        return VisualDescription(text=self.text, model="fake-vl")


class SwitchableText(EchoProvider):
    """Text brain with scripted health/send behavior."""

    def __init__(self, healthy=False, synthesis="QWEN SYNTHESIS",
                 fail_send=False):
        self.healthy = healthy
        self.synthesis = synthesis
        self.fail_send = fail_send
        self.health_calls = 0
        self.send_calls = 0

    async def health_check(self):
        self.health_calls += 1
        return self.healthy

    async def send_message(self, messages):
        self.send_calls += 1
        if self.fail_send:
            raise ProviderUnavailable("text died mid-handoff")
        return ChatResponse(content=self.synthesis, model="t", provider="t")


def _assistant(tmp_path, provider, vision=None, handoff=None, **over):
    kwargs = dict(ai_provider="echo", ai_model="test",
                  nexus_allowed_roots=[str(tmp_path)], vision_enabled=True)
    kwargs.update(over)
    return NexusAssistant(settings=Settings(**kwargs), provider=provider,
                          vision_describer=vision or FakeVision(),
                          synthesis_handoff=handoff)


def _ops_recorder(fail_ensure=""):
    """Fake servers.* ops log for handoff factory tests."""
    ops = []

    def fake_stop(name, port):
        ops.append(("stop", name, port))
        return S.ServerResult(ok=True, detail=f"Stopped {name} (PID 1).")

    def fake_ensure(spec):
        ops.append(("ensure", spec.name, spec.port))
        if fail_ensure:
            return S.ServerResult(ok=False, detail=fail_ensure)
        return S.ServerResult(ok=True, detail=f"{spec.name} ready.")

    def fake_discover():
        return {"text": "C:/t.gguf", "vision": "C:/v.gguf",
                "mmproj": "C:/m.gguf"}

    return ops, fake_stop, fake_ensure, fake_discover


# 1. Successful vision -> text handoff -----------------------------------

@pytest.mark.asyncio
async def test_handoff_success_synthesizes(tmp_path):
    """The final answer is authored by the text model from the report."""
    text = SwitchableText(healthy=False)
    hook_calls = []

    def hook():
        hook_calls.append(1)
        text.healthy = True  # handoff brings text up
        return True, "Text model ready."

    a = _assistant(tmp_path, text, handoff=hook)
    reply = await a.achat_with_image(
        str(_make_png(tmp_path / "b.png")), "What do you see?")
    assert reply == "QWEN SYNTHESIS"  # text model authored the answer
    assert reply != PROVIDER_ERROR_REPLY
    assert hook_calls == [1]  # exactly once: no loops, no recursion


# 2 + 9. Sequential order, ownership-safe scope ----------------------------

def test_handoff_factory_order_and_scope(monkeypatch):
    ops, fake_stop, fake_ensure, fake_discover = _ops_recorder()
    monkeypatch.setattr(S, "stop_server", fake_stop)
    monkeypatch.setattr(S, "ensure_server", fake_ensure)
    monkeypatch.setattr(S, "discover_model_files", fake_discover)
    settings = Settings(ai_model="text-model-id")

    ok, detail = make_text_handoff(settings)()

    assert ok
    # Sequential: vision stopped BEFORE text ensured — never concurrent.
    assert ops[0] == ("stop", "vision", S.VISION_PORT)
    assert ops[1][0] == "ensure" and ops[1][1:] == ("text", S.TEXT_PORT)
    assert len(ops) == 2  # nothing else touched: no text stop, no vision start
    assert "Text model ready" in detail


def test_handoff_factory_stop_failure_aborts(monkeypatch):
    ops, _, fake_ensure, fake_discover = _ops_recorder()

    def bad_stop(name, port):
        ops.append(("stop", name, port))
        return S.ServerResult(ok=False, detail="cannot stop PID 9")

    monkeypatch.setattr(S, "stop_server", bad_stop)
    monkeypatch.setattr(S, "ensure_server", fake_ensure)
    monkeypatch.setattr(S, "discover_model_files", fake_discover)

    ok, detail = make_text_handoff(Settings(ai_model="m"))()
    assert not ok
    assert "Vision shutdown failed" in detail
    assert ops == [("stop", "vision", S.VISION_PORT)]  # text never started


def test_handoff_factory_text_failure_state(monkeypatch):
    ops, fake_stop, fake_ensure, fake_discover = _ops_recorder(
        fail_ensure="Only 1.5 GB RAM free.")
    monkeypatch.setattr(S, "stop_server", fake_stop)
    monkeypatch.setattr(S, "ensure_server", fake_ensure)
    monkeypatch.setattr(S, "discover_model_files", fake_discover)

    ok, detail = make_text_handoff(Settings(ai_model="m"))()
    assert not ok
    assert "Text server failed" in detail
    assert "1.5 GB" in detail  # recoverable state surfaced


# 3. Vision failure: handoff never runs -------------------------------------

@pytest.mark.asyncio
async def test_vision_failure_skips_handoff(tmp_path):
    class Broken:
        async def describe(self, *a, **k):
            raise ProviderUnavailable("vl down")

    called = []

    def hook():
        called.append(1)
        return True, "x"

    a = _assistant(tmp_path, SwitchableText(healthy=False),
                   vision=Broken(), handoff=hook)
    reply = await a.achat_with_image(
        str(_make_png(tmp_path / "b.png")), "What?")
    assert "couldn't analyze" in reply.lower()
    assert called == []


# 4 + 7. Handoff failure -> direct report with fallback notice ---------------

@pytest.mark.asyncio
async def test_handoff_failure_falls_back_with_notice(tmp_path):
    text = SwitchableText(healthy=False)

    def hook():
        return False, "Only 1.5 GB RAM free; need 3.5 GB."

    a = _assistant(tmp_path, text, handoff=hook)
    reply = await a.achat_with_image(
        str(_make_png(tmp_path / "b.png")), "What do you see?")
    assert "a red barn" in reply  # vision report kept
    assert "attempted but unavailable" in reply
    assert "1.5 GB" in reply  # failure detail surfaced, recoverable
    assert "QWEN SYNTHESIS" not in reply  # synthesis never claimed
    assert reply != PROVIDER_ERROR_REPLY


@pytest.mark.asyncio
async def test_synthesis_failure_after_handoff_keeps_report(tmp_path):
    text = SwitchableText(healthy=False, fail_send=True)

    def hook():
        text.healthy = True
        text.fail_send = True  # text dies between handoff and synthesis
        return True, "Text model ready."

    a = _assistant(tmp_path, text, handoff=hook)
    reply = await a.achat_with_image(
        str(_make_png(tmp_path / "b.png")), "What do you see?")
    assert reply != PROVIDER_ERROR_REPLY
    assert "a red barn" in reply
    assert "could not compose" in reply.lower()
    assert "QWEN SYNTHESIS" not in reply


# 5. Low RAM covered at factory level (see test_handoff_factory_text_...);
#    hook-level low-RAM covered by test_handoff_failure_falls_back_with_notice.


# 6. Invalid / missing images: handoff never attempted -------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize("name,content", [("missing.png", None),
                                          ("evil.txt", b"not an image")])
async def test_bad_images_skip_handoff(tmp_path, name, content):
    if content is not None:
        (tmp_path / name).write_bytes(content)
    called = []

    def hook():
        called.append(1)
        return True, "x"

    a = _assistant(tmp_path, SwitchableText(healthy=False), handoff=hook)
    reply = await a.achat_with_image(str(tmp_path / name), "What?")
    assert "couldn't use that image" in reply.lower()
    assert called == []


# 7b. Synthesis disabled: no text contact at all ---------------------------------

@pytest.mark.asyncio
async def test_synthesis_disabled_shows_report_directly(tmp_path):
    text = SwitchableText(healthy=True)

    def hook():  # must not run when synthesis is off
        raise AssertionError("handoff must not run")

    a = _assistant(tmp_path, text, handoff=hook,
                   vision_text_synthesis=False)
    reply = await a.achat_with_image(
        str(_make_png(tmp_path / "b.png")), "What?")
    assert "a red barn" in reply
    assert text.health_calls == 0 and text.send_calls == 0
    assert "offline" not in reply.lower()


# 10. Wiring + /vision compatibility ---------------------------------------------

def test_build_assistant_wires_handoff(tmp_path):
    from app.main import build_assistant

    hook = lambda: (True, "ok")  # noqa: E731
    a = build_assistant(
        Settings(ai_provider="echo",
                 nexus_allowed_roots=[str(tmp_path)]),
        synthesis_handoff=hook,
    )
    assert a.synthesis_handoff is hook
    b = build_assistant(
        Settings(ai_provider="echo",
                 nexus_allowed_roots=[str(tmp_path)]))
    assert b.synthesis_handoff is None  # backward compatible default
