"""Vision -> text model handoff for `/vision` (one model at a time).

When `/vision` finishes analyzing an image but the text server is down,
the assistant can call the closure from `make_text_handoff()` ONCE to
bring the text model up for synthesis. The handoff reuses `servers.py`
exclusively — no second process manager, no duplicated switching logic:

1. Stop the vision server, but ONLY with launcher ownership proof
   (pidfile + command-line check inside `stop_server`).
2. Ensure the text server (`ensure_server`: file check, RAM gate, port
   conflict refusal, bounded readiness poll, cleanup on failure).
3. Report `(ok, detail)`; the caller synthesizes on success or falls back
   to the direct vision report on failure.

Sequential by construction: step 2 never starts before step 1 finishes,
so the two large models are never resident concurrently because of this
handoff. End state is documented and predictable: success leaves the
TEXT server running (chat continues on text); failure leaves whatever
`stop/ensure` reported, surfaced in `detail` for a recoverable retry.
"""

from __future__ import annotations

from collections.abc import Callable

from app.core.config import Settings
from app.launcher import servers as S

# Handoff hook type: () -> (text_ready, detail). Synchronous (blocking)
# by design — callers run it in a worker thread, exactly once per call.
TextHandoff = Callable[[], tuple[bool, str]]


def make_text_handoff(settings: Settings) -> TextHandoff:
    """Build the one-shot vision -> text switch for these settings."""

    def ensure_text() -> tuple[bool, str]:
        stopped = S.stop_server("vision", S.VISION_PORT)
        if not stopped.ok:
            return False, f"Vision shutdown failed: {stopped.detail}"
        try:
            files = S.discover_model_files()
        except FileNotFoundError as exc:
            return False, f"Text model files unavailable: {exc}"
        expected = [m for m in (settings.ai_model,) if m]
        if files["text"] not in expected:
            expected.append(files["text"])
        started = S.ensure_server(S.text_spec(files["text"], expected))
        if not started.ok:
            return False, f"Text server failed: {started.detail}"
        return True, (
            f"Text model ready ({started.detail}); "
            f"vision shutdown: {stopped.detail}"
        )

    return ensure_text
