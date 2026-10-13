"""NEXUS entry point (Phase 8: text + voice + vision + web + orchestration + health).

All AI inference uses NVIDIA cloud APIs: chat/reasoning/tools via Nemotron
3.5 Lightning, vision via the hosted vision model, and voice speech-to-text
via hosted Parakeet ASR (+ optional Magpie cloud TTS). Microphone audio is
captured only after the user explicitly starts voice mode and is sent to
the verified NVIDIA STT endpoint for transcription; transcripts go to the
existing assistant with tools + memory. Spoken replies default to the local
Windows OS synthesizer (SAPI — an OS utility, not NVIDIA); `VOICE_TTS_BACKEND=nvidia`
selects NVIDIA cloud Magpie TTS instead. Text mode always works.

Usage:
    python run.py --text     # interactive text conversation (tools enabled)
    python run.py --voice    # live voice conversation (mic + NVIDIA STT)
    python run.py --voice --script "hello | status"   # headless pipeline check
    python run.py --health    # print provider/config health as JSON
    python run.py --version   # print version

Text commands include: /vision <image-path> [question] (explicit upload).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import __version__
from app.core.config import get_settings
from app.core.logger import get_logger, setup_logging
from app.main import build_assistant, get_nexus_info
from app.security.confirmation import ConfirmationRequest, format_request

log = get_logger("nexus.cli")

try:
    # Model output may contain characters the console codec lacks;
    # replace instead of crashing text mode.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


async def console_confirm(request: ConfirmationRequest) -> bool:
    """Interactive approval prompt for confirmation-required tools."""
    print(format_request(request))
    try:
        answer = await asyncio.to_thread(input, "Allow? [y/N] ")
    except (KeyboardInterrupt, EOFError):
        print("Denied.")
        return False
    return answer.strip().lower() in ("y", "yes")


def _preview(text: str, limit: int = 120) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit] + "..."


def _memory_line(rec) -> str:  # type: ignore[no-untyped-def]
    created = rec.created_at.isoformat(timespec="seconds") if rec.created_at else "?"
    return f"#{rec.id} [{rec.category}] ({rec.conversation_id}) {_preview(rec.content)} ({created})"


def cmd_memory_list(assistant) -> str:  # type: ignore[no-untyped-def]
    """List top memories across conversations (safe, deterministic)."""
    if assistant.memory is None:
        return "Memory is disabled. Set MEMORY_ENABLED=true to enable it."
    records = assistant.memory.search("", limit=10)
    if not records:
        return "No memories stored yet."
    return "Memories:\n" + "\n".join(_memory_line(rec) for rec in records)


def cmd_memory_search(assistant, text: str) -> str:  # type: ignore[no-untyped-def]
    """Search all memories."""
    if assistant.memory is None:
        return "Memory is disabled. Set MEMORY_ENABLED=true to enable it."
    if not text.strip():
        return "Usage: /memory search <text>"
    records = assistant.memory.search(text.strip(), limit=10)
    if not records:
        return f"No memories matching '{text.strip()}'."
    return f"Matches for '{text.strip()}':\n" + "\n".join(
        _memory_line(rec) for rec in records
    )


def cmd_forget(assistant, id_text: str) -> str:  # type: ignore[no-untyped-def]
    """Forget one memory by id (soft delete)."""
    if assistant.memory is None:
        return "Memory is disabled. Set MEMORY_ENABLED=true to enable it."
    try:
        entry_id = int(id_text.strip())
    except (ValueError, AttributeError):
        return "Usage: /forget <id>  (see /memory for ids)"
    if assistant.memory.forget(entry_id):
        return f"Forgot memory #{entry_id}."
    return f"No memory #{id_text.strip()} found."


def parse_vision_args(arg_text: str) -> tuple[str, str]:
    """Split '/vision <path> [question]' args, honoring quotes.

    Returns (image_path, question). Empty path means usage error.
    """
    import shlex

    text = (arg_text or "").strip()
    if not text:
        return "", ""
    try:
        parts = shlex.split(text, posix=True)
    except ValueError:
        parts = text.split()
    if not parts:
        return "", ""
    return parts[0], " ".join(parts[1:]).strip()


def _is_cmd(low: str, name: str) -> bool:
    """True when `low` is exactly `name` or starts with `name + ' '`.

    Prevents prefix over-matching (e.g. '/visionary' is not '/vision').
    """
    return low == name or low.startswith(name + " ")


def cmd_vision(assistant, arg_text: str) -> str:  # type: ignore[no-untyped-def]
    """Handle '/vision <image-path> [question]' (explicit upload only)."""
    from pathlib import Path

    image_path, question = parse_vision_args(arg_text)
    if not image_path:
        return "Usage: /vision <image-path> [question]"
    vision_enabled = bool(getattr(assistant.settings, "vision_enabled", True))
    if not vision_enabled:
        return "Vision is disabled (VISION_ENABLED=false)."
    settings = getattr(assistant, "settings", None)
    vision_model = getattr(settings, "vision_model", "?")
    if settings is not None and getattr(settings, "ai_provider", "").strip().lower() in (
        "nvidia",
        "nvidia-nemotron",
        "nvidia_nemotron",
        "nemotron",
    ):
        try:
            from app.brain.provider import resolve_nvidia_vision_settings

            _, resolved, _ = resolve_nvidia_vision_settings(settings)
            vision_model = resolved or vision_model
        except Exception:
            pass
    # M1: disclose the basename only, never the full filesystem path.
    print(
        f"[vision] Uploading '{Path(image_path).name}' for one-time analysis "
        f"with {vision_model} (NVIDIA cloud; nothing else is sent)."
    )
    return assistant.chat_with_image(
        image_path, question, confirm=console_confirm
    )


def cmd_health(debug: bool = False) -> int:
    from app.brain.provider import create_provider

    settings = get_settings()
    setup_logging(settings.log_level)
    info = get_nexus_info(settings)
    # Cloud-only vision verification: resolved model/endpoint + key
    # presence, no secrets, no image upload, no API quota consumed.
    vision_info: dict = {}
    if bool(getattr(settings, "vision_enabled", True)):
        try:
            from app.brain.provider import (
                nvidia_vision_stale_override,
                resolve_nvidia_vision_settings,
            )

            if settings.ai_provider.strip().lower() in (
                "nvidia",
                "nvidia-nemotron",
                "nvidia_nemotron",
                "nemotron",
            ):
                v_base, v_model, v_key = resolve_nvidia_vision_settings(settings)
                v_detail = (
                    "key set, no image uploaded by health check"
                    if v_key
                    else "set NVIDIA_API_KEY in .env "
                    "(VISION_API_KEY fallback also empty)"
                )
                stale = nvidia_vision_stale_override(settings)
                if stale:
                    v_detail += (
                        f" [note: stale override {stale!r} retired "
                        "2026-08-26, ignored — remove it from .env]"
                    )
                vision_info = {
                    "vision_status": "configured" if v_key else "missing_credentials",
                    "vision_model": v_model,
                    "vision_base_url": v_base,
                    "vision_detail": v_detail,
                }
        except Exception as exc:  # vision must never break text health
            vision_info = {"vision_status": "warn", "vision_detail": str(exc)}
    try:
        provider = create_provider(settings)
        # NVIDIA provider offers categorized health (missing creds, auth,
        # rate-limit, network, ok); other providers keep the bool check.
        detail_fn = getattr(provider, "health_detail", None)
        if callable(detail_fn):
            detail = asyncio.run(detail_fn())  # type: ignore[operator]
            healthy = bool(detail.get("ok"))
            out = {
                **info,
                **vision_info,
                "healthy": healthy,
                "provider_status": detail.get("status"),
                "provider_detail": detail.get("detail"),
            }
        else:
            healthy = asyncio.run(provider.health_check())
            out = {**info, **vision_info, "healthy": healthy}
    except Exception as exc:  # never crash: report honestly
        print(
            json.dumps(
                {**info, **vision_info, "healthy": False, "provider_status": "error", "error": str(exc)},
                indent=2,
            )
        )
        return 1
    if debug:
        out["config"] = settings.model_dump_safe()
    print(json.dumps(out, indent=2))
    return 0 if healthy else 1


def cmd_text(debug: bool = False, assistant=None) -> int:  # type: ignore[no-untyped-def]
    settings = get_settings()
    if debug:
        settings.text_mode_debug = True
    setup_logging("DEBUG" if debug else settings.log_level)
    if assistant is None:
        assistant = build_assistant(settings)

    print(f"{settings.app_name} v{__version__} - text mode (Phase 8 Jarvis parity, tools + vision + web enabled).")
    print("Type /help for commands, /exit to quit.")
    if debug:
        print(f"[debug] provider={settings.ai_provider} model={settings.ai_model}")

    while True:
        try:
            user = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return 0
        if not user:
            continue
        low = user.lower()
        if low in ("/exit", "/quit", "exit", "quit"):
            print("Goodbye.")
            return 0
        if low == "/help":
            print("Commands: /help /tools /memory /memory search <text> /forget <id> /clear /status /vision <image-path> [question] /persona <name> /tone <name> /screen /briefing /exit")
            print("Tools needing approval will ask: Allow? [y/N]")
            print("Vision (/vision) sends only that image to the vision model; ordinary chat sends no images.")
            print("/screen captures the desktop to data/screenshots/ for /vision. Personas: assistant,tutor,coach,interviewer,writer,debugger.")
            continue
        if low.startswith("/persona"):
            from app.personas.catalog import PERSONAS, resolve_persona
            arg = user[len("/persona"):].strip().lower()
            if not arg or arg not in PERSONAS:
                print(f"Usage: /persona <name>  ({', '.join(sorted(PERSONAS))})")
                continue
            settings.nexus_persona = arg
            p = resolve_persona(arg)
            print(f"Persona: {arg} — {p['description']}")
            continue
        if low == "/personas":
            from app.personas.catalog import PERSONAS
            for k in sorted(PERSONAS):
                print(f"- {k}: {PERSONAS[k]['description']}")
            continue
        if low.startswith("/tone"):
            from app.personas.catalog import TONES
            arg = user[len("/tone"):].strip().lower()
            if not arg or arg not in TONES:
                print(f"Usage: /tone <name>  ({', '.join(sorted(TONES))})")
                continue
            settings.nexus_tone = arg
            print(f"Tone: {arg}")
            continue
        if low == "/screen":
            try:
                from app.screen.capture import capture_screen
                shot = capture_screen(project_root=".")
                print(f"Screenshot: {shot.path} ({shot.width}x{shot.height}). Ask me with /vision \"{shot.path}\" <question>.")
            except Exception as exc:
                print(f"Screen capture failed: {exc}")
            continue
        if low == "/briefing":
            from app.server.briefing import build_briefing
            print(json.dumps(build_briefing(settings, conversation_id=assistant.conversation_id), indent=2))
            continue
        if low == "/tools":
            tools = assistant.registry.list_tools() if assistant.registry else []
            print("Available tools (internal name; model-visible name in parentheses):")
            for tool in tools:
                confirm = "needs approval" if tool.requires_confirmation else "auto"
                try:
                    from app.tools.schemas import to_external_name

                    external = to_external_name(tool.name)
                    label = f"{tool.name} (model: {external})" if external != tool.name else tool.name
                except Exception:
                    label = tool.name
                print(f"- {label} [{tool.category}/{tool.risk_level.value}/{confirm}]")
                print(f"    {tool.description}")
            continue
        if low == "/memory":
            print(cmd_memory_list(assistant))
            continue
        if low.startswith("/memory search "):
            print(cmd_memory_search(assistant, user[len("/memory search "):]))
            continue
        if _is_cmd(low, "/forget"):
            print(cmd_forget(assistant, user[len("/forget"):]))
            continue
        if _is_cmd(low, "/vision"):
            print(cmd_vision(assistant, user[len("/vision"):]))
            continue
        if low == "/clear":
            assistant.clear_history()
            print("Conversation cleared.")
            continue
        if low == "/status":
            info = get_nexus_info(settings)
            info["tools"] = len(assistant.registry) if assistant.registry else 0
            info["conversation_id"] = assistant.conversation_id
            if assistant.memory is not None:
                info["memory_count"] = assistant.memory.count(
                    conversation_id=assistant.conversation_id
                )
            print(json.dumps(info, indent=2))
            continue
        reply = assistant.chat(user, confirm=console_confirm)
        if debug:
            print(f"[debug] history={len(assistant.get_history())}")
        print(f"NEXUS> {reply}")
    return 0


def cmd_voice(script: list[str] | None = None) -> int:
    """Voice interface (NVIDIA cloud speech + local OS TTS default).

    Live mode captures microphone audio only after the user explicitly
    starts `--voice`, transcribes each utterance with verified NVIDIA
    cloud Parakeet STT, sends transcripts to the existing NVIDIA Nemotron
    assistant (tools + memory), and speaks replies with the configured
    TTS backend. Never crashes: missing credentials, unavailable devices,
    or network failures produce clear guidance (exit 2), never a
    traceback. `--script` runs headless pipeline checks with canned
    transcripts (no mic, no network). Text mode always works.
    """
    from app.voice.audio import check_microphone
    from app.voice.errors import VoiceUnavailableError
    from app.voice.factories import (
        create_audio_config,
        create_recorder,
        create_stt,
        create_tts,
    )
    from app.voice.session import VoiceSession

    settings = get_settings()
    setup_logging(settings.log_level)
    if not settings.voice_enabled and script is None:
        print("Voice mode is disabled (VOICE_ENABLED=false).")
        return 2
    try:
        stt = create_stt(settings, script=script)
        tts = create_tts(settings)
        recorder = create_recorder(settings)
        assistant = build_assistant(settings)
    except VoiceUnavailableError as exc:
        print(f"Voice mode unavailable.\n{exc}")
        return 2

    stt_ok, stt_detail = stt.is_available()
    tts_ok, tts_detail = tts.is_available()
    if script is not None:
        mic_ok, mic_detail = True, "scripted (no microphone needed)"
    else:
        mic_ok, mic_detail = check_microphone(settings.voice_sample_rate)
    stt_label = "script" if script is not None else settings.voice_stt_backend
    print("NEXUS Voice Mode")
    print(f"STT: {stt_label} ({stt_detail})")
    print(f"TTS: {settings.voice_tts_backend} ({tts_detail})")
    print(f"Microphone: {'available' if mic_ok else 'UNAVAILABLE'} ({mic_detail})")
    print(f"Speaker: {'available' if tts_ok else 'UNAVAILABLE'} ({tts_detail})")
    if script is None and not stt_ok:
        print(
            "Cannot start live voice mode: the speech service is not ready. "
            "Set NVIDIA_API_KEY in .env (see docs/VOICE.md), or use text "
            "mode (python run.py --text) or a headless check "
            "(python run.py --voice --script \"hello | status\")."
        )
        return 2
    if script is None and not tts_ok:
        print(
            "Warning: spoken output is unavailable; replies will print as "
            "text only. Fix TTS (see docs/VOICE.md) or use "
            "VOICE_TTS_BACKEND=null for silent mode."
        )
    if not mic_ok:
        print("Cannot start voice mode without a microphone. See docs/VOICE.md.")
        return 2
    session = VoiceSession(
        assistant=assistant,
        stt=stt,
        tts=tts,
        recorder=recorder,
        config=create_audio_config(settings),
        stop_phrases=settings.voice_stop_phrases,
    )
    return session.run_forever()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEXUS personal AI assistant")
    parser.add_argument("--text", action="store_true", help="interactive text mode")
    parser.add_argument("--health", action="store_true", help="health check as JSON")
    parser.add_argument("--version", action="store_true", help="print version")
    parser.add_argument("--voice", action="store_true", help="voice mode (live mic + NVIDIA cloud STT; --script for headless check)")
    parser.add_argument(
        "--script",
        default=None,
        help="headless voice check: '|' separated utterances, no mic needed",
    )
    parser.add_argument("--debug", action="store_true", help="verbose tool/config output")
    parser.add_argument("--briefing", action="store_true", help="print Command Center briefing JSON")
    parser.add_argument("--routines-due", action="store_true", help="print overdue routines JSON")
    parser.add_argument("--routines-run", action="store_true", help="execute due routines via assistant")
    parser.add_argument("--serve", action="store_true", help="serve Command Center HTTP API on 127.0.0.1:8765")
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0
    if args.health:
        return cmd_health(debug=args.debug)
    if args.briefing:
        from app.server.briefing import build_briefing
        settings = get_settings()
        print(json.dumps(build_briefing(settings), indent=2))
        return 0
    if args.routines_due:
        from app.routines.store import RoutineStore
        settings = get_settings()
        due = RoutineStore(settings.routines_db_path).due()
        print(json.dumps({"due": [{"id": r.id, "name": r.name, "prompt": r.prompt} for r in due]}, indent=2))
        return 0
    if args.routines_run:
        from app.main import build_assistant
        settings = get_settings()
        setup_logging(settings.log_level)
        assistant = build_assistant(settings)
        from app.routines.runner import run_due_routines
        print(json.dumps(run_due_routines(assistant), indent=2))
        return 0
    if args.serve:
        from app.server.api import serve_briefing_api
        settings = get_settings()
        setup_logging(settings.log_level)
        serve_briefing_api(settings)
        return 0
    if args.voice:
        script = (
            [s.strip() for s in args.script.split("|") if s.strip()]
            if args.script
            else None
        )
        return cmd_voice(script=script)
    if args.text:
        return cmd_text(debug=args.debug)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
