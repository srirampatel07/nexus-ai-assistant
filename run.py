"""NEXUS entry point (Phase 4: text + voice + health).

Usage:
    python run.py --text     # interactive text conversation (tools enabled)
    python run.py --voice    # microphone voice conversation
    python run.py --voice --script "hello | status"   # headless pipeline check
    python run.py --health    # print provider/config health as JSON
    python run.py --version   # print version
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


def cmd_health(debug: bool = False) -> int:
    from app.brain.provider import create_provider

    settings = get_settings()
    setup_logging(settings.log_level)
    info = get_nexus_info(settings)
    try:
        provider = create_provider(settings)
        healthy = asyncio.run(provider.health_check())
    except Exception as exc:  # never crash: report honestly
        print(json.dumps({**info, "healthy": False, "error": str(exc)}, indent=2))
        return 1
    out = {**info, "healthy": healthy}
    if debug:
        out["config"] = settings.model_dump_safe()
    print(json.dumps(out, indent=2))
    return 0 if healthy else 1


def cmd_text(debug: bool = False) -> int:
    settings = get_settings()
    if debug:
        settings.text_mode_debug = True
    setup_logging("DEBUG" if debug else settings.log_level)
    assistant = build_assistant(settings)

    print(f"{settings.app_name} v{__version__} - text mode (Phase 2, tools enabled).")
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
            print("Commands: /help /tools /memory /memory search <text> /forget <id> /clear /status /exit")
            print("Tools needing approval will ask: Allow? [y/N]")
            continue
        if low == "/tools":
            tools = assistant.registry.list_tools() if assistant.registry else []
            print("Available tools:")
            for tool in tools:
                confirm = "needs approval" if tool.requires_confirmation else "auto"
                print(f"- {tool.name} [{tool.category}/{tool.risk_level.value}/{confirm}]")
                print(f"    {tool.description}")
            continue
        if low == "/memory":
            print(cmd_memory_list(assistant))
            continue
        if low.startswith("/memory search "):
            print(cmd_memory_search(assistant, user[len("/memory search "):]))
            continue
        if low.startswith("/forget"):
            print(cmd_forget(assistant, user[len("/forget"):]))
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
    """Real voice interface (Phase 4). Never crashes: missing backends or
    devices produce clear setup guidance (exit 2), never a traceback."""
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
    print("NEXUS Voice Mode")
    print(f"STT: {settings.voice_stt_backend if script is None else 'script'} ({stt_detail})")
    print(f"TTS: {settings.voice_tts_backend} ({tts_detail})")
    print(f"Microphone: {'available' if mic_ok else 'UNAVAILABLE'} ({mic_detail})")
    print(f"Speaker: {'available' if tts_ok else 'UNAVAILABLE'} ({tts_detail})")
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
    parser.add_argument("--voice", action="store_true", help="microphone voice mode")
    parser.add_argument(
        "--script",
        default=None,
        help="headless voice check: '|' separated utterances, no mic needed",
    )
    parser.add_argument("--debug", action="store_true", help="verbose tool/config output")
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0
    if args.health:
        return cmd_health(debug=args.debug)
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
