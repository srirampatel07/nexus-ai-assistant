"""NEXUS entry point (Phase 2: text mode + tools + health).

Usage:
    python run.py --text     # interactive text conversation (tools enabled)
    python run.py --health    # print provider/config health as JSON
    python run.py --version   # print version
    python run.py --voice     # Phase 5 stub (clear setup message, no crash)
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
            print("Commands: /help /tools /clear /status /exit")
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
        if low == "/clear":
            assistant.clear_history()
            print("Conversation cleared.")
            continue
        if low == "/status":
            info = get_nexus_info(settings)
            info["tools"] = len(assistant.registry) if assistant.registry else 0
            print(json.dumps(info, indent=2))
            continue
        reply = assistant.chat(user, confirm=console_confirm)
        if debug:
            print(f"[debug] history={len(assistant.get_history())}")
        print(f"NEXUS> {reply}")
    return 0


def cmd_voice() -> int:
    print(
        "Voice mode is planned for Phase 5 and is not available yet.\n"
        "Missing audio dependencies (microphone / STT / TTS / wake-word).\n"
        "Use: python run.py --text"
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEXUS personal AI assistant")
    parser.add_argument("--text", action="store_true", help="interactive text mode")
    parser.add_argument("--health", action="store_true", help="health check as JSON")
    parser.add_argument("--version", action="store_true", help="print version")
    parser.add_argument("--voice", action="store_true", help="voice mode (Phase 5)")
    parser.add_argument("--debug", action="store_true", help="verbose tool/config output")
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0
    if args.health:
        return cmd_health(debug=args.debug)
    if args.voice:
        return cmd_voice()
    if args.text:
        return cmd_text(debug=args.debug)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
