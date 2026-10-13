"""Cloud-only one-click launcher for NEXUS (Windows console).

Double-click Start-NEXUS.bat (or `python app/launcher/menu.py`) and pick:

    [1] Start NEXUS (NVIDIA cloud)  (no local servers, direct to hosted API)
    [2] Health Check                (structured status, no changes made)
    [3] Exit

NVIDIA-only operation: text uses nvidia/nemotron-3.5-lightning-30b-a3b and
vision uses meta/llama-3.2-11b-vision-instruct through
https://integrate.api.nvidia.com/v1. No llama-server.exe is started, ports
:8080/:8081 are never used (closed ports are healthy), and no GGUF model
or projector file is required. The menu loops until Exit; leaving NEXUS
returns here. Ctrl+C exits gracefully. All assistant behavior comes from
run.py — nothing duplicated. Nothing here kills any process.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import get_settings  # noqa: E402

MENU = """\
NEXUS — CLOUD AI ASSISTANT (NVIDIA)

[1] Start NEXUS (NVIDIA cloud)
[2] Health Check
[3] Exit

Select an option: """


def run_cloud(assistant=None) -> None:  # type: ignore[no-untyped-def]
    """Start the NEXUS text UI on NVIDIA cloud (no local servers).

    `assistant` is injectable for tests; None builds the standard cloud
    assistant with no local-model handoff attached.
    """
    from run import cmd_text

    if assistant is None:
        from app.main import build_assistant

        assistant = build_assistant(get_settings())
    try:
        cmd_text(assistant=assistant)
    except (KeyboardInterrupt, EOFError):
        print("\nLeaving NEXUS.")


def collect_health() -> list[tuple[str, str, str]]:
    """Structured health rows: (label, state, detail). Read-only.

    Cloud-only: verifies NVIDIA text/vision configuration without printing
    secrets and without consuming API quota (no network calls, no image
    uploads). Closed local ports are healthy — local servers are never
    used. Never marks NEXUS unhealthy merely because :8080/:8081 are down.
    """
    rows: list[tuple[str, str, str]] = []
    root = Path(__file__).resolve().parents[2]
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    rows.append(("Python", "OK" if os.path.isfile(sys.executable) else "FAIL",
                 sys.version.split()[0]))
    rows.append(("Virtualenv", "OK" if venv_python.is_file() else "FAIL",
                 str(venv_python) if venv_python.is_file()
                 else "missing .venv"))
    settings = get_settings()
    from app.brain.provider import (
        NVIDIA_DEFAULT_BASE_URL,
        NVIDIA_DEFAULT_MODEL,
        NVIDIA_DEFAULT_VISION_MODEL,
        nvidia_vision_stale_override,
        resolve_nvidia_settings,
        resolve_nvidia_vision_settings,
    )

    try:
        base_url, model, api_key = resolve_nvidia_settings(settings)
    except Exception as exc:  # never break health on resolve errors
        rows.append(("NVIDIA text", "WARN", str(exc)))
    else:
        if not model:
            model = NVIDIA_DEFAULT_MODEL
        if not base_url:
            base_url = NVIDIA_DEFAULT_BASE_URL
        if api_key:
            rows.append(("NVIDIA text", "CONFIGURED",
                         f"model={model} endpoint={base_url} "
                         "(key set, use run.py --health for live check)"))
        else:
            rows.append(("NVIDIA text", "MISSING",
                         "set NVIDIA_API_KEY in .env "
                         "(AI_API_KEY fallback also empty)"))
    if not bool(getattr(settings, "vision_enabled", True)):
        rows.append(("NVIDIA vision", "OFF", "VISION_ENABLED=false"))
    else:
        try:
            v_base, v_model, v_key = resolve_nvidia_vision_settings(settings)
        except Exception as exc:
            rows.append(("NVIDIA vision", "WARN", str(exc)))
        else:
            if not v_model:
                v_model = NVIDIA_DEFAULT_VISION_MODEL
            if not v_base:
                v_base = NVIDIA_DEFAULT_BASE_URL
            if v_key:
                detail = (f"model={v_model} endpoint={v_base} "
                          "(key set, no image uploaded by health check)")
                stale = nvidia_vision_stale_override(settings)
                if stale:
                    detail += (f" [note: stale override {stale!r} retired "
                               "2026-08-26, ignored — remove it from .env]")
                rows.append(("NVIDIA vision", "CONFIGURED", detail))
            else:
                rows.append(("NVIDIA vision", "MISSING",
                             "set NVIDIA_API_KEY in .env "
                             "(VISION_API_KEY fallback also empty)"))
    rows.append(("Local servers", "NOT USED",
                 "cloud-only: :8080/:8081 never started, closed ports are healthy"))
    db = Path(settings.memory_database_path)
    if not db.is_absolute():
        db = root / db
    rows.append(("Memory database",
                 "OK" if db.is_file() else "WARN",
                 f"{db} ({db.stat().st_size // 1024} KB)"
                 if db.is_file() else "not created yet"))
    rows.append(("Voice", "CONFIGURED" if settings.voice_enabled else "OFF",
                 f"stt={settings.voice_stt_backend} tts={settings.voice_tts_backend}"))
    rows.append(("Vision", "CONFIGURED" if settings.vision_enabled else "OFF",
                 f"provider={settings.vision_provider}"))
    rows.append(("Web", "CONFIGURED" if settings.web_enabled else "OFF",
                 f"provider={settings.web_search_provider}"))
    return rows


def show_health() -> None:
    print("\nNEXUS health (read-only — nothing started or changed):")
    for label, state, detail in collect_health():
        print(f"  [{state:10s}] {label}: {detail}")
    print()


def main() -> int:
    print("NEXUS launcher — NVIDIA cloud (no local models).")
    while True:
        try:
            choice = input(MENU).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            return 0
        if choice == "1":
            run_cloud()
        elif choice == "2":
            show_health()
        elif choice == "3":
            print("Goodbye.")
            return 0
        else:
            print("Please choose 1, 2, or 3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
