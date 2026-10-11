"""Interactive one-click menu for NEXUS (Windows console).

Double-click Start-NEXUS.bat (or `python app/launcher/menu.py`) and pick:

    [1] Text Mode — Qwen3-8B      (ensures :8080, then existing text UI)
    [2] Vision Mode — Qwen2.5-VL-3B (ensures :8081, then text UI + /vision)
    [3] Health Check              (structured status, no changes made)
    [4] Exit                      (stops launcher-owned servers only)

The menu loops until Exit; leaving a mode returns here. Ctrl+C exits
gracefully. All assistant behavior comes from run.py — nothing duplicated.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.config import get_settings  # noqa: E402
from app.launcher import servers as S  # noqa: E402

MENU = """\
NEXUS — LOCAL AI ASSISTANT

[1] Text Mode — Qwen3-8B
[2] Vision Mode — Qwen2.5-VL-3B
[3] Health Check
[4] Exit

Select an option: """


def _owned_running() -> list[str]:
    running = []
    for name, port in (("text", S.TEXT_PORT), ("vision", S.VISION_PORT)):
        if S._owns_process(name, port, S.real_list_llama_processes) is not None:
            running.append(f"{name} (:{port})")
    return running


def ensure_mode(mode: str) -> bool:
    """Stop the other owned server, ensure this one. True when ready."""
    other = "vision" if mode == "text" else "text"
    other_port = S.VISION_PORT if other == "vision" else S.TEXT_PORT
    stopped = S.stop_server(other, other_port)
    if not stopped.ok:
        print(f"WARNING: {stopped.detail}")
    elif "Stopped" in stopped.detail:
        print(stopped.detail)
    try:
        files = S.discover_model_files()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        return False
    settings = get_settings()
    if mode == "text":
        expected = [m for m in (settings.ai_model,) if m]
        if files["text"] not in expected:
            expected.append(files["text"])
        spec = S.text_spec(files["text"], expected)
    else:
        expected = [m for m in (settings.vision_model, S.VISION_ALIAS,
                                files["vision"]) if m]
        spec = S.vision_spec(files["vision"], files["mmproj"], expected)
    result = S.ensure_server(spec)
    print(result.detail)
    if not result.ok:
        return False
    if mode == "vision":
        print("In text mode, use: /vision <image-path> [question]")
    return True


def run_text_ui(mode: str = "text") -> None:
    """Run the existing interactive text UI, with the vision -> text
    handoff attached in vision mode (plain text mode is untouched)."""
    from run import cmd_text

    assistant = None
    if mode == "vision":
        from app.launcher.handoff import make_text_handoff
        from app.main import build_assistant

        settings = get_settings()
        assistant = build_assistant(
            settings, synthesis_handoff=make_text_handoff(settings)
        )
    try:
        cmd_text(assistant=assistant)
    except (KeyboardInterrupt, EOFError):
        print("\nLeaving text mode.")


def collect_health() -> list[tuple[str, str, str]]:
    """Structured health rows: (label, state, detail). Read-only."""
    rows: list[tuple[str, str, str]] = []
    root = S.project_root()
    venv_python = root / ".venv" / "Scripts" / "python.exe"
    rows.append(("Python", "OK" if os.path.isfile(sys.executable) else "FAIL",
                 sys.version.split()[0]))
    rows.append(("Virtualenv", "OK" if venv_python.is_file() else "FAIL",
                 str(venv_python) if venv_python.is_file()
                 else "missing .venv"))
    try:
        exe = S.find_llama_server()
        rows.append(("llama-server", "OK", exe))
    except FileNotFoundError as exc:
        rows.append(("llama-server", "FAIL", str(exc)))
    try:
        files = S.discover_model_files()
        rows.append(("Text model file", "OK", files["text"]))
        rows.append(("Vision model file", "OK", files["vision"]))
        rows.append(("Vision projector", "OK", files["mmproj"]))
    except FileNotFoundError as exc:
        rows.append(("Model files", "FAIL", str(exc)))
    settings = get_settings()
    for label, port, want in (
        ("Text endpoint :8080", S.TEXT_PORT, [settings.ai_model]),
        ("Vision endpoint :8081", S.VISION_PORT,
         [settings.vision_model, S.VISION_ALIAS]),
    ):
        listing = S.real_http_models(f"http://127.0.0.1:{port}/v1", 5.0)
        if not listing.ok:
            state = "DOWN"
            if S.port_busy(port):
                detail = f"port busy but API unreachable ({listing.detail})"
            else:
                detail = f"not running ({listing.detail})"
        elif S.models_match(listing.model_id, listing.aliases, want):
            state = "HEALTHY"
            detail = f"serving {listing.model_id}"
        else:
            state = "WARN"
            detail = f"serving unexpected model {listing.model_id}"
        rows.append((label, state, detail))
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
                 f"provider={settings.vision_provider} model={settings.vision_model}"))
    rows.append(("Web", "CONFIGURED" if settings.web_enabled else "OFF",
                 f"provider={settings.web_search_provider}"))
    try:
        free_gb = S.real_ram_free_gb()
        rows.append(("RAM free",
                     "OK" if free_gb >= S.MIN_FREE_GB else "WARN",
                     f"{free_gb:.1f} GB (need {S.MIN_FREE_GB} GB per model)"))
    except Exception as exc:
        rows.append(("RAM free", "WARN", str(exc)))
    for label, port in (("Port :8080", S.TEXT_PORT), ("Port :8081", S.VISION_PORT)):
        try:
            pid = S.real_netstat_owner(port)
        except Exception:
            pid = None
        rows.append((label, "BUSY" if pid else "FREE",
                     f"PID {pid}" if pid else "available"))
    return rows


def show_health() -> None:
    print("\nNEXUS health (read-only — nothing started or changed):")
    for label, state, detail in collect_health():
        print(f"  [{state:10s}] {label}: {detail}")
    print()


def stop_owned() -> None:
    for name, port in (("text", S.TEXT_PORT), ("vision", S.VISION_PORT)):
        result = S.stop_server(name, port)
        if "Stopped" in result.detail:
            print(result.detail)


def main() -> int:
    print("NEXUS launcher — one model at a time (16 GB laptop).")
    while True:
        try:
            choice = input(MENU).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            stop_owned()
            return 0
        if choice == "1":
            if ensure_mode("text"):
                run_text_ui("text")
        elif choice == "2":
            if ensure_mode("vision"):
                run_text_ui("vision")
        elif choice == "3":
            show_health()
        elif choice == "4":
            print("Goodbye.")
            stop_owned()
            return 0
        else:
            print("Please choose 1, 2, 3, or 4.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
