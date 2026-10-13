"""NEXUS agent console API (stdlib only, localhost).

GET  /api/briefing -> briefing JSON
GET  /api/tools    -> tool list
POST /api/chat     {text, approve?} -> {reply}
POST /api/tool/run {name, arguments?, approve?} -> ToolResult JSON
POST /api/screen/capture {} -> screenshot info
POST /api/vision   {path, question?} -> vision reply
POST /api/persona  {persona?, tone?} -> applied
POST /api/routines/run {} -> run_due_routines()
GET  /             -> dashboard HTML

Safety: binds 127.0.0.1 by default; MEDIUM+ tools auto-deny unless
approve:true is sent explicitly (dashboard toggle, default off).
Bodies capped at 32KB; text inputs capped at 4000 chars.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

BODY_CAP = 32 * 1024

# One server per port: Windows SO_REUSEADDR otherwise lets two `--serve`
# processes share 127.0.0.1:8765, splitting requests across them. Then
# Connect stores Gmail creds in process A while chat lands on process B
# ("connector isn't configured" forever). Refuse a second server loudly.
_STARTED_AT = ""


def _another_server_running(host: str, port: int) -> bool:
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.settimeout(1.0)
    try:
        return probe.connect_ex((host, port)) == 0
    except OSError:
        return False
    finally:
        probe.close()


# Connector creds that may live in .env (server loads them at startup so
# any process + any restart sees them; Connect can persist them there).
_CONNECTOR_ENV_KEYS = ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD",
                       "WHATSAPP_TOKEN", "WHATSAPP_PHONE_NUMBER_ID")


def _load_connector_env(env_path: str = ".env") -> list[str]:
    """Copy connector keys from the .env file into process env (no logging)."""
    import os
    loaded: list[str] = []
    try:
        text = open(env_path, encoding="utf-8").read()
    except OSError:
        return loaded
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key in _CONNECTOR_ENV_KEYS and value.strip():
            os.environ.setdefault(key, value.strip().strip("'").strip('"'))
            loaded.append(key)
    return loaded


def _upsert_env_file(pairs: dict[str, str], env_path: str = ".env") -> None:
    """Write/overwrite KEY=VALUE lines in .env (never prints values)."""
    import os
    path = env_path
    try:
        lines = open(path, encoding="utf-8").read().splitlines()
    except OSError:
        lines = []
    remaining = dict(pairs)
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.partition("=")[0].strip()
            if key in remaining:
                out.append(f"{key}={remaining.pop(key)}")
                continue
        out.append(line)
    for key, value in remaining.items():
        out.append(f"{key}={value}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    os.replace(tmp, path)


def _assistant_singleton(settings):
    if not hasattr(_assistant_singleton, "box"):
        _assistant_singleton.box = {}
    box = _assistant_singleton.box
    if "assistant" not in box:
        from app.main import build_assistant
        box["assistant"] = build_assistant(settings, conversation_id="dashboard")
    return box["assistant"]


async def _confirm_maybe(approve: bool):
    async def _cb(request):  # type: ignore[no-untyped-def]
        return bool(approve)
    return _cb


def serve_briefing_api(settings, *, host: str = "127.0.0.1", port: int = 8765) -> None:
    from app.server.briefing import build_briefing

    dashboard = Path("frontend/dashboard/index.html")

    def send_json(handler: BaseHTTPRequestHandler, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj).encode("utf-8")
        try:
            handler.send_response(code)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except (ConnectionAbortedError, BrokenPipeError, ConnectionResetError):
            pass  # browser navigated away mid-reply; harmless

    def read_json(handler: BaseHTTPRequestHandler) -> dict:
        try:
            size = int(handler.headers.get("Content-Length") or 0)
        except ValueError:
            size = 0
        if size <= 0 or size > BODY_CAP:
            return {}
        try:
            raw = handler.rfile.read(size)
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def do_GET(self):
            path = (self.path or "/").split("?")[0]
            if path.startswith("/api/briefing"):
                send_json(self, build_briefing(settings))
                return
            if path.startswith("/api/tools"):
                try:
                    assistant = _assistant_singleton(settings)
                    tools = assistant.registry.list_tools() if assistant.registry else []
                    send_json(self, {"tools": [
                        {"name": t.name, "description": t.description,
                         "category": t.category, "risk": t.risk_level.value,
                         "needs_approval": bool(t.requires_confirmation)}
                        for t in tools]})
                except Exception as exc:
                    send_json(self, {"error": str(exc)[:300]}, 500)
                return
            if dashboard.exists():
                body = dashboard.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(404)
            self.end_headers()

        def do_POST(self):
            path = (self.path or "/").split("?")[0]
            data = read_json(self)
            approve = bool(data.get("approve"))
            try:
                assistant = _assistant_singleton(settings)
            except Exception as exc:
                send_json(self, {"error": str(exc)[:300]}, 500)
                return
            if path.startswith("/api/chat"):
                text = str(data.get("text") or "")[:4000]
                if not text.strip():
                    send_json(self, {"error": "empty text"}, 400)
                    return
                try:
                    import asyncio

                    cb = asyncio.run(_confirm_maybe(approve))
                    reply = assistant.chat(text, confirm=cb)
                    send_json(self, {"reply": reply})
                except Exception as exc:
                    send_json(self, {"error": str(exc)[:300]}, 500)
                return
            if path.startswith("/api/tool/run"):
                name = str(data.get("name") or "")
                args = data.get("arguments") or {}
                if not isinstance(args, dict):
                    send_json(self, {"error": "arguments must be an object"}, 400)
                    return
                if not name:
                    send_json(self, {"error": "missing tool name"}, 400)
                    return
                try:
                    import asyncio

                    cb = asyncio.run(_confirm_maybe(approve))
                    result = asyncio.run(
                        assistant.executor.execute(name, args, confirm=cb))
                    send_json(self, {"ok": result.ok, "output": result.output,
                                     "error": result.error})
                except Exception as exc:
                    send_json(self, {"error": str(exc)[:300]}, 500)
                return
            if path.startswith("/api/screen/capture"):
                try:
                    import asyncio

                    cb = asyncio.run(_confirm_maybe(approve))
                    result = asyncio.run(assistant.executor.execute(
                        "screen.capture", {}, confirm=cb))
                    send_json(self, {"ok": result.ok, "output": result.output,
                                     "error": result.error})
                except Exception as exc:
                    send_json(self, {"error": str(exc)[:300]}, 500)
                return
            if path.startswith("/api/vision"):
                img = str(data.get("path") or "")[:1000]
                question = str(data.get("question") or "")[:1000]
                if not img:
                    send_json(self, {"error": "missing path"}, 400)
                    return
                try:
                    import asyncio

                    cb = asyncio.run(_confirm_maybe(approve))
                    reply = assistant.chat_with_image(img, question, confirm=cb)
                    send_json(self, {"reply": reply})
                except Exception as exc:
                    send_json(self, {"error": str(exc)[:300]}, 500)
                return
            if path.startswith("/api/persona"):
                persona = str(data.get("persona") or "").strip().lower()
                tone = str(data.get("tone") or "").strip().lower()
                try:
                    from app.personas.catalog import PERSONAS, TONES
                    from app.brain.prompts import build_system_prompt
                    if persona:
                        if persona not in PERSONAS:
                            send_json(self, {"error": f"unknown persona (try {sorted(PERSONAS)})"}, 400)
                            return
                        settings.nexus_persona = persona
                    if tone:
                        if tone not in TONES:
                            send_json(self, {"error": f"unknown tone (try {sorted(TONES)})"}, 400)
                            return
                        settings.nexus_tone = tone
                    tools = assistant.registry.tool_definitions() if assistant.registry else []
                    assistant.system_prompt = build_system_prompt(
                        tools=tools, persona=settings.nexus_persona, tone=settings.nexus_tone)
                    send_json(self, {"persona": settings.nexus_persona, "tone": settings.nexus_tone})
                except Exception as exc:
                    send_json(self, {"error": str(exc)[:300]}, 500)
                return
            if path.startswith("/api/routines/run"):
                try:
                    from app.routines.runner import run_due_routines
                    send_json(self, run_due_routines(assistant))
                except Exception as exc:
                    send_json(self, {"error": str(exc)[:300]}, 500)
                return
            if path.startswith("/api/upload"):
                import base64
                from datetime import datetime
                from pathlib import Path as _P
                name = str(data.get("filename") or "paste.png")[:80]
                b64 = str(data.get("data_b64") or "")
                if not b64:
                    send_json(self, {"error": "missing data_b64"}, 400)
                    return
                try:
                    raw = base64.b64decode(b64, validate=True)
                except Exception:
                    send_json(self, {"error": "bad base64"}, 400)
                    return
                if len(raw) > 10_000_000 or not raw:
                    send_json(self, {"error": "image empty or >10MB"}, 400)
                    return
                ext = _P(name).suffix.lower() or ".png"
                if ext not in (".png", ".jpg", ".jpeg", ".webp"):
                    ext = ".png"
                out_dir = _P("data") / "screenshots"
                out_dir.mkdir(parents=True, exist_ok=True)
                out = out_dir / f"paste-{datetime.now().strftime('%Y%m%d-%H%M%S')}{ext}"
                try:
                    out.write_bytes(raw)
                except OSError as exc:
                    send_json(self, {"error": f"save failed: {exc}"}, 500)
                    return
                send_json(self, {"ok": True, "path": str(out.resolve()),
                                 "bytes": len(raw)})
                return
            if path.startswith("/api/connectors/gmail"):
                import os as _os
                address = str(data.get("address") or "").strip()[:320]
                app_pw = "".join(str(data.get("app_password") or "").split())[:64]
                remember = bool(data.get("remember", False))
                if address:
                    _os.environ["GMAIL_ADDRESS"] = address
                if app_pw:
                    _os.environ["GMAIL_APP_PASSWORD"] = app_pw
                addr = _os.environ.get("GMAIL_ADDRESS", "")
                pw = "".join(_os.environ.get("GMAIL_APP_PASSWORD", "").split())
                if not addr or not pw:
                    send_json(self, {"configured": False,
                                     "hint": "Enter address + 16-letter app password."})
                    return
                if remember and (address or app_pw):
                    try:
                        pairs = {}
                        if address:
                            pairs["GMAIL_ADDRESS"] = address
                        if app_pw:
                            pairs["GMAIL_APP_PASSWORD"] = app_pw
                        _upsert_env_file(pairs)
                        persisted = True
                    except OSError:
                        persisted = False
                else:
                    persisted = False
                if _os.environ.get("NEXUS_GMAIL_SKIP_LOGIN_TEST"):
                    send_json(self, {"configured": True, "login_ok": None,
                                     "hint": "Saved (login test skipped)."})
                    return
                # Immediate login test so the UI reports wrong-password
                # instead of a later vague "not configured".
                where = "saved to .env (survives restarts)" if persisted else "session-only (restart wipes it)"
                try:
                    import imaplib
                    import socket as _sock
                    _sock.setdefaulttimeout(10)
                    box = imaplib.IMAP4_SSL("imap.gmail.com", 993)
                    try:
                        box.login(addr, pw)
                        box.logout()
                        send_json(self, {"configured": True, "login_ok": True,
                                         "hint": f"Connected ({where}). Try 'check my unread gmail'."})
                    except Exception:
                        send_json(self, {"configured": True, "login_ok": False,
                                         "hint": f"Saved ({where}), but Gmail rejected the login. Use a 16-letter App password (Google → 2-Step → App passwords), not your normal password. IMAP must be enabled."})
                except Exception as exc:
                    send_json(self, {"configured": True, "login_ok": False,
                                     "hint": f"Saved ({where}), login test failed: {type(exc).__name__}."})
                finally:
                    try:
                        import socket as _sock2
                        _sock2.setdefaulttimeout(None)
                    except Exception:
                        pass
                return
            if path.startswith("/api/connectors/whatsapp"):
                import os as _os
                token = str(data.get("token") or "").strip()
                phone_id = str(data.get("phone_number_id") or "").strip()[:64]
                remember = bool(data.get("remember", False))
                if token:
                    _os.environ["WHATSAPP_TOKEN"] = token
                if phone_id:
                    _os.environ["WHATSAPP_PHONE_NUMBER_ID"] = phone_id
                if remember:
                    try:
                        pairs = {}
                        if token:
                            pairs["WHATSAPP_TOKEN"] = token
                        if phone_id:
                            pairs["WHATSAPP_PHONE_NUMBER_ID"] = phone_id
                        if pairs:
                            _upsert_env_file(pairs)
                    except OSError:
                        pass
                ok = bool(_os.environ.get("WHATSAPP_TOKEN") and _os.environ.get("WHATSAPP_PHONE_NUMBER_ID"))
                send_json(self, {"configured": ok,
                                 "hint": "Saved to .env." if (ok and remember) else
                                 "Meta Developers → WhatsApp Cloud API token + phone number ID. Without it, use whatsapp.draft."})
                return
            send_json(self, {"error": "unknown endpoint"}, 404)

    print(f"NEXUS Command Center: http://{host}:{port}/  (/api/briefing)")
    loaded = _load_connector_env()
    if loaded:
        print(f"Loaded connector creds from .env: {', '.join(loaded)}")
    if _another_server_running(host, port):
        print(f"ERROR: another NEXUS server already answers {host}:{port}. "
              "Use that tab, or stop it (Ctrl+C) before starting a second one. "
              "Two servers on one port split requests and break connectors.")
        raise SystemExit(2)
    import datetime as _dt
    global _STARTED_AT
    _STARTED_AT = _dt.datetime.now().isoformat(timespec="seconds")

    from http.server import ThreadingHTTPServer as _Base

    class _SingleBind(_Base):
        # allow_reuse_address=True (stdlib default) lets a second Windows
        # process steal/share the port — exactly the split-brain above.
        allow_reuse_address = False

    try:
        _SingleBind((host, port), Handler).serve_forever()
    except OSError as exc:
        print(f"ERROR: cannot bind {host}:{port} ({exc}). "
              "Another server is already running there — use it, don't start a second.")
        raise SystemExit(2) from exc


def server_identity() -> dict:
    import os
    return {"pid": os.getpid(), "started_at": _STARTED_AT}
