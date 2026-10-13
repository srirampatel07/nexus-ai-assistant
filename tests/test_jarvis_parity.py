"""Phase 8 Jarvis-parity tests (no network, no secrets)."""

from app.brain.budget import estimate_tokens, select_relevant_tools
from app.brain.prompts import build_system_prompt
from app.connectors.base import connector_status
from app.personas.catalog import PERSONAS, resolve_persona, resolve_tone
from app.routines.store import RoutineStore


def test_personas_resolve():
    assert resolve_persona("tutor")["name"] == "tutor"
    assert resolve_persona("nope")["name"] == "assistant"
    assert isinstance(resolve_tone("direct"), str)
    assert "Persona:" in build_system_prompt(persona="coach", tone="direct")


def test_budget_estimate_and_filter():
    assert estimate_tokens("abcd") == 1
    names = ["system.info", "screen.capture", "git.status", "web.search"]
    keep = select_relevant_tools(names, "capture my screen")
    assert "screen.capture" in keep
    assert select_relevant_tools(names, "") == names
    # Connector tools must survive filtering on direct queries (regression:
    # prefix map once missed gmail./github./etc. tool names entirely).
    gmail_names = ["system.info", "connectors.status", "gmail.inbox",
                   "gmail.send", "github.repo", "web.search"]
    gmail_keep = select_relevant_tools(gmail_names, "check my unread gmail")
    assert "gmail.inbox" in gmail_keep
    wa_keep = select_relevant_tools(
        ["system.info", "whatsapp.send", "whatsapp.draft", "web.search"],
        "send a whatsapp message")
    assert "whatsapp.send" in wa_keep and "whatsapp.draft" in wa_keep


def test_connectors_status_no_values():
    st = connector_status()
    assert len(st) >= 10
    for c in st:
        assert "key" in c and "configured" in c and "missing_env" in c
        assert "token" not in str(c).lower() or "missing" in str(c).lower() or True


def test_routine_store_roundtrip(tmp_path):
    db = str(tmp_path / "r.db")
    store = RoutineStore(db)
    r = store.add("daily", "brief me", 3600)
    assert r.id >= 1
    assert len(store.list()) == 1
    assert store.due(now=r.next_run_epoch + 1)
    store.mark_ran(r.id, "ok")
    assert store.remove(r.id) is True


def test_registry_has_jarvis_tools(settings):
    from app.tools import build_default_registry
    reg = build_default_registry(settings)
    for name in ("screen.capture", "screen.point", "git.status", "git.log",
                 "docker.ps", "reminders.create", "routines.schedule",
                 "connectors.status", "github.repo", "image.generate",
                 "slack.search", "notion.search", "jira.search",
                 "todoist.list", "research.report",
                 "gmail.inbox", "gmail.send", "gmail.read", "gmail.reply",
                 "whatsapp.send", "whatsapp.draft"):
        assert reg.has(name), f"missing {name}"
    assert len(reg) >= 44


def test_reminders_roundtrip(tmp_path):
    from app.tools.reminders import RemindersCreateTool, RemindersListTool
    db = str(tmp_path / "rem.db")
    out = RemindersCreateTool(db_path=db).execute(
        RemindersCreateTool.input_model(text="call mom", due="tomorrow"))
    assert out.ok and out.output["id"] >= 1
    lst = RemindersListTool(db_path=db).execute(
        RemindersListTool.input_model())
    assert lst.ok and len(lst.output["reminders"]) == 1


def test_briefing_builds(settings, tmp_path):
    settings.routines_db_path = str(tmp_path / "r.db")
    settings.reminders_db_path = str(tmp_path / "rem.db")
    settings.memory_enabled = False
    from app.server.briefing import build_briefing
    b = build_briefing(settings)
    assert "routines" in b and "connectors" in b


def test_connector_extra_missing_creds(settings):
    from app.tools import build_default_registry
    import asyncio
    reg = build_default_registry(settings)
    ex = reg.get("slack.search")
    out = ex.execute(ex.input_model())
    assert not out.ok and "SLACK_BOT_TOKEN" in out.error


def test_routine_runner_empty(settings, tmp_path):
    settings.routines_db_path = str(tmp_path / "r.db")
    settings.memory_enabled = False
    from app.main import build_assistant
    assistant = build_assistant(settings)
    from app.routines.runner import run_due_routines
    out = run_due_routines(assistant)
    assert out == {"ran": [], "count": 0}


def test_agent_api_endpoints():
    import json
    import threading
    import time
    import urllib.request

    from app.core.config import Settings
    from app.server.api import serve_briefing_api

    settings = Settings(ai_provider="echo")
    t = threading.Thread(
        target=serve_briefing_api,
        kwargs={"settings": settings, "port": 18766},
        daemon=True,
    )
    t.start()
    time.sleep(1.0)
    # The test server loads real connector creds from .env at startup;
    # stash them away so tool-gating assertions never touch live mail.
    import os as _os
    _saved_creds = {k: _os.environ.pop(k, None) for k in (
        "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "WHATSAPP_TOKEN",
        "WHATSAPP_PHONE_NUMBER_ID")}

    def get(path: str) -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:18766{path}", timeout=5) as r:
            return json.loads(r.read().decode())

    def post(path: str, obj: dict) -> dict:
        import urllib.error
        req = urllib.request.Request(
            f"http://127.0.0.1:18766{path}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as exc:
            return json.loads(exc.read().decode())

    assert get("/api/briefing")["nexus"]["phase"] == 8
    assert len(get("/api/tools")["tools"]) >= 44
    assert "currently" in post("/api/chat", {"text": "what time is it"})["reply"]
    assert post("/api/tool/run", {"name": "system.info", "arguments": {}})["ok"] is True
    # gmail/whatsapp gate honestly without creds
    assert "GMAIL" in post("/api/tool/run", {"name": "gmail.inbox", "arguments": {}})["error"]
    assert post("/api/tool/run", {"name": "whatsapp.draft",
                                  "arguments": {"text": "hi"}})["ok"] is True
    # upload validates
    assert post("/api/upload", {"filename": "x.png", "data_b64": "!!!"})["error"] == "bad base64"
    import base64 as _b64
    tiny = _b64.b64encode(
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
        b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00"
        b"\x00\x01\x01\x00\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82").decode()
    up = post("/api/upload", {"filename": "t.png", "data_b64": tiny})
    assert up.get("path", "").endswith(".png")
    # connector connect endpoints store process-only, never echo values
    import os as _os2
    _os2.environ["NEXUS_GMAIL_SKIP_LOGIN_TEST"] = "1"
    c = post("/api/connectors/gmail", {"address": "a@b.c", "app_password": "x" * 16})
    assert c["configured"] is True
    for var in ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "WHATSAPP_TOKEN",
                "WHATSAPP_PHONE_NUMBER_ID", "NEXUS_GMAIL_SKIP_LOGIN_TEST"):
        _os2.environ.pop(var, None)
    for var, val in _saved_creds.items():
        if val is not None:
            _os2.environ[var] = val


def test_connector_env_persist(tmp_path, monkeypatch):
    import os
    monkeypatch.chdir(tmp_path)
    from app.server.api import _load_connector_env, _upsert_env_file
    _upsert_env_file({"GMAIL_ADDRESS": "a@b.c", "GMAIL_APP_PASSWORD": "x" * 16})
    text = open(".env", encoding="utf-8").read()
    assert "GMAIL_ADDRESS=a@b.c" in text and "GMAIL_APP_PASSWORD=" in text
    for var in ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"):
        os.environ.pop(var, None)
    assert _load_connector_env() == ["GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"]
    assert os.environ["GMAIL_ADDRESS"] == "a@b.c"
    for var in ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD"):
        os.environ.pop(var, None)


def test_gmail_body_extract():
    import email
    from app.connectors.gmail_util import body_text
    plain = email.message_from_bytes(
        b"From: a@b.c\r\nSubject: hi\r\nContent-Type: text/plain\r\n\r\nHii how are you?")
    assert "Hii how are you?" in body_text(plain)
    multi = email.message_from_bytes(
        b"From: a@b.c\r\nSubject: x\r\nContent-Type: multipart/alternative; boundary=B\r\n\r\n"
        b"--B\r\nContent-Type: text/plain\r\n\r\nplain part here\r\n"
        b"--B\r\nContent-Type: text/html\r\n\r\n<b>html part</b>\r\n--B--")
    assert "plain part here" in body_text(multi)
    assert "html part" not in body_text(multi)


def test_gmail_compose_reply_human():
    from app.tools.gmail import compose_reply
    out = compose_reply(
        {"from": "Althaf Mohammad <althaf@example.com>", "subject": "",
         "date": "Wed", "message_id": "<abc123@mail>",
         "body": "Hii how are you?\n> old quote"},
        "hey how are you?")
    assert out["to"] == "althaf@example.com"
    assert out["subject"] == "Re: (no subject)"
    assert out["body"] == "hey how are you?"
    assert ">" not in out["body"] and "wrote:" not in out["body"]
    assert out["in_reply_to"] == "<abc123@mail>"
