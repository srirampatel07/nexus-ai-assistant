"""Visible Brave launcher + search-result resolution tests.

`brave.open` accepts any valid public http(s) URL (no domain allowlist) or
a current-search reference ("result 2", "second", "first", "last") resolved
against real `web.search` output. Process launches are mocked; no browser
starts, no network touches live sites.
"""

import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import ToolDenied
from app.security.audit import AuditLogger
from app.security.confirmation import auto_approve, auto_deny
from app.security.policy import PermissionLevel
from app.tools import build_default_registry
from app.tools.brave import (
    BraveOpenTool,
    find_brave_executable,
    normalize_bare_host,
    validate_desktop_url,
)
from app.tools.executor import ToolExecutor
from app.web.client import WebContext
from app.web.errors import WebError
from app.web.search_session import SearchSession


class _Proc:
    def __init__(self, pid=4321):
        self.pid = pid


class _PopenSpy:
    """Mock subprocess.Popen capturing argv; never starts a process."""

    def __init__(self, fail_with=None):
        self.calls = []
        self.fail_with = fail_with

    def __call__(self, argv, **kwargs):
        self.calls.append({"argv": argv, "kwargs": kwargs})
        if self.fail_with is not None:
            raise self.fail_with
        return _Proc()


def _session_three():
    session = SearchSession()
    session.update(
        "esp32 docs",
        [
            {"title": "ESP32 Docs", "url": "https://docs.espressif.com/esp32",
             "snippet": "official"},
            {"title": "Random blog", "url": "https://never-seen-before.example/blog?a=1#top",
             "snippet": "blog"},
            {"title": "Shop", "url": "https://shop.example.org/deep/path?q=x&y=2",
             "snippet": "shop"},
        ],
    )
    return session


def _fake_finder(explicit=None):
    return "/fake/brave.exe"


def _tool(session=None, exe="/fake/brave.exe", popen=None, finder=None):
    spy = popen if isinstance(popen, _PopenSpy) else _PopenSpy()
    tool = BraveOpenTool(
        session=session or SearchSession(), brave_exe=exe, popen=spy,
        finder=finder or _fake_finder,
    )
    return tool, spy


# -- arbitrary destinations ------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://never-seen-before.example/",
        "https://docs.espressif.com/projects/esp32/en/latest/?q=adc#section-2",
        "http://intranet.local:8080/a/b?x=1&y=2#frag",
        "https://127.0.0.1:3000/dev/console",  # client-side: SSRF N/A
        "https://example.com/p/a/t/h?query=string&v=1#fragment",
    ],
)
async def test_arbitrary_urls_open_exactly(url):
    tool, spy = _tool(exe="/fake/brave.exe")
    result = tool.execute(tool.validate({"target": url}))
    assert result.ok
    assert result.output["url"] == url  # path/query/fragment preserved
    assert result.output["browser"] == "brave"
    assert spy.calls[0]["argv"] == ["/fake/brave.exe", "--new-window", url]
    assert spy.calls[0]["kwargs"].get("shell") is False  # never via shell


@pytest.mark.asyncio
async def test_bare_hostname_gets_https():
    tool, spy = _tool()
    result = tool.execute(tool.validate({"target": "python.org/doc"}))
    assert result.ok
    assert result.output["url"] == "https://python.org/doc"


def test_normalize_bare_host_rejects_garbage():
    assert normalize_bare_host("not a host!!") is None
    assert normalize_bare_host("result 2") is None
    assert normalize_bare_host("https://example.com/x") is None


# -- result selection -------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference,expected",
    [
        ("result 1", "https://docs.espressif.com/esp32"),
        ("result 2", "https://never-seen-before.example/blog?a=1#top"),
        ("2", "https://never-seen-before.example/blog?a=1#top"),
        ("#3", "https://shop.example.org/deep/path?q=x&y=2"),
        ("second", "https://never-seen-before.example/blog?a=1#top"),
        ("the second result", "https://never-seen-before.example/blog?a=1#top"),
        ("first", "https://docs.espressif.com/esp32"),
        ("last", "https://shop.example.org/deep/path?q=x&y=2"),
        ("3rd", "https://shop.example.org/deep/path?q=x&y=2"),
    ],
)
async def test_result_references_resolve_exactly(reference, expected):
    tool, spy = _tool(session=_session_three())
    result = tool.execute(tool.validate({"target": reference}))
    assert result.ok
    assert result.output["url"] == expected
    assert spy.calls[0]["argv"][2] == expected


@pytest.mark.asyncio
async def test_out_of_range_and_empty_session_are_honest():
    tool, _ = _tool(session=_session_three())
    bad = tool.execute(tool.validate({"target": "result 9"}))
    assert not bad.ok
    assert "Could not resolve" in bad.error

    lonely, _ = _tool(session=SearchSession())
    result = lonely.execute(lonely.validate({"target": "result 1"}))
    assert not result.ok
    assert "no search results" in result.error


@pytest.mark.asyncio
async def test_vague_reference_needs_single_result():
    single = SearchSession()
    single.update("x", [{"title": "Only", "url": "https://only.example/", "snippet": ""}])
    tool, _ = _tool(session=single)
    assert tool.execute(tool.validate({"target": "that website"}))
    assert tool.execute(tool.validate({"target": "that website"})).output["url"] == \
        "https://only.example/"

    multi, _ = _tool(session=_session_three())
    vague = multi.execute(multi.validate({"target": "that website"}))
    assert not vague.ok
    assert "Could not resolve" in vague.error


# -- URL validation ----------------------------------------------------------

@pytest.mark.parametrize(
    "bad",
    [
        "file:///C:/secret.txt",
        "javascript:alert(1)",
        "data:text/html,<h1>x</h1>",
        "ftp://example.com/file",
        "vbscript:msgbox(1)",
        "https://user:pass@example.com/",
        "https://user@example.com/",
        "nota url at all!!!",
        "https://",
        "x" * 3000,
        "https://example.com/has space",
    ],
)
def test_unsafe_targets_rejected(bad):
    with pytest.raises(ToolDenied):
        validate_desktop_url(bad)


def test_safe_targets_accepted():
    assert validate_desktop_url("https://example.com/") == "https://example.com/"
    assert validate_desktop_url("http://example.com/a?b=c#d")


# -- Brave discovery ----------------------------------------------------------

def test_find_brave_explicit_path():
    assert find_brave_executable(
        explicit="C:\\brave\\brave.exe",
        path_checker=lambda p: p == "C:\\brave\\brave.exe",
    ) == "C:\\brave\\brave.exe"


def test_find_brave_missing_returns_none(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda *a, **k: None)
    monkeypatch.delenv("PROGRAMFILES", raising=False)
    monkeypatch.delenv("PROGRAMFILES(X86)", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    assert find_brave_executable(path_checker=lambda p: False) is None


@pytest.mark.asyncio
async def test_missing_brave_is_honest(monkeypatch):
    import shutil as _shutil

    monkeypatch.setattr(_shutil, "which", lambda *a, **k: None)
    tool = BraveOpenTool(session=SearchSession(), brave_exe=None,
                         popen=_PopenSpy(),
                         finder=lambda explicit=None: None)
    result = tool.execute(tool.validate({"target": "https://example.com/"}))
    assert not result.ok
    assert "not found" in result.error
    assert "Nothing was opened" in result.error


@pytest.mark.asyncio
async def test_launch_failure_is_honest():
    spy = _PopenSpy(fail_with=OSError("noexec"))
    tool, _ = _tool(popen=spy)
    result = tool.execute(tool.validate({"target": "https://example.com/"}))
    assert not result.ok
    assert "Could not start Brave" in result.error
    assert "launched" not in (result.output or {})


def test_launch_reports_start_not_page_load():
    tool, _ = _tool()
    result = tool.execute(tool.validate({"target": "https://example.com/"}))
    assert result.ok
    assert result.output["launched"] is True
    assert "not verified" in result.output["note"]


# -- permissions / confirmations / audit ---------------------------------------

def test_low_risk_no_confirmation():
    """Ordinary website opening runs automatically (LOW, no approval)."""
    tool, _ = _tool()
    assert tool.requires_confirmation is False
    assert tool.risk_level == PermissionLevel.LOW


@pytest.mark.asyncio
async def test_valid_launch_runs_without_confirmation(tmp_path):
    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )
    registry = build_default_registry(settings)
    spy = _PopenSpy()
    registry.unregister("brave.open")
    registry.register(
        BraveOpenTool(session=SearchSession(), brave_exe="/fake/brave.exe",
                      popen=spy, finder=_fake_finder)
    )
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(settings.tools_audit_file),
        confirm=auto_deny,
    )
    # auto_deny would block any confirmation-gated tool — brave.open must
    # not even ask for ordinary website launches.
    result = await executor.execute(
        "brave.open", {"target": "https://example.com/"}, confirm=auto_deny
    )
    assert result.ok
    assert len(spy.calls) == 1
    logged = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "brave.open" in logged
    assert '"confirmation": "not_required"' in logged


@pytest.mark.asyncio
async def test_invalid_target_rejected_without_confirmation(tmp_path):
    """No-confirmation never bypasses validation: bad targets are denied
    at assess time and Brave is never started."""
    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )
    registry = build_default_registry(settings)
    spy = _PopenSpy()
    registry.unregister("brave.open")
    registry.register(
        BraveOpenTool(session=SearchSession(), brave_exe="/fake/brave.exe",
                      popen=spy, finder=_fake_finder)
    )
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(settings.tools_audit_file),
        confirm=auto_deny,
    )
    for bad in ("javascript:alert(1)",
                "https://user:pass@example.com/",
                "not a url!!!"):
        result = await executor.execute("brave.open", {"target": bad})
        assert not result.ok
    assert spy.calls == []
    logged = (tmp_path / "audit.jsonl").read_text(encoding="utf-8")
    assert "denied:policy" in logged


def test_other_tools_permissions_unchanged():
    """The relaxation is brave.open-only: side-effecting tools keep theirs."""
    registry = build_default_registry(Settings(ai_provider="echo"))
    click = registry.get("browser.click")
    assert click.risk_level == PermissionLevel.MEDIUM
    assert click.requires_confirmation is True
    navigate = registry.get("browser.navigate")
    assert navigate.risk_level == PermissionLevel.LOW
    launcher = registry.get("applications.launch")
    assert launcher.risk_level == PermissionLevel.LOW
    terminal = registry.get("terminal.execute")
    assert terminal.risk_level == PermissionLevel.MEDIUM
    assert registry.get("brave.open").risk_level == PermissionLevel.LOW


@pytest.mark.asyncio
async def test_browser_click_still_needs_approval(tmp_path):
    """Form/link-following keeps its confirmation gate (auto-deny blocks)."""
    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )
    registry = build_default_registry(settings)
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(settings.tools_audit_file),
        confirm=auto_deny,
    )
    result = await executor.execute(
        "browser.click", {"target": "1"}, confirm=auto_deny
    )
    assert not result.ok
    assert "approval" in result.error


# -- registry mapping + search integration --------------------------------------

def test_external_name_mapping(tmp_path):
    registry = build_default_registry(Settings(ai_provider="echo"))
    assert registry.has("brave.open")
    assert registry.has("brave_open")
    assert registry.internal_name("brave_open") == "brave.open"
    names = [d.name for d in registry.tool_definitions()]
    assert "brave_open" in names
    assert "brave.open" not in names


@pytest.mark.asyncio
async def test_search_success_feeds_session_then_opens_result(tmp_path):
    """Dynamic workflow: search (mocked HTTP) -> resolve -> Brave (mocked)."""
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "Heading": "",
                "AbstractText": "",
                "RelatedTopics": [
                    {"Text": "A page", "FirstURL": "https://dynamic.example/a"},
                    {"Text": "B page", "FirstURL": "https://dynamic.example/b?x=1"},
                ],
            },
        )
    )
    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )
    registry = build_default_registry(settings)
    search = registry.get("web.search")
    search.ctx = WebContext(
        search_url="https://search.test/",
        client_factory=lambda timeout: httpx.Client(
            transport=transport, timeout=timeout),
        dns_resolver=lambda host: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(settings.tools_audit_file),
        confirm=auto_approve,
    )
    found = await executor.execute("web.search", {"query": "arbitrary topic xyz"})
    assert found.ok
    assert found.output["count"] == 2

    brave = registry.get("brave.open")
    spy = _PopenSpy()
    brave._popen = spy
    brave._brave_exe = "/fake/brave.exe"
    opened = await executor.execute("brave.open", {"target": "result 2"})
    assert opened.ok
    assert opened.output["url"] == "https://dynamic.example/b?x=1"


@pytest.mark.asyncio
async def test_search_failure_is_honest_and_leaves_no_stale_session(tmp_path):
    def handler(request: httpx.Request) -> httpx.Response:
        raise WebError("search backend down")

    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )
    registry = build_default_registry(settings)
    search = registry.get("web.search")
    search.ctx = WebContext(
        search_url="https://search.test/",
        client_factory=lambda timeout: httpx.Client(
            transport=httpx.MockTransport(handler), timeout=timeout),
        dns_resolver=lambda host: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(settings.tools_audit_file),
        confirm=auto_deny,
    )
    # Seed a stale session, then fail: stale URLs must be gone afterwards.
    search.session.update("old", [{"title": "t", "url": "https://stale.example/"}])
    result = await executor.execute("web.search", {"query": "anything"})
    assert not result.ok
    assert len(search.session) == 0
    brave = registry.get("brave.open")
    brave._popen = _PopenSpy()
    brave._brave_exe = "/fake/brave.exe"
    opened = await executor.execute("brave.open", {"target": "result 1"})
    assert not opened.ok
    assert "no search results" in opened.error
    assert "stale.example" not in opened.error


@pytest.mark.asyncio
async def test_agent_loop_search_then_open_result(tmp_path):
    """Full dynamic workflow through the agent loop: the model searches,
    then opens a result reference — the executor resolves the real URL."""
    from app.brain.assistant import NexusAssistant
    from app.brain.provider import AIProvider, ChatMessage, ChatResponse
    from app.tools.schemas import LlmToolCall

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "Heading": "",
                "AbstractText": "",
                "RelatedTopics": [
                    {"Text": "ESP32 docs", "FirstURL": "https://docs.example/esp32"},
                ],
            },
        )
    )

    class Scripted(AIProvider):
        def __init__(self):
            self.turns = 0

        @property
        def name(self):
            return "scripted-brave"

        async def send_message(self, messages):
            return ChatResponse(content="Opened the ESP32 docs for you.",
                                model="s", provider="s")

        async def stream_message(self, messages):
            yield "Opened the ESP32 docs for you."

        async def send_message_with_tools(self, messages, tools):
            self.turns += 1
            if self.turns == 1:
                return ChatResponse(
                    content="", model="s", provider="s",
                    tool_calls=[LlmToolCall(id="c1", name="web_search",
                                            arguments={"query": "ESP32 documentation"})],
                )
            return ChatResponse(
                content="", model="s", provider="s",
                tool_calls=[LlmToolCall(id="c2", name="brave_open",
                                        arguments={"target": "result 1"})],
            )

        async def health_check(self):
            return True

    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )
    registry = build_default_registry(settings)
    registry.get("web.search").ctx = WebContext(
        search_url="https://search.test/",
        client_factory=lambda timeout: httpx.Client(
            transport=transport, timeout=timeout),
        dns_resolver=lambda host: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    spy = _PopenSpy()
    brave = registry.get("brave.open")
    brave._popen = spy
    brave._finder = _fake_finder
    assistant = NexusAssistant(
        settings=settings, provider=Scripted(),
        registry=registry,
        executor=ToolExecutor(
            registry=registry, audit=AuditLogger(settings.tools_audit_file),
            # auto_deny: brave.open must not need approval anymore.
            confirm=auto_deny),
    )
    reply = await assistant.achat("Search for ESP32 docs and open the result.")
    assert "Opened the ESP32 docs" in reply
    assert spy.calls and spy.calls[0]["argv"][2] == "https://docs.example/esp32"


@pytest.mark.asyncio
async def test_loop_answers_within_budget_after_brave_open(tmp_path):
    """The reported `tool loop hit max iterations (3)` case: search, open,
    then answer directly. The brave.open result tells the model not to
    re-fetch the visible page, so the loop ends on turn 3 with a plain
    answer — no budget exhaustion, no extra synthesis turn, no limit bump."""
    from app.brain.assistant import NexusAssistant
    from app.brain.provider import AIProvider, ChatMessage, ChatResponse
    from app.tools.schemas import LlmToolCall

    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "Heading": "",
                "AbstractText": "",
                "RelatedTopics": [
                    {"Text": "ESP32 docs", "FirstURL": "https://docs.example/esp32"},
                ],
            },
        )
    )

    class Scripted(AIProvider):
        def __init__(self):
            self.tool_turns = 0
            self.plain_turns = 0

        @property
        def name(self):
            return "scripted-brave-budget"

        async def send_message(self, messages):
            self.plain_turns += 1
            return ChatResponse(content="synthesis-fallback", model="s",
                                provider="s")

        async def stream_message(self, messages):
            yield "synthesis-fallback"

        async def send_message_with_tools(self, messages, tools):
            self.tool_turns += 1
            if self.tool_turns == 1:
                return ChatResponse(
                    content="", model="s", provider="s",
                    tool_calls=[LlmToolCall(id="c1", name="web_search",
                                            arguments={"query": "ESP32 docs"})],
                )
            if self.tool_turns == 2:
                return ChatResponse(
                    content="", model="s", provider="s",
                    tool_calls=[LlmToolCall(id="c2", name="brave_open",
                                            arguments={"target": "result 1"})],
                )
            # Model follows the result note: no re-fetch, plain answer.
            return ChatResponse(content="Opened https://docs.example/esp32 "
                                        "in Brave for you.",
                                model="s", provider="s")

        async def health_check(self):
            return True

    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )
    registry = build_default_registry(settings)
    registry.get("web.search").ctx = WebContext(
        search_url="https://search.test/",
        client_factory=lambda timeout: httpx.Client(
            transport=transport, timeout=timeout),
        dns_resolver=lambda host: [(2, 1, 6, "", ("93.184.216.34", 0))],
    )
    spy = _PopenSpy()
    brave = registry.get("brave.open")
    brave._popen = spy
    brave._finder = _fake_finder
    provider = Scripted()
    assistant = NexusAssistant(
        settings=settings, provider=provider,
        registry=registry,
        executor=ToolExecutor(
            registry=registry, audit=AuditLogger(settings.tools_audit_file),
            confirm=auto_deny),
    )
    reply = await assistant.achat("Search for ESP32 docs and open them in Brave.")
    assert reply == "Opened https://docs.example/esp32 in Brave for you."
    assert provider.tool_turns == 3  # within the default budget of 3
    assert provider.plain_turns == 0  # no max-iterations synthesis needed
    assert spy.calls[0]["argv"][2] == "https://docs.example/esp32"


def test_brave_result_note_discourages_refetch():
    tool, _ = _tool()
    result = tool.execute(tool.validate({"target": "https://example.com/"}))
    assert result.ok
    assert "do not" in result.output["note"].lower()
    assert "web.fetch" in result.output["note"]
    assert "not verified" in result.output["note"]


@pytest.mark.asyncio
async def test_fetch_ssrf_retained_while_brave_allows_local(tmp_path):
    """Server fetch still refuses loopback; desktop open is a separate op."""
    from app.tools.web import WebFetchTool

    settings = Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )
    registry = build_default_registry(settings)
    fetch = registry.get("web.fetch")
    denied = await ToolExecutor(
        registry=registry, audit=AuditLogger(settings.tools_audit_file),
    ).execute("web.fetch", {"url": "http://127.0.0.1/"})
    assert not denied.ok

    tool, spy = _tool()
    ok = tool.execute(tool.validate({"target": "http://127.0.0.1:3000/"}))
    assert ok.ok
    assert isinstance(fetch, WebFetchTool)  # fetch path untouched
