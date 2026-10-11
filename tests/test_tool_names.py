"""Tool-name compatibility regression tests (Phase 6 audit).

Root cause: every built-in internal name uses dots (e.g. "system.info"),
which OpenRouter/OpenAI-compatible endpoints reject with HTTP 400
("invalid name ... Only a-z, A-Z, 0-9, underscores, and dashes are
allowed"). The registry maps internal dotted names to model-visible
external names (dots -> underscores); the executor resolves both
spellings; the provider validates definitions locally before any
network round-trip. All HTTP mocked; offline.
"""

import httpx
import pytest

from app.brain.provider import OpenAICompatibleProvider
from app.core.config import Settings
from app.core.exceptions import ProviderUnavailable
from app.security.audit import AuditLogger
from app.security.confirmation import auto_deny
from app.tools import build_default_registry
from app.tools.base import Tool, ToolResult
from app.tools.executor import ToolExecutor
from app.tools.registry import ToolRegistry
from app.tools.schemas import (
    LlmTool,
    is_valid_tool_name,
    to_external_name,
    to_openai_tools,
    validate_tool_definitions,
)
from app.security.policy import PermissionLevel
from pydantic import BaseModel


class _Capture:
    """Fake httpx.AsyncClient capturing requests, replaying responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.requests = []

    def __call__(self, *a, **k):
        capture = self

        class FakeClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def post(self, url, headers=None, json=None):
                capture.requests.append(json)
                return capture._responses.pop(0)

        return FakeClient()


def _settings(tmp_path):
    return Settings(
        ai_provider="echo",
        nexus_allowed_roots=[str(tmp_path)],
        tools_audit_file=str(tmp_path / "audit.jsonl"),
    )


# -- A1: all exported function names are compatible ---------------------

def test_all_registry_external_names_are_compatible(tmp_path):
    registry = build_default_registry(_settings(tmp_path))
    definitions = registry.tool_definitions()
    assert len(definitions) == len(registry)
    for definition in definitions:
        assert is_valid_tool_name(definition.name), definition.name
        assert "." not in definition.name


def test_wire_definitions_have_no_dots(tmp_path):
    registry = build_default_registry(_settings(tmp_path))
    wire = to_openai_tools(registry.tool_definitions())
    for entry in wire:
        assert is_valid_tool_name(entry["function"]["name"])


# -- A2: invalid names rejected before transmission ----------------------

def test_to_openai_tools_rejects_dotted_names():
    with pytest.raises(ValueError, match="[Ii]nvalid"):
        to_openai_tools(
            [LlmTool(name="system.info", description="x",
                     parameters={"type": "object", "properties": {}})]
        )


def test_validate_rejects_empty_bad_and_duplicate_names():
    good = {"description": "x", "parameters": {"type": "object", "properties": {}}}
    with pytest.raises(ValueError):
        validate_tool_definitions([LlmTool(name="", **good)])
    with pytest.raises(ValueError):
        validate_tool_definitions([LlmTool(name="has space", **good)])
    with pytest.raises(ValueError):
        validate_tool_definitions(
            [LlmTool(name="a_ok", **good), LlmTool(name="a_ok", **good)]
        )


@pytest.mark.asyncio
async def test_provider_rejects_invalid_definitions_without_http(monkeypatch):
    capture = _Capture([])
    monkeypatch.setattr(httpx, "AsyncClient", capture)
    provider = OpenAICompatibleProvider(
        base_url="https://fake.test/v1", model="m", api_key="k"
    )
    from app.brain.provider import ChatMessage

    with pytest.raises(ProviderUnavailable, match="[Ii]nvalid tool definition"):
        await provider.send_message_with_tools(
            [ChatMessage(role="user", content="hi")],
            [LlmTool(name="system.info", description="x",
                     parameters={"type": "object", "properties": {}})],
        )
    assert capture.requests == []  # no network round-trip happened


# -- A3/A4: consistent mapping + correct dispatch -------------------------

def test_external_mapping_is_deterministic():
    assert to_external_name("system.info") == "system_info"
    assert to_external_name("filesystem.create_directory") == "filesystem_create_directory"
    assert to_external_name("browser.navigate") == "browser_navigate"
    assert to_external_name("system.info") == to_external_name("system.info")


def test_registry_resolves_both_spellings(tmp_path):
    registry = build_default_registry(_settings(tmp_path))
    assert registry.resolve("system.info").name == "system.info"
    assert registry.resolve("system_info").name == "system.info"
    assert registry.resolve("filesystem.read").name == "filesystem.read"
    assert registry.resolve("filesystem_read").name == "filesystem.read"
    assert registry.internal_name("web_search") == "web.search"
    assert registry.external_name("web.search") == "web_search"


@pytest.mark.asyncio
async def test_external_name_dispatches_to_correct_tool(tmp_path):
    settings = _settings(tmp_path)
    registry = build_default_registry(settings)
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(settings.tools_audit_file),
        confirm=auto_deny,
    )
    internal = await executor.execute("system.info", {})
    external = await executor.execute("system_info", {})
    assert internal.ok and external.ok
    assert internal.output == external.output
    assert "os" in external.output


# -- A5: unknown names never execute anything ------------------------------

@pytest.mark.asyncio
async def test_unknown_names_do_not_execute(tmp_path):
    settings = _settings(tmp_path)
    registry = build_default_registry(settings)
    before = len(registry)
    executor = ToolExecutor(
        registry=registry, audit=AuditLogger(settings.tools_audit_file),
        confirm=auto_deny,
    )
    for bad in ("nope.missing", "nope_missing", "system.info.evil",
                "", "system info", "SYSTEM.INFO.TOO.MUCH"):
        result = await executor.execute(bad, {})
        assert not result.ok
        assert "Unknown tool" in result.error
    assert len(registry) == before  # nothing registered as a side effect


# -- A6: no collisions after normalization ----------------------------------

def test_no_collisions_after_normalization(tmp_path):
    registry = build_default_registry(_settings(tmp_path))
    externals = [to_external_name(t.name) for t in registry.list_tools()]
    assert len(set(externals)) == len(externals)


def test_register_rejects_normalization_collision():
    registry = ToolRegistry()

    class _In(BaseModel):
        pass

    class ToolA(Tool):
        name = "a.b"
        description = "a"
        input_model = _In

        def execute(self, validated):
            return ToolResult(ok=True)

    class ToolB(Tool):
        name = "a_b"
        description = "b"
        input_model = _In

        def execute(self, validated):
            return ToolResult(ok=True)

    registry.register(ToolA())
    with pytest.raises(ValueError, match="[Cc]ollid"):
        registry.register(ToolB())


# -- A7/permissions intact ---------------------------------------------------

def test_permissions_and_confirmation_survive_mapping(tmp_path):
    registry = build_default_registry(_settings(tmp_path))
    assert registry.resolve("system_info").risk_level == PermissionLevel.SAFE
    assert registry.resolve("filesystem_write").requires_confirmation is True
    assert registry.resolve("terminal_execute").risk_level == PermissionLevel.MEDIUM


# -- B: schema serialization ---------------------------------------------------

def test_schema_serialization_shape_and_required():
    tool = LlmTool(
        name="filesystem_read",
        description="Read a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    wire = to_openai_tools([tool])
    fn = wire[0]["function"]
    assert wire[0]["type"] == "function"
    assert fn["name"] == "filesystem_read"
    assert fn["parameters"]["required"] == ["path"]
    assert fn["parameters"]["properties"]["path"]["type"] == "string"


def test_multiple_tool_definitions_all_valid(tmp_path):
    registry = build_default_registry(_settings(tmp_path))
    wire = to_openai_tools(registry.tool_definitions())
    assert len(wire) == len(registry)
    names = [entry["function"]["name"] for entry in wire]
    assert len(set(names)) == len(names)


def test_invalid_schema_handling():
    with pytest.raises(ValueError):
        validate_tool_definitions(
            [LlmTool(name="ok_name", description="x", parameters={"type": "string"})]
        )
    with pytest.raises(ValueError):
        validate_tool_definitions(
            [LlmTool(name="ok_name", description="x", parameters="nope")]  # type: ignore[arg-type]
        )
