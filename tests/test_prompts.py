"""Prompt builder tests (Phase 1)."""

from app.brain.prompts import NEXUS_IDENTITY, build_system_prompt
from app.tools.schemas import LlmTool


def test_identity_names_nexus():
    assert "NEXUS" in NEXUS_IDENTITY


def test_prompt_without_context():
    p = build_system_prompt()
    assert "NEXUS" in p


def test_prompt_with_context():
    p = build_system_prompt("user prefers concise answers")
    assert "concise answers" in p


def test_prompt_lists_tools_when_provided():
    tools = [LlmTool(name="system.info", description="Get system info.")]
    p = build_system_prompt(tools=tools)
    assert "system.info" in p
    assert "Available tools" in p


def test_prompt_without_tools_has_no_tool_section():
    assert "call by exact name" not in build_system_prompt()
