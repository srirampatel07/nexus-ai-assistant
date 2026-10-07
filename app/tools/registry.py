"""Central plugin-style tool registry (Phase 2).

Tools register themselves here; the assistant never hard-codes tool
selection — the AI receives tool definitions dynamically and the
ToolExecutor dispatches by name.
"""

from __future__ import annotations

from app.core.exceptions import ToolNotAvailable
from app.tools.base import Tool
from app.tools.schemas import LlmTool


class ToolRegistry:
    """Name-keyed registry. Tool names must be unique."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Rejects duplicates and unnamed tools."""
        if not tool.name:
            raise ValueError("Cannot register a tool without a name.")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool. Returns True if it existed."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool:
        """Return a tool by name, or raise ToolNotAvailable."""
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotAvailable(f"Unknown tool '{name}'.") from None

    def has(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def tool_definitions(self) -> list[LlmTool]:
        """Metadata for the AI model (dynamic tool selection)."""
        return [tool.definition() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
