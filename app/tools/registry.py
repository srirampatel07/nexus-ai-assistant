"""Central plugin-style tool registry (Phase 2).

Tools register themselves here; the assistant never hard-codes tool
selection — the AI receives tool definitions dynamically and the
ToolExecutor dispatches by name.

Naming: internal tool names are canonical and may contain dots
(e.g. "system.info"). Model-visible function names must satisfy the
OpenAI-compatible restriction ([A-Za-z0-9_-]+), so the registry exposes
`to_external_name` mapping (dots -> underscores) and resolves both
spellings on dispatch. Definitions sent to the provider always carry
the external form.
"""

from __future__ import annotations

from app.core.exceptions import ToolNotAvailable
from app.tools.base import Tool
from app.tools.schemas import LlmTool, is_valid_tool_name, to_external_name


class ToolRegistry:
    """Name-keyed registry. Internal tool names must be unique, and their
    external (model-visible) forms must also be unique (no collisions)."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._external_to_internal: dict[str, str] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool. Rejects duplicates, unnamed tools, and
        external-name collisions after normalization."""
        if not tool.name:
            raise ValueError("Cannot register a tool without a name.")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        external = to_external_name(tool.name)
        if not is_valid_tool_name(external):
            raise ValueError(
                f"Tool '{tool.name}' maps to invalid external name "
                f"{external!r}; only a-z, A-Z, 0-9, underscores, dashes allowed."
            )
        owner = self._external_to_internal.get(external)
        if owner is not None and owner != tool.name:
            raise ValueError(
                f"Tool '{tool.name}' collides with '{owner}' after "
                f"normalization (both map to {external!r})."
            )
        self._tools[tool.name] = tool
        self._external_to_internal[external] = tool.name

    def unregister(self, name: str) -> bool:
        """Remove a tool by internal or external name. Returns True if found."""
        internal = self._external_to_internal.get(name, name)
        tool = self._tools.pop(internal, None)
        if tool is None:
            return False
        self._external_to_internal.pop(to_external_name(tool.name), None)
        return True

    def get(self, name: str) -> Tool:
        """Return a tool by internal or external (model-visible) name.

        Exact internal matches win first (deterministic); otherwise the
        external mapping is consulted. Unknown or malformed names raise
        ToolNotAvailable — never an unrelated tool.
        """
        if name in self._tools:
            return self._tools[name]
        internal = self._external_to_internal.get(name or "")
        if internal is not None and internal in self._tools:
            return self._tools[internal]
        raise ToolNotAvailable(f"Unknown tool '{name}'.") from None

    # `resolve` is the dispatch-facing alias for `get`: the executor uses
    # it so model-returned external names reach the intended implementation.
    resolve = get

    def has(self, name: str) -> bool:
        return name in self._tools or name in self._external_to_internal

    def external_name(self, internal_name: str) -> str:
        """Model-visible name for a registered internal tool name."""
        if internal_name not in self._tools:
            raise ToolNotAvailable(f"Unknown tool '{internal_name}'.")
        return to_external_name(internal_name)

    def internal_name(self, external_name: str) -> str:
        """Canonical internal name for a model-visible name."""
        try:
            return self.get(external_name).name
        except ToolNotAvailable:
            raise ToolNotAvailable(f"Unknown tool '{external_name}'.") from None

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def tool_definitions(self) -> list[LlmTool]:
        """Metadata for the AI model (dynamic tool selection).

        Names are the external (provider-compatible) form; the executor
        maps them back to internal implementations on dispatch.
        """
        definitions: list[LlmTool] = []
        for tool in self._tools.values():
            definition = tool.definition()
            definition.name = to_external_name(definition.name)
            if not is_valid_tool_name(definition.name):
                raise ValueError(
                    f"Tool '{tool.name}' has an incompatible external name "
                    f"{definition.name!r}."
                )
            definitions.append(definition)
        names = [d.name for d in definitions]
        if len(set(names)) != len(names):
            raise ValueError(
                f"Tool registry has colliding external names: {names}"
            )
        return definitions

    def __len__(self) -> int:
        return len(self._tools)
