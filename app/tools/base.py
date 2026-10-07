"""Reusable Tool abstraction (Phase 2).

Every tool defines: name, description, category, risk level, input schema
(a Pydantic model — arbitrary unvalidated input is never accepted),
whether confirmation is required, and an execution timeout.

Tools declare a static `risk_level` but may override `assess()` to compute
a dynamic risk from the validated arguments (e.g. terminal.execute raises
the risk for sensitive commands or BLOCKED for destructive ones).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ValidationError

from app.core.exceptions import ToolValidationError
from app.security.policy import PermissionLevel
from app.tools.schemas import LlmTool


class ToolResult(BaseModel):
    """Structured outcome of one tool execution (JSON-serializable)."""

    ok: bool
    output: dict[str, Any] = {}
    error: str | None = None
    duration_ms: int = 0


class Tool(ABC):
    """Base class for all NEXUS tools. Subclass and register in the registry."""

    name: str = ""
    description: str = ""
    category: str = "general"
    risk_level: PermissionLevel = PermissionLevel.MEDIUM
    requires_confirmation: bool = False
    timeout_seconds: float = 30.0
    input_model: type[BaseModel] | None = None

    def validate(self, arguments: dict[str, Any]) -> BaseModel:
        """Validate raw arguments against the tool's Pydantic input model."""
        if self.input_model is None:
            raise ToolValidationError(f"Tool '{self.name}' has no input model.")
        if not isinstance(arguments, dict):
            raise ToolValidationError(f"Tool '{self.name}' needs an object argument.")
        try:
            return self.input_model(**arguments)
        except ValidationError as exc:
            raise ToolValidationError(
                f"Invalid arguments for '{self.name}': {exc.errors()[0]['msg']}"
            ) from exc

    def assess(self, validated: BaseModel) -> PermissionLevel:
        """Return the effective risk for validated arguments (dynamic override)."""
        return self.risk_level

    @abstractmethod
    def execute(self, validated: BaseModel) -> ToolResult:
        """Run the tool synchronously. Must not raise for expected failures."""
        ...

    def definition(self) -> LlmTool:
        """Expose this tool's metadata as a provider-neutral LLM definition."""
        if self.input_model is None:
            raise ToolValidationError(f"Tool '{self.name}' has no input model.")
        schema = self.input_model.model_json_schema()
        return LlmTool(
            name=self.name,
            description=self.description,
            parameters={
                "type": "object",
                "properties": schema.get("properties", {}),
                "required": schema.get("required", []),
            },
        )
