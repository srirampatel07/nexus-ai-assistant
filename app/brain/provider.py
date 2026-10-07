"""Provider-independent AI abstraction (Phase 1).

NEXUS talks to AI only through `AIProvider`. Switching models/providers
must never require changes outside this module + configuration.

Interface:
- send_message()
- stream_message()
- generate_tool_call()  (Phase 1 stub — tools arrive in Phase 2)
- health_check()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.core.exceptions import ConfigError, ProviderAuthError, ProviderUnavailable
from app.security.audit import sanitize
from app.tools.schemas import (
    LlmTool,
    LlmToolCall,
    parse_openai_tool_calls,
    to_openai_tools,
)

Role = Literal["system", "user", "assistant", "tool"]

MAX_ERROR_BODY_CHARS = 500


def describe_http_error(status: int, body: str, model: str) -> str:
    """Build a safe diagnostic message: status + sanitized body snippet.

    Secrets are redacted before inclusion; the body is truncated. Never
    contains the API key, headers, or .env contents.
    """
    snippet = sanitize(body or "")[:MAX_ERROR_BODY_CHARS]
    detail = f" (model={model}): {snippet}" if snippet else f" (model={model})"
    return f"AI provider error (HTTP {status}){detail}"


class ChatMessage(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[dict] = Field(default_factory=list)
    tool_call_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Serialize for the OpenAI-compatible wire format.

        Empty `tool_calls` and null `tool_call_id` are omitted: strict
        providers (e.g. Groq) reject `tool_calls` on non-assistant messages,
        so         only populated fields are sent. Provider-neutral hygiene.
        """
        data = self.model_dump(exclude_none=True)
        if not data.get("tool_calls"):
            data.pop("tool_calls", None)
        return data


class ChatResponse(BaseModel):
    content: str
    model: str = ""
    provider: str = ""
    usage: dict = Field(default_factory=dict)
    tool_calls: list[LlmToolCall] = Field(default_factory=list)


class AIProvider(ABC):
    """Abstract AI provider. All providers must implement this interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    async def send_message(self, messages: list[ChatMessage]) -> ChatResponse:
        """Send messages, return the assistant's reply."""
        ...

    @abstractmethod
    async def stream_message(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[str]:
        """Stream the assistant's reply in chunks."""
        ...

    async def generate_tool_call(self, messages: list[ChatMessage]) -> None:
        """Return a tool call, or None. Phase 2: use send_message_with_tools."""
        return None

    async def send_message_with_tools(
        self, messages: list[ChatMessage], tools: list[LlmTool]
    ) -> ChatResponse:
        """Send messages with tool definitions. Default: tools unsupported,
        safe fallback to a plain response with no tool calls (never faked)."""
        response = await self.send_message(messages)
        response.tool_calls = []
        return response

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable/configured."""
        ...


class EchoProvider(AIProvider):
    """Offline provider for development and tests. No network, no credentials."""

    @property
    def name(self) -> str:
        return "echo"

    async def send_message(self, messages: list[ChatMessage]) -> ChatResponse:
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"),
            "...",
        )
        return ChatResponse(
            content=f"You said: {last_user}",
            model="echo-1",
            provider="echo",
        )

    async def stream_message(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[str]:
        response = await self.send_message(messages)
        yield response.content

    async def health_check(self) -> bool:
        return True


class OpenAICompatibleProvider(AIProvider):
    """Minimal OpenAI-compatible `/chat/completions` client via httpx.

    Configure with AI_BASE_URL (e.g. https://api.openai.com/v1),
    AI_MODEL, and AI_API_KEY. No hard-coded credentials.
    """

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 30.0):
        if not api_key:
            raise ProviderAuthError("AI_API_KEY is not set.")
        if not base_url:
            raise ProviderAuthError("AI_BASE_URL is not set.")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "openai-compatible"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def send_message(self, messages: list[ChatMessage]) -> ChatResponse:
        payload = {
            "model": self._model,
            "messages": [m.to_wire() for m in messages],
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ProviderUnavailable(f"AI provider unreachable: {exc}") from exc
        if resp.status_code == 401:
            raise ProviderAuthError("AI provider rejected credentials (401).")
        if resp.status_code != 200:
            raise ProviderUnavailable(
                describe_http_error(resp.status_code, resp.text, self._model)
            )
        try:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderUnavailable("Malformed AI provider response.") from exc
        return ChatResponse(
            content=content, model=self._model, provider=self.name
        )

    async def stream_message(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[str]:
        response = await self.send_message(messages)
        yield response.content

    async def send_message_with_tools(
        self, messages: list[ChatMessage], tools: list[LlmTool]
    ) -> ChatResponse:
        """Full tool loop turn: sends `tools` + `tool_choice: auto`, parses
        any `tool_calls` from the response. If the model/endpoint rejects
        tools (HTTP 400 mentioning tools), falls back to a plain response
        with no calls — never faked."""
        payload: dict = {
            "model": self._model,
            "messages": [m.to_wire() for m in messages],
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
            payload["tool_choice"] = "auto"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise ProviderUnavailable(f"AI provider unreachable: {exc}") from exc
        if resp.status_code == 401:
            raise ProviderAuthError("AI provider rejected credentials (401).")
        if resp.status_code == 400 and tools and "tool" in resp.text.lower():
            fallback = await self.send_message(messages)
            fallback.tool_calls = []
            return fallback
        if resp.status_code != 200:
            raise ProviderUnavailable(
                describe_http_error(resp.status_code, resp.text, self._model)
            )
        try:
            data = resp.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderUnavailable("Malformed AI provider response.") from exc
        return ChatResponse(
            content=content,
            model=self._model,
            provider=self.name,
            tool_calls=parse_openai_tool_calls(message),
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models", headers=self._headers()
                )
                return resp.status_code == 200
        except Exception:
            return False


def create_provider(settings: Settings) -> AIProvider:
    """Factory: build the configured provider. Raises ConfigError if unknown."""
    key = settings.ai_provider.strip().lower()
    if key == "echo":
        return EchoProvider()
    if key in ("openai", "openai-compatible", "openai_compatible"):
        return OpenAICompatibleProvider(
            base_url=settings.ai_base_url,
            model=settings.ai_model,
            api_key=settings.ai_api_key.get_secret_value(),
            timeout=settings.ai_timeout_seconds,
        )
    raise ConfigError(
        f"Unknown AI_PROVIDER '{settings.ai_provider}'. "
        "Supported in Phase 1: 'echo', 'openai' / 'openai-compatible'."
    )
