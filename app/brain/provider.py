
"""Provider-independent AI abstraction (Phase 1).

NEXUS talks to AI only through `AIProvider`. Switching models/providers
must never require changes outside this module + configuration.

Interface:
- send_message()
- stream_message()
- generate_tool_call() (Phase 1 stub — tools arrive in Phase 2)
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
    validate_tool_definitions,
)

Role = Literal["system", "user", "assistant", "tool"]

MAX_ERROR_BODY_CHARS = 500

# Signals that the endpoint genuinely does not support tool calling
# (safe to fall back to a plain answer) — as opposed to signals that our
# own definitions are malformed (must raise, never hide behind fallback).
_TOOL_UNSUPPORTED_HINTS = (
    "tool_choice not supported",
    "tools not supported",
    "does not support tools",
    "does not support tool",
    "unsupported parameter",
    "unknown parameter",
    "no tools",
)
_TOOL_DEFINITION_HINTS = (
    "invalid name",
    "invalid function",
    "function at index",
    "only a-z",
    "alphanumeric",
    "underscore",
    "invalid schema",
    "invalid parameter",
    "not valid json schema",
    "additional properties",
)


def describe_http_error(status: int, body: str, model: str) -> str:
    """Build a safe diagnostic message: status + sanitized body snippet.

    Secrets are redacted before inclusion; the body is truncated. Never
    contains the API key, headers, or .env contents. The category hint
    (auth / rate-limit / upstream / bad-request) is derived from status
    so callers can distinguish failure modes without leaking secrets.
    """
    snippet = sanitize(body or "")[:MAX_ERROR_BODY_CHARS]
    if status in (401, 403):
        category = "authentication/authorization failure"
    elif status == 402:
        category = "payment/quota failure"
    elif status == 429:
        category = "rate limited / quota exhausted"
    elif status == 400:
        category = "bad request"
    elif 500 <= status < 600:
        category = "upstream provider failure"
    else:
        category = "provider error"
    detail = f" (model={model}): {snippet}" if snippet else f" (model={model})"
    return f"AI provider error (HTTP {status}, {category}){detail}"


def _is_tools_unsupported_400(body: str) -> bool:
    """True only when a 400 means 'tools not supported here'.

    An invalid-name/invalid-schema 400 is a local definition defect and
    must NOT fall back — it must surface so the definitions get fixed.
    """
    low = (body or "").lower()
    if any(hint in low for hint in _TOOL_DEFINITION_HINTS):
        return False
    return any(hint in low for hint in _TOOL_UNSUPPORTED_HINTS) or (
        "tool" in low and "support" in low
    )


class ChatMessage(BaseModel):
    role: Role
    content: str = ""
    tool_calls: list[dict] = Field(default_factory=list)
    tool_call_id: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """Serialize for the OpenAI-compatible wire format.

        Empty `tool_calls` and null `tool_call_id` are omitted because
        strict providers may reject empty tool-call fields.
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
        """Send messages and return the assistant's reply."""
        ...

    @abstractmethod
    async def stream_message(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[str]:
        """Stream the assistant's reply in chunks."""
        ...

    async def generate_tool_call(self, messages: list[ChatMessage]) -> None:
        """Phase 1 stub; Phase 2 uses send_message_with_tools."""
        return None

    async def send_message_with_tools(
        self, messages: list[ChatMessage], tools: list[LlmTool]
    ) -> ChatResponse:
        """Safe fallback for providers that do not support tools."""
        response = await self.send_message(messages)
        response.tool_calls = []
        return response

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider is reachable/configured."""
        ...


class EchoProvider(AIProvider):
    """Offline provider for development and tests. No network or credentials."""

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
    """OpenAI-compatible chat client using httpx.

    Configure with AI_BASE_URL, AI_MODEL, and optionally AI_API_KEY.
    An empty API key is supported for local servers such as llama.cpp.
    When a key is supplied, Bearer authentication is retained.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 30.0,
    ):
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
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    async def send_message(self, messages: list[ChatMessage]) -> ChatResponse:
        payload = {
            "model": self._model,
            "messages": [message.to_wire() for message in messages],
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(
                f"AI provider timed out (model={self._model}): {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"AI provider unreachable (model={self._model}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code in (401, 403):
            raise ProviderAuthError(
                describe_http_error(
                    response.status_code, response.text, self._model
                )
            )

        if response.status_code != 200:
            raise ProviderUnavailable(
                describe_http_error(
                    response.status_code, response.text, self._model
                )
            )

        try:
            data = response.json()
            content = data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, ValueError, AttributeError) as exc:
            raise ProviderUnavailable(
                "Malformed AI provider response."
            ) from exc

        return ChatResponse(
            content=content,
            model=self._model,
            provider=self.name,
            usage=data.get("usage") or {},
        )

    async def stream_message(
        self, messages: list[ChatMessage]
    ) -> AsyncIterator[str]:
        """Compatibility implementation yielding the completed response."""
        response = await self.send_message(messages)
        yield response.content

    async def send_message_with_tools(
        self, messages: list[ChatMessage], tools: list[LlmTool]
    ) -> ChatResponse:
        """Send tool definitions and parse any returned tool calls.

        Tool definitions are validated locally first: incompatible names
        or malformed schemas raise ProviderUnavailable immediately with a
        diagnostic (no network round-trip, no silent HTTP 400). A 400 that
        genuinely means "tools unsupported here" falls back to a plain
        response; every other failure (invalid definitions, auth,
        rate-limit, upstream, network, timeout) raises with its category
        preserved. Never fabricate tool calls.
        """
        if tools:
            try:
                validate_tool_definitions(tools)
            except ValueError as exc:
                raise ProviderUnavailable(
                    f"Invalid tool definition (model={self._model}): {exc}"
                ) from exc

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.to_wire() for message in messages],
        }

        if tools:
            payload["tools"] = to_openai_tools(tools)
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(
                f"AI provider timed out (model={self._model}): {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"AI provider unreachable (model={self._model}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if response.status_code in (401, 403):
            raise ProviderAuthError(
                describe_http_error(
                    response.status_code, response.text, self._model
                )
            )

        if response.status_code == 400 and tools and _is_tools_unsupported_400(
            response.text
        ):
            fallback = await self.send_message(messages)
            fallback.tool_calls = []
            return fallback

        if response.status_code != 200:
            raise ProviderUnavailable(
                describe_http_error(
                    response.status_code, response.text, self._model
                )
            )

        try:
            data = response.json()
            message = data["choices"][0]["message"]
            content = message.get("content") or ""
        except (KeyError, IndexError, ValueError, AttributeError) as exc:
            raise ProviderUnavailable(
                "Malformed AI provider response."
            ) from exc

        return ChatResponse(
            content=content,
            model=self._model,
            provider=self.name,
            usage=data.get("usage") or {},
            tool_calls=parse_openai_tool_calls(message),
        )

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(),
                )
                return response.status_code == 200
        except (httpx.HTTPError, ValueError):
            return False


def create_provider(settings: Settings) -> AIProvider:
    """Build the configured provider."""
    provider_key = settings.ai_provider.strip().lower()

    if provider_key == "echo":
        return EchoProvider()

    if provider_key in (
        "openai",
        "openai-compatible",
        "openai_compatible",
    ):
        return OpenAICompatibleProvider(
            base_url=settings.ai_base_url,
            model=settings.ai_model,
            api_key=settings.ai_api_key.get_secret_value(),
            timeout=settings.ai_timeout_seconds,
        )

    raise ConfigError(
        f"Unknown AI_PROVIDER '{settings.ai_provider}'. "
        "Supported providers: 'echo', 'openai', 'openai-compatible'."
    )
