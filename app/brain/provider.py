
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
# Reference NVIDIA Nemotron defaults (from the supplied working Python file).
# Do not guess or substitute another model: these are the official values.
NVIDIA_DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_DEFAULT_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"
# NVIDIA-hosted vision model (the Lightning text model does not accept
# image input). Verified live (HTTP 200, Oct 2026) on the hosted endpoint
# with an OpenAI-compatible image_url data-URI payload.
NVIDIA_DEFAULT_VISION_MODEL = "meta/llama-3.2-11b-vision-instruct"
# Retired hosted vision IDs: never sent, even if left in .env overrides.
# nvidia/nemotron-nano-12b-v2-vl was retired 2026-08-26 (HTTP 410 Gone).
_RETIRED_NVIDIA_VISION_MODELS = frozenset(
    {"nvidia/nemotron-nano-12b-v2-vl"}
)
# Reference chat parameters: temperature=1, top_p=0.95, max_tokens=16384,
# SDK extra_body={"chat_template_kwargs": {"enable_thinking": True},
#                 "reasoning_budget": 16384}, stream=True.
# NEXUS posts raw JSON (no SDK), so the extra_body *contents* are sent
# top-level (see NvidiaNemotronProvider._extra_payload); NEXUS uses
# non-streaming POST for tool-loop atomicity; reasoning_content is
# tolerated, not required.
NVIDIA_TEMPERATURE = 1.0
NVIDIA_TOP_P = 0.95
NVIDIA_MAX_TOKENS = 16384
NVIDIA_REASONING_BUDGET = 16384

_AI_MODEL_PLACEHOLDERS = frozenset({"", "nexus-default", "nexus_default"})


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
    An empty API key omits the Authorization header (dev/echo transports);
    the NVIDIA provider always requires a key and refuses to construct
    without one. When a key is supplied, Bearer authentication is used.
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

    def _extra_payload(self) -> dict[str, Any]:
        """Extra chat-completion fields for this endpoint (default: none).

        Subclasses (e.g. NVIDIA Nemotron) override this to send the exact
        reference parameters without duplicating request/error logic.
        """
        return {}

    async def send_message(self, messages: list[ChatMessage]) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.to_wire() for message in messages],
        }
        payload.update(self._extra_payload())

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
        payload.update(self._extra_payload())

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


def _is_loopback_url(raw: str) -> bool:
    low = (raw or "").strip().lower()
    return (
        "127.0.0.1" in low or "localhost" in low or "::1" in low
    )


def _looks_like_local_model_id(raw: str) -> bool:
    """True for legacy local-model file paths accidentally left in AI_MODEL.

    Switching a legacy local config to NVIDIA cloud must not send a GGUF
    path to the hosted endpoint: such values are ignored in favor of the
    official NVIDIA default. Hosted model ids never contain backslashes
    or '.gguf'.
    """
    low = (raw or "").strip().lower()
    return "\\" in raw or ".gguf" in low


def resolve_nvidia_settings(settings: Settings) -> tuple[str, str, str]:
    """Resolve NVIDIA endpoint/model/key from settings.

    Precedence (never hardcode secrets):
    - key: NVIDIA_API_KEY, else AI_API_KEY.
    - base URL: AI_BASE_URL when explicitly set to a hosted URL, else
      NVIDIA_BASE_URL, else the official default. Loopback AI_BASE_URL
      values left over from legacy local configs are ignored so a
      provider switch cannot accidentally target a loopback port.
    - model: AI_MODEL when explicitly set to a hosted id, else
      NVIDIA_MODEL, else the official default. Local GGUF paths left over
      from legacy local configs are ignored for the same reason.
    """
    api_key = (
        settings.nvidia_api_key.get_secret_value()
        or settings.ai_api_key.get_secret_value()
    )
    ai_base = settings.ai_base_url.strip()
    if ai_base and _is_loopback_url(ai_base):
        ai_base = ""
    base_url = (
        ai_base
        or settings.nvidia_base_url.strip()
        or NVIDIA_DEFAULT_BASE_URL
    )
    ai_model = settings.ai_model.strip()
    if (
        ai_model.lower() in _AI_MODEL_PLACEHOLDERS
        or _looks_like_local_model_id(ai_model)
    ):
        model = settings.nvidia_model.strip() or NVIDIA_DEFAULT_MODEL
    else:
        model = ai_model
    return base_url, model, api_key


_VISION_MODEL_PLACEHOLDERS = frozenset(
    {"", "openrouter/free", "openrouter_free", "qwen2.5-vl-3b-local"}
)


def resolve_nvidia_vision_settings(settings: Settings) -> tuple[str, str, str]:
    """Resolve NVIDIA-hosted vision endpoint/model/key (never local).

    Precedence (never hardcode secrets):
    - key: VISION_API_KEY, else NVIDIA_API_KEY, else AI_API_KEY.
    - base URL: VISION_BASE_URL when explicitly set to a hosted URL, else
      NVIDIA_BASE_URL, else the official default. Loopback URLs left over
      from legacy local vision configs are ignored so cloud images can
      never be routed to a loopback port.
    - model: VISION_MODEL when explicitly set to a hosted id, else
      NVIDIA_VISION_MODEL, else the official vision default. Local GGUF
      paths / local aliases left over from legacy configs are ignored.
    """
    api_key = (
        settings.vision_api_key.get_secret_value()
        or settings.nvidia_api_key.get_secret_value()
        or settings.ai_api_key.get_secret_value()
    )
    vision_base = settings.vision_base_url.strip()
    if vision_base and _is_loopback_url(vision_base):
        vision_base = ""
    base_url = (
        vision_base
        or settings.nvidia_base_url.strip()
        or NVIDIA_DEFAULT_BASE_URL
    )
    vision_model = settings.vision_model.strip()
    if (
        not vision_model
        or vision_model.lower() in _VISION_MODEL_PLACEHOLDERS
        or vision_model.lower() in _RETIRED_NVIDIA_VISION_MODELS
        or _looks_like_local_model_id(vision_model)
    ):
        candidate = settings.nvidia_vision_model.strip()
        if not candidate or candidate.lower() in _RETIRED_NVIDIA_VISION_MODELS:
            candidate = NVIDIA_DEFAULT_VISION_MODEL
        model = candidate
    else:
        model = vision_model
    return base_url, model, api_key


def nvidia_vision_stale_override(settings: Settings) -> str | None:
    """Return the stale retired vision model id from .env, if present.

    Non-None means the user has `VISION_MODEL` or `NVIDIA_VISION_MODEL`
    pinned to a retired hosted id (HTTP 410). The resolver already ignores
    it in favor of the current default; callers surface this string in
    health output so the stale line can be removed explicitly.
    """
    for raw in (settings.vision_model, settings.nvidia_vision_model):
        if (raw or "").strip().lower() in _RETIRED_NVIDIA_VISION_MODELS:
            return (raw or "").strip()
    return None


class NvidiaNemotronProvider(OpenAICompatibleProvider):
    """NVIDIA Nemotron cloud provider (OpenAI-compatible, raw httpx).

    Uses the exact official endpoint/model from the reference file:
    https://integrate.api.nvidia.com/v1 with
    nvidia/nemotron-3.5-lightning-30b-a3b, Bearer auth, temperature=1,
    top_p=0.95, max_tokens=16384, reasoning_budget=16384 and
    chat_template_kwargs.enable_thinking.

    Critical wire-format note: the reference file uses the OpenAI Python
    SDK, where `extra_body={...}` is an SDK-side helper whose *contents*
    are merged into the top-level JSON body. NEXUS posts raw JSON via
    httpx, so it must send those contents top-level directly. Sending a
    literal `"extra_body"` key is rejected by NVIDIA with
    `HTTP 400 {"message":"Validation: Unsupported parameter(s):
    extra_body"}` (the endpoint schema sets `additionalProperties: false`).
    The official curl/Node examples confirm the top-level shape:
    `{"model", "messages", "temperature", "top_p", "max_tokens",
    "reasoning_budget", "chat_template_kwargs", ...}` with no wrapper.

    NEXUS issues non-streaming POSTs (tool-loop atomicity); the reference
    file's stream=True SSE shape (delta.content + delta.reasoning_content)
    is tolerated on responses — reasoning_content is ignored for the final
    answer, never required. stream_message() keeps the architecture's
    compatibility behavior (yields the completed reply).

    Tool calling reuses the shared OpenAI-compatible path: validated
    definitions, parsed tool_calls, honest fallback only when the endpoint
    genuinely reports tools-unsupported. Never falls back to another
    provider; failures raise with their category preserved.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        timeout: float = 30.0,
    ):
        if not api_key:
            raise ProviderAuthError(
                "NVIDIA_API_KEY is not set (AI_API_KEY fallback also empty). "
                "Set NVIDIA_API_KEY in .env to use AI_PROVIDER=nvidia."
            )
        super().__init__(
            base_url=base_url or NVIDIA_DEFAULT_BASE_URL,
            model=model or NVIDIA_DEFAULT_MODEL,
            api_key=api_key,
            timeout=timeout,
        )

    @property
    def name(self) -> str:
        return "nvidia"

    def _extra_payload(self) -> dict[str, Any]:
        # Flattened form of the reference SDK call's
        # extra_body={"chat_template_kwargs": {"enable_thinking": True},
        #             "reasoning_budget": 16384}.
        # The SDK merges extra_body contents top-level; raw HTTP must do
        # the same — a literal "extra_body" key is rejected (HTTP 400).
        # reasoning_budget is an explicit top-level property in the
        # official nemotron-3.5-lightning schema (default 16384);
        # chat_template_kwargs appears top-level in the official
        # curl/Node examples for this model.
        return {
            "temperature": NVIDIA_TEMPERATURE,
            "top_p": NVIDIA_TOP_P,
            "max_tokens": NVIDIA_MAX_TOKENS,
            "chat_template_kwargs": {"enable_thinking": True},
            "reasoning_budget": NVIDIA_REASONING_BUDGET,
        }

    async def health_detail(self) -> dict[str, Any]:
        """Rich health status distinguishing failure modes (no secrets).

        Returns {"ok": bool, "status": str, "detail": str} where status is
        one of: ok | missing_credentials | auth_error | rate_limited |
        quota_error | bad_request | provider_error | timeout | network_error.

        The check is a single inexpensive GET to /v1/models with a short
        probe budget (5s) — it reports configuration + reachability, not a
        full inference round-trip. Chat/vision requests use the configured
        `AI_TIMEOUT_SECONDS` instead. A timeout here means the probe
        budget elapsed, not that the key is invalid; "ok" requires an
        actual HTTP 200, never just a configured key.
        """
        if not self._api_key:
            return {
                "ok": False,
                "status": "missing_credentials",
                "detail": "NVIDIA_API_KEY is not set.",
            }
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    f"{self._base_url}/models",
                    headers=self._headers(),
                )
        except httpx.TimeoutException as exc:
            return {
                "ok": False,
                "status": "timeout",
                "detail": (
                    f"NVIDIA health probe timed out (5s probe budget, "
                    f"model={self._model}): {exc}"
                ),
            }
        except httpx.HTTPError as exc:
            return {
                "ok": False,
                "status": "network_error",
                "detail": (
                    f"NVIDIA endpoint unreachable (model={self._model}): "
                    f"{type(exc).__name__}: {exc}"
                ),
            }
        if response.status_code == 200:
            return {
                "ok": True,
                "status": "ok",
                "detail": (
                    f"Connected to {self._base_url} (model={self._model})."
                ),
            }
        detail = describe_http_error(
            response.status_code, response.text, self._model
        )
        if response.status_code in (401, 403):
            status = "auth_error"
        elif response.status_code == 429:
            status = "rate_limited"
        elif response.status_code == 402:
            status = "quota_error"
        elif response.status_code == 400:
            status = "bad_request"
        elif 500 <= response.status_code < 600:
            status = "provider_error"
        else:
            status = "provider_error"
        return {"ok": False, "status": status, "detail": detail}


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

    if provider_key in (
        "nvidia",
        "nvidia-nemotron",
        "nvidia_nemotron",
        "nemotron",
    ):
        base_url, model, api_key = resolve_nvidia_settings(settings)
        return NvidiaNemotronProvider(
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=settings.ai_timeout_seconds,
        )

    raise ConfigError(
        f"Unknown AI_PROVIDER '{settings.ai_provider}'. "
        "Supported providers: 'echo', 'openai', 'openai-compatible', 'nvidia'."
    )
