"""Vision describers (NVIDIA-only: cloud inference, no local models).

`VisionDescriber` turns upload-ready JPEG bytes into untrusted text.
Two implementations:

- `OpenRouterVisionDescriber`: OpenAI-compatible chat endpoint with the
  separate `VISION_MODEL`, never `AI_MODEL`. NVIDIA cloud mode uses the
  shared NVIDIA base URL + API key; generic hosted mode reuses
  `VISION_API_KEY` or `AI_API_KEY`. The regular chat model (`AI_MODEL`)
  never receives images; vision requests always carry `VISION_MODEL`
  explicitly.
  Sends `content: [{type: text}, {type: image_url, url: data:...;base64}]`.
- `MetadataDescriber`: offline fallback. No network, no cost, no upload —
  dimensions/format/size only, honestly labeled as metadata.

Images are uploaded ONLY inside `describe()`, i.e. only after an explicit
`/vision <path>` / `achat_with_image()` request. Ordinary chat and voice
never call this module. Base64 is never logged or included in errors.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx

from app.brain.provider import MAX_ERROR_BODY_CHARS, describe_http_error
from app.core.exceptions import ProviderAuthError, ProviderUnavailable
from app.core.logger import get_logger
from app.security.audit import sanitize

log = get_logger("nexus.vision")

# Debug preview length: long enough to confirm a real description arrived,
# short enough to stay out of the way. DEBUG-only; never printed by default.
PREVIEW_CHARS = 120


@dataclass(frozen=True)
class VisualDescription:
    """Untrusted text describing an image. Never auto-executed."""

    text: str
    model: str
    width: int = 0
    height: int = 0


@runtime_checkable
class VisionDescriber(Protocol):
    """Image bytes in, untrusted description text out."""

    async def describe(
        self, image_data: bytes, mime: str, question: str
    ) -> VisualDescription:
        ...


class MetadataDescriber:
    """Offline fallback: no upload, metadata only (honestly labeled)."""

    def __init__(self, width: int = 0, height: int = 0) -> None:
        self._width = width
        self._height = height

    async def describe(
        self, image_data: bytes, mime: str, question: str
    ) -> VisualDescription:
        dims = (
            f"{self._width}x{self._height}"
            if self._width and self._height
            else "unknown dimensions"
        )
        return VisualDescription(
            text=(
                f"[Offline image metadata only — no vision model contacted. "
                f"Format: {mime}, size: {dims}, {len(image_data)} bytes. "
                f"Question was: {question.strip() or 'Describe this image.'}]"
            ),
            model="metadata-offline",
            width=self._width,
            height=self._height,
        )


class OpenRouterVisionDescriber:
    """Vision model via an OpenAI-compatible chat endpoint (hosted or local).

    Uses the configured vision model, never AI_MODEL. In NVIDIA-only
    operation it carries the NVIDIA cloud vision model
    (meta/llama-3.2-11b-vision-instruct) on the shared NVIDIA base URL +
    API key with the standard image_url content format; in generic hosted
    mode the key is VISION_API_KEY or AI_API_KEY. The class itself is
    endpoint-agnostic — the factory guarantees cloud mode never receives
    a loopback base URL.
    """

    def __init__(
        self, base_url: str, model: str, api_key: str, timeout: float = 30.0
    ) -> None:
        if not base_url:
            raise ProviderAuthError("Vision base URL is not set.")
        if not model:
            raise ProviderAuthError("VISION_MODEL is not set.")
        # An empty api_key omits the Authorization header (kept for the
        # dev/echo transports); NVIDIA cloud vision always supplies a key
        # (the factory raises when it is missing).
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._timeout = timeout

    @property
    def model(self) -> str:
        return self._model

    @property
    def base_url(self) -> str:
        return self._base_url

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _payload(
        self, image_data: bytes, mime: str, question: str
    ) -> dict:
        b64 = base64.b64encode(image_data).decode("ascii")
        prompt = question.strip() or "Describe this image in detail."
        return {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime};base64,{b64}"
                            },
                        },
                    ],
                }
            ],
        }

    async def describe(
        self, image_data: bytes, mime: str, question: str
    ) -> VisualDescription:
        if not image_data:
            raise ProviderUnavailable("Empty image data.")
        if mime not in ("image/jpeg", "image/png", "image/webp"):
            # The loader normalizes to JPEG, but never trust the caller:
            # refuse anything outside the OpenAI-compatible image types.
            raise ProviderUnavailable(f"Unsupported image MIME type: {mime}.")
        log.debug(
            "vision request model=%s image_bytes=%d mime=%s",
            self._model,
            len(image_data),
            mime,
        )
        payload = self._payload(image_data, mime, question)
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(
                f"Vision model timed out (model={self._model}): {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(
                f"Vision model unreachable (model={self._model}): "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if resp.status_code in (401, 403):
            raise ProviderAuthError(
                describe_http_error(resp.status_code, resp.text, self._model)
            )
        if resp.status_code == 429:
            raise ProviderUnavailable(
                describe_http_error(resp.status_code, resp.text, self._model)
            )
        if resp.status_code != 200:
            # describe_http_error sanitizes; body never contains our base64
            # because it is the *response* body, not the request. A 400 here
            # usually means the model cannot take images — surfaced honestly,
            # never converted into a fabricated description.
            raise ProviderUnavailable(
                describe_http_error(resp.status_code, resp.text, self._model)
            )
        try:
            data = resp.json()
            choice = data["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            if isinstance(content, list):
                # Some endpoints return content parts; flatten text parts.
                content = " ".join(
                    str(p.get("text", ""))
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            text = str(content).strip()
            finish = str(choice.get("finish_reason") or "")
        except (KeyError, IndexError, ValueError, AttributeError) as exc:
            raise ProviderUnavailable(
                "Malformed vision model response."
            ) from exc
        if not text:
            log.debug("vision response model=%s finish=%s empty=True", self._model, finish)
            raise ProviderUnavailable("Vision model returned an empty description.")
        # Truncate runaway descriptions to keep history/memory bounded.
        if len(text) > 4000:
            text = text[:4000] + "…"
        log.debug(
            "vision response model=%s finish=%s chars=%d preview=%s",
            self._model,
            finish,
            len(text),
            text[:PREVIEW_CHARS],
        )
        return VisualDescription(text=text, model=self._model)

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(
                    f"{self._base_url}/models", headers=self._headers()
                )
                return resp.status_code == 200
        except (httpx.HTTPError, ValueError):
            return False


# Backwards-compatible alias: the Phase 5 name for the same class.
GroqVisionDescriber = OpenRouterVisionDescriber


def sanitize_vision_log(value: str) -> str:
    """Extra guard: strip anything resembling base64/data-URIs from logs."""
    cleaned = sanitize(value)
    if isinstance(cleaned, str) and len(cleaned) > MAX_ERROR_BODY_CHARS:
        return cleaned[:MAX_ERROR_BODY_CHARS]
    return cleaned  # type: ignore[return-value]
