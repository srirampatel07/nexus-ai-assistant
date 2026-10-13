"""Speech-to-text backends (NVIDIA cloud + scripted headless).

Live microphone transcription uses the verified NVIDIA hosted ASR
capability — `nvidia/parakeet-ctc-1_1b-asr` (English) served through an
NVCF function invocation over HTTPS:

    POST https://<function-id>.invocation.api.nvcf.nvidia.com/v1/audio/transcriptions
    Authorization: Bearer <NVIDIA_API_KEY>   (nvapi-... from build.nvidia.com)
    multipart/form-data: file=<WAV audio>, language=en-US

Verified live (HTTP 200, Oct 2026): silence WAV -> {"text": ""};
Magpie-generated "Hello from NEXUS voice test." -> the same sentence.
No local AI speech models are used anywhere: requesting whisper/vosk
raises an explicit error. `ScriptedSTT` replays canned transcripts for
headless pipeline checks (`--script`) and tests — no audio, no network.

Audio is encoded to WAV in memory and sent to NVIDIA for transcription,
then dropped. Raw audio is never written to disk or logged, and the API
key never appears in errors or logs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from app.core.logger import get_logger
from app.voice.audio import pcm_to_wav_bytes
from app.voice.errors import STTError, STTUnavailableError, VoiceCancelledError

log = get_logger("nexus.voice.stt")

# Verified NVIDIA STT invocation (public routing ID, not a secret).
# Model: nvidia/parakeet-ctc-1_1b-asr (English). Function page:
# https://build.nvidia.com/nvidia/parakeet-ctc-1_1b-asr/api
DEFAULT_STT_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"
DEFAULT_STT_MODEL = "nvidia/parakeet-ctc-1_1b-asr"


def default_stt_base_url(function_id: str = DEFAULT_STT_FUNCTION_ID) -> str:
    """Derive the NVCF invocation base URL for a speech function ID."""
    return f"https://{function_id}.invocation.api.nvcf.nvidia.com"


@dataclass
class TranscriptionResult:
    text: str = ""
    confidence: float | None = None
    duration_seconds: float | None = None


@runtime_checkable
class SpeechToText(Protocol):
    """PCM (16-bit mono) in, transcription out."""

    def is_available(self) -> tuple[bool, str]:
        """(usable, detail) probe without transcribing."""
        ...

    def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptionResult:
        """Transcribe one utterance."""
        ...


class ScriptedSTT:
    """Dev/headless backend: replays canned transcripts in order."""

    def __init__(self, transcripts: list[str]) -> None:
        self._transcripts = list(transcripts)

    def is_available(self) -> tuple[bool, str]:
        return True, (
            f"scripted ({len(self._transcripts)} utterance(s) queued)"
        )

    def transcribe(
        self,
        pcm: bytes,
        sample_rate: int,
    ) -> TranscriptionResult:
        if not self._transcripts:
            raise VoiceCancelledError(
                "Script exhausted; ending voice session."
            )

        return TranscriptionResult(
            text=self._transcripts.pop(0)
        )


class NvidiaSTT:
    """NVIDIA cloud speech-to-text (Parakeet via NVCF HTTPS, no local model).

    Sends one WAV-encoded utterance per `transcribe()` call to the
    verified `/v1/audio/transcriptions` endpoint and returns the
    `{"text": ...}` transcript for the existing NEXUS assistant
    (NVIDIA Nemotron) to answer. Failures raise STTError (session
    continues); construction with missing credentials raises
    STTUnavailableError (voice mode refuses to start with guidance).
    Never falls back to local Whisper/Vosk.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        function_id: str = DEFAULT_STT_FUNCTION_ID,
        language: str = "en-US",
        timeout_seconds: float = 30.0,
        http_client_factory=None,
    ) -> None:
        self._api_key = api_key or ""
        fid = (function_id or DEFAULT_STT_FUNCTION_ID).strip()
        self._function_id = fid
        base = (base_url or "").strip().rstrip("/")
        self._base_url = base or default_stt_base_url(fid)
        self._language = (language or "en-US").strip() or "en-US"
        self._timeout = max(1.0, float(timeout_seconds or 30.0))
        self._client_factory = http_client_factory

    def is_available(self) -> tuple[bool, str]:
        if not self._api_key:
            return False, (
                "missing credentials: set NVIDIA_API_KEY in .env "
                "(nvapi-... from build.nvidia.com; AI_API_KEY fallback "
                "also empty). Use text mode (python run.py --text)."
            )
        return True, (
            f"nvidia {DEFAULT_STT_MODEL} (cloud STT; {self._base_url})"
        )

    def _client(self):  # type: ignore[no-untyped-def]
        if self._client_factory is not None:
            return self._client_factory(timeout=self._timeout)
        import httpx

        return httpx.Client(timeout=self._timeout)

    def transcribe(self, pcm: bytes, sample_rate: int) -> TranscriptionResult:
        if not self._api_key:
            raise STTError(
                "Speech service credentials are missing: set "
                "NVIDIA_API_KEY in .env (AI_API_KEY fallback also empty)."
            )
        if not pcm or len(pcm) < 2:
            raise STTError("Empty audio captured; nothing to transcribe.")
        if sample_rate <= 0:
            raise STTError(f"Invalid sample rate ({sample_rate}).")
        try:
            wav_bytes = pcm_to_wav_bytes(pcm, sample_rate, channels=1)
        except ValueError as exc:
            raise STTError(str(exc)) from exc

        url = f"{self._base_url}/v1/audio/transcriptions"
        try:
            with self._client() as client:
                response = client.post(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    files={"file": ("utterance.wav", wav_bytes, "audio/wav")},
                    data={
                        "language": self._language,
                        "response_format": "json",
                    },
                )
        except Exception as exc:
            kind = type(exc).__name__
            if "Timeout" in kind or "timeout" in kind.lower():
                raise STTError(
                    f"Speech service timed out after {self._timeout:g}s; "
                    "check your network and try again."
                ) from exc
            raise STTError(
                f"Speech service unreachable ({kind}); check your "
                "network and try again."
            ) from exc

        status = getattr(response, "status_code", 0)
        if status == 200:
            try:
                data = response.json()
            except Exception:
                # Some deployments return plain text on 200.
                try:
                    text = response.text or ""
                except Exception:
                    text = ""
                return TranscriptionResult(text=text.strip())
            if isinstance(data, dict):
                text = data.get("text", "")
                if isinstance(text, list):
                    text = " ".join(str(p) for p in text)
                return TranscriptionResult(text=str(text or "").strip())
            raise STTError("Malformed speech service response.")
        body = ""
        try:
            body = (response.text or "")[:300]
        except Exception:
            body = ""
        # Keep diagnostics safe: status + short snippet, never the key.
        if status in (401, 403):
            raise STTError(
                "Speech service authentication failed (HTTP "
                f"{status}): check NVIDIA_API_KEY in .env."
            )
        if status == 429:
            snippet = f" {body}" if body else ""
            raise STTError(
                f"Speech service rate-limited (HTTP 429).{snippet} "
                "Wait and try again."
            )
        if status == 402:
            raise STTError(
                "Speech service quota/payment error (HTTP 402): "
                "check your NVIDIA account."
            )
        if status == 400:
            snippet = f" {body}" if body else ""
            raise STTError(
                f"Speech service rejected the audio (HTTP 400).{snippet}"
            )
        if 500 <= status < 600:
            raise STTError(
                f"Speech service upstream failure (HTTP {status}). "
                "Try again shortly."
            )
        snippet = f" {body}" if body else ""
        raise STTError(f"Speech service error (HTTP {status}).{snippet}")


def require_no_local_stt() -> None:
    """Guard: local AI speech backends must not exist in this module."""
    import app.voice.stt as _mod

    for name in ("WhisperSTT", "VoskSTT", "FasterWhisperSTT"):
        if hasattr(_mod, name):
            raise AssertionError(
                f"local STT backend {name} must not exist (NVIDIA-only)"
            )
