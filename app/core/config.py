"""NEXUS configuration — environment variables / .env.

Never hard-code credentials. All secrets come from the environment.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "NEXUS"
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    data_dir: Path = Field(default=Path("data"))
    logs_dir: Path = Field(default=Path("logs"))

    # --- AI provider abstraction ---
    # NVIDIA is the single cloud AI provider. `echo` is an offline,
    # non-AI stub for tests and development (it repeats input; it performs
    # no inference). `openai`/`openai-compatible` select a generic hosted
    # endpoint explicitly; NVIDIA paths never fall back to another
    # provider — failures raise and are reported honestly.
    ai_provider: str = Field(default="echo")
    ai_model: str = Field(default="nexus-default")
    ai_api_key: SecretStr = Field(default=SecretStr(""))
    ai_base_url: str = Field(default="")
    ai_timeout_seconds: float = Field(default=30.0)

    # --- NVIDIA Nemotron cloud provider (OpenAI-compatible) ---
    # Reference implementation uses https://integrate.api.nvidia.com/v1 with
    # model nvidia/nemotron-3.5-lightning-30b-a3b and Bearer auth.
    # NVIDIA_API_KEY is preferred; AI_API_KEY is accepted as fallback so
    # existing hosted configs keep working. AI_BASE_URL / AI_MODEL override
    # the NVIDIA defaults when explicitly set.
    nvidia_api_key: SecretStr = Field(default=SecretStr(""))
    nvidia_base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1"
    )
    nvidia_model: str = Field(
        default="nvidia/nemotron-3.5-lightning-30b-a3b"
    )
    # NVIDIA-hosted vision model (cloud-only). The Nemotron 3.5 Lightning
    # text model does not accept image input, so vision uses this separate
    # vision-capable model on the same NVIDIA base URL + API key.
    # Verified live (HTTP 200): meta/llama-3.2-11b-vision-instruct accepts
    # OpenAI-compatible image_url content (data-URI or URL) via POST
    # /v1/chat/completions. Retired IDs (e.g. nemotron-nano-12b-v2-vl,
    # retired 2026-08-26) are ignored by the resolver even if left in .env.
    nvidia_vision_model: str = Field(
        default="meta/llama-3.2-11b-vision-instruct"
    )

    # --- Conversation context ---
    max_history_messages: int = Field(default=20)

    # --- Text CLI ---
    text_mode_debug: bool = Field(default=False)

    # --- Tool system (Phase 2) ---
    nexus_allowed_roots: list[str] = Field(
        default_factory=lambda: [r"C:\chanti\nexus"]
    )
    nexus_allowed_apps: list[str] = Field(
        default_factory=lambda: [
            "notepad",
            "explorer",
            "powershell",
            "code",
        ]
    )
    terminal_default_timeout_seconds: float = Field(default=30.0)
    terminal_max_output_chars: int = Field(default=65536)
    filesystem_max_read_bytes: int = Field(default=1048576)
    filesystem_max_write_bytes: int = Field(default=1048576)
    tools_audit_file: str = Field(default="logs/tools.jsonl")
    max_tool_iterations: int = Field(default=3)
    # Phase 7 orchestration limits: overall wall-clock budget for one
    # agentic tool loop (the first model turn always runs), and the max
    # characters of one tool result fed back into model context (with a
    # truncation marker so the model knows output was cut).
    tool_loop_timeout_seconds: float = Field(default=300.0)
    tool_result_max_chars: int = Field(default=20000)

    # --- Persistent memory (Phase 3) ---
    memory_enabled: bool = Field(default=True)
    memory_database_path: str = Field(default="data/nexus.db")
    memory_context_limit: int = Field(default=10)
    memory_context_max_chars: int = Field(default=4000)
    memory_recent_limit: int = Field(default=10)
    memory_default_category: str = Field(default="conversation")
    memory_retention_days: int = Field(default=0)

    # --- Voice interface (Phase 4; NVIDIA cloud speech) ---
    # Speech-to-text uses the verified NVIDIA hosted ASR capability
    # (Parakeet CTC 1.1B English via NVCF function invocation over HTTPS;
    # see docs/VOICE.md). No local AI speech models are used. Spoken
    # output defaults to the local Windows OS synthesizer (SAPI, an OS
    # utility, not AI inference); `nvidia` TTS selects the verified
    # NVIDIA hosted Magpie voice instead.
    voice_enabled: bool = Field(default=True)

    # STT backend setting:
    #   nvidia = NVIDIA cloud Parakeet (live microphone works)
    #   (aliases: parakeet, parakeet-ctc-1_1b-asr)
    # Legacy local-AI values (whisper/vosk) and unknown names raise an
    # explicit error telling the user to use `nvidia`.
    voice_stt_backend: str = Field(default="nvidia")

    # TTS backend:
    #   sapi = local Windows OS speech synthesizer via pyttsx3 (OS utility,
    #          not AI inference, not NVIDIA speech generation)
    #   nvidia = NVIDIA cloud Magpie TTS (HTTPS, audio played locally)
    #   null = silent backend for tests
    voice_tts_backend: str = Field(default="sapi")

    # Verified NVIDIA speech function IDs (public routing identifiers,
    # not secrets). Override only when NVIDIA publishes a new ID.
    # STT: nvidia/parakeet-ctc-1_1b-asr (English). Verified live HTTP 200
    # Oct 2026: POST {base}/v1/audio/transcriptions, Bearer NVIDIA_API_KEY,
    # multipart file=WAV + language=en-US -> {"text": "..."}.
    nvidia_stt_function_id: str = Field(
        default="1598d209-5e27-4d3c-8079-4751568b1081"
    )
    # Explicit STT invocation base URL override (empty = derived as
    # https://<function-id>.invocation.api.nvcf.nvidia.com).
    nvidia_stt_base_url: str = Field(default="")
    # TTS: nvidia/magpie-tts-multilingual. Verified live HTTP 200 Oct 2026:
    # POST {base}/v1/audio/synthesize, Bearer NVIDIA_API_KEY, multipart
    # text+language+voice -> audio/wav. Voices listed via
    # GET {base}/v1/audio/list_voices.
    nvidia_tts_function_id: str = Field(
        default="877104f7-e885-42b9-8de8-f6e4c6303969"
    )
    # Explicit TTS invocation base URL override (empty = derived).
    nvidia_tts_base_url: str = Field(default="")

    # BCP-47 speech language for NVIDIA STT/TTS (en-US verified).
    voice_stt_language: str = Field(default="en-US")
    voice_tts_language: str = Field(default="en-US")
    # Default Magpie voice (verified in list_voices Oct 2026).
    voice_tts_voice: str = Field(
        default="Magpie-Multilingual.EN-US.Aria"
    )
    voice_tts_sample_rate_hz: int = Field(default=44100)
    # HTTPS timeouts for NVIDIA speech calls (seconds).
    voice_stt_timeout_seconds: float = Field(default=30.0)
    voice_tts_timeout_seconds: float = Field(default=30.0)

    # Legacy local-STT tuning keys (retained for compatibility; inert —
    # no local model is loaded).
    voice_stt_model_path: str = Field(default="small.en")

    voice_stt_device: str = Field(default="cpu")
    voice_stt_compute_type: str = Field(default="int8")

    voice_sample_rate: int = Field(default=16000)
    voice_channels: int = Field(default=1)

    # Whisper used "en", not "en-US" (legacy local-STT key, inert).
    voice_language: str = Field(default="en")

    voice_silence_timeout_seconds: float = Field(default=1.2)
    voice_max_utterance_seconds: float = Field(default=15.0)

    voice_tts_rate: int = Field(default=175)
    voice_tts_volume: float = Field(default=1.0)

    voice_stop_phrases: list[str] = Field(
        default_factory=lambda: [
            "stop listening",
            "goodbye nexus",
            "exit voice",
        ]
    )

    # --- Vision interface (NVIDIA-only) ---
    # Image analysis uses a vision-capable model only (never the text-only
    # AI_MODEL) through an OpenAI-compatible chat endpoint.
    # VISION_PROVIDER selects the mode:
    #   "nvidia" (used whenever AI_PROVIDER=nvidia) — NVIDIA-hosted vision
    #     (NVIDIA_VISION_MODEL, same NVIDIA base URL + API key). Loopback
    #     endpoints are never contacted; missing credentials raise an
    #     explicit error instead of silently falling back.
    #   "openai-compatible" — explicit generic hosted vision via
    #     VISION_BASE_URL or AI_BASE_URL; key is VISION_API_KEY or
    #     AI_API_KEY.
    #   "local" — removed. Local model inference is gone; requesting it
    #     raises an explicit error telling the user to use NVIDIA cloud.
    vision_enabled: bool = Field(default=True)
    vision_provider: str = Field(default="openai-compatible")
    vision_model: str = Field(default="openrouter/free")
    vision_base_url: str = Field(default="")
    vision_api_key: SecretStr = Field(default=SecretStr(""))
    # When true (default), /vision hands the vision report to the NVIDIA
    # text model for a synthesized final answer. When false, /vision shows
    # the vision report directly without any text-model call. When the text
    # provider is unreachable, the vision report is shown directly with an
    # honest notice — never routed to another provider.
    vision_text_synthesis: bool = Field(default=True)
    vision_max_bytes: int = Field(default=10_485_760)
    vision_max_dim: int = Field(default=2048)
    vision_jpeg_quality: int = Field(default=85)

    # --- Web + browser (Phase 6) ---
    # Search uses the keyless DuckDuckGo HTML results page by default
    # (verified live; real result links). A legacy Instant Answer JSON
    # endpoint still parses when configured; fetch/browser share SSRF
    # protection + size caps.
    web_enabled: bool = Field(default=True)
    web_search_provider: str = Field(default="duckduckgo")
    web_search_url: str = Field(default="https://html.duckduckgo.com/html/")
    web_search_timeout: float = Field(default=15.0)
    web_search_max_results: int = Field(default=8)
    web_fetch_timeout: float = Field(default=20.0)
    web_fetch_max_bytes: int = Field(default=1_048_576)
    web_fetch_max_redirects: int = Field(default=5)
    web_text_max_chars: int = Field(default=20000)

    def model_dump_safe(self) -> dict:
        """Return settings as a JSON-serializable dict with secrets redacted."""
        data = self.model_dump(mode="json")
        data["ai_api_key"] = (
            "***"
            if self.ai_api_key.get_secret_value()
            else ""
        )
        data["nvidia_api_key"] = (
            "***"
            if self.nvidia_api_key.get_secret_value()
            else ""
        )
        data["vision_api_key"] = (
            "***"
            if self.vision_api_key.get_secret_value()
            else ""
        )
        return data


@lru_cache
def get_settings() -> Settings:
    """Return cached settings (reads environment / .env once)."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read settings (used in tests)."""
    get_settings.cache_clear()
    return get_settings()