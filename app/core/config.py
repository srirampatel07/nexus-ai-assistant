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
    ai_provider: str = Field(default="echo")
    ai_model: str = Field(default="nexus-default")
    ai_api_key: SecretStr = Field(default=SecretStr(""))
    ai_base_url: str = Field(default="")
    ai_timeout_seconds: float = Field(default=30.0)

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

    # --- Persistent memory (Phase 3) ---
    memory_enabled: bool = Field(default=True)
    memory_database_path: str = Field(default="data/nexus.db")
    memory_context_limit: int = Field(default=10)
    memory_context_max_chars: int = Field(default=4000)
    memory_recent_limit: int = Field(default=10)
    memory_default_category: str = Field(default="conversation")
    memory_retention_days: int = Field(default=0)

    # --- Voice interface (Phase 4) ---
    voice_enabled: bool = Field(default=True)

    # STT backend:
    #   whisper = local Faster-Whisper
    #   vosk   = local Vosk fallback
    voice_stt_backend: str = Field(default="whisper")

    # TTS backend:
    #   sapi = Windows SAPI / pyttsx3
    #   null = silent backend for tests
    voice_tts_backend: str = Field(default="sapi")

    # Whisper model.
    # `small.en` is downloaded automatically on first use.
    voice_stt_model_path: str = Field(default="small.en")

    # Faster-Whisper CPU configuration.
    voice_stt_device: str = Field(default="cpu")
    voice_stt_compute_type: str = Field(default="int8")

    voice_sample_rate: int = Field(default=16000)
    voice_channels: int = Field(default=1)

    # Whisper uses "en", not "en-US".
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

    # --- Vision interface (Phase 5) ---
    # Default text model (AI_MODEL, e.g. openai/gpt-oss-120b) is text-only.
    # Image analysis uses VISION_MODEL only, via the same OpenAI-compatible
    # endpoint + API key (no second provider). Verified against Groq docs:
    # gpt-oss-120b is INPUT Text / OUTPUT Text; vision lives on
    # qwen/qwen3.8-27b (20 MB limit, max 3 images, 2048 tokens/image).
    vision_enabled: bool = Field(default=True)
    vision_model: str = Field(default="qwen/qwen3.8-27b")
    vision_max_bytes: int = Field(default=10_485_760)
    vision_max_dim: int = Field(default=2048)
    vision_jpeg_quality: int = Field(default=85)

    def model_dump_safe(self) -> dict:
        """Return settings as a JSON-serializable dict with secrets redacted."""
        data = self.model_dump(mode="json")
        data["ai_api_key"] = (
            "***"
            if self.ai_api_key.get_secret_value()
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