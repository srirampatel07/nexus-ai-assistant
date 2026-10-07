"""NEXUS configuration — environment variables / .env (Phase 1).

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

    # --- AI provider abstraction (provider-independent) ---
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
    # Comma-separated or JSON list in the environment, e.g.
    # NEXUS_ALLOWED_ROOTS="C:\chanti\nexus,C:\chanti\projects"
    nexus_allowed_roots: list[str] = Field(default_factory=lambda: [r"C:\chanti\nexus"])
    nexus_allowed_apps: list[str] = Field(
        default_factory=lambda: ["notepad", "explorer", "powershell", "code"]
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
    memory_retention_days: int = Field(default=0)  # 0 = keep forever

    def model_dump_safe(self) -> dict:
        """Return settings as a JSON-serializable dict with secrets redacted."""
        data = self.model_dump(mode="json")
        data["ai_api_key"] = "***" if self.ai_api_key.get_secret_value() else ""
        return data


@lru_cache
def get_settings() -> Settings:
    """Return cached settings (reads environment / .env once)."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the cache and re-read settings (used in tests)."""
    get_settings.cache_clear()
    return get_settings()
