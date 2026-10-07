"""Configuration tests (Phase 1). No real credentials."""

from app.core.config import Settings, get_settings, reload_settings
from app.core.exceptions import ConfigError


def test_defaults_are_sane():
    s = Settings(ai_provider="echo")
    assert s.app_name == "NEXUS"
    assert s.max_history_messages > 0
    assert s.ai_timeout_seconds > 0


def test_unknown_provider_rejected_by_factory():
    from app.brain.provider import create_provider

    s = Settings(ai_provider="definitely-not-a-provider")
    try:
        create_provider(s)
    except ConfigError:
        return
    raise AssertionError("expected ConfigError for unknown provider")


def test_secrets_redacted_in_safe_dump():
    s = Settings(ai_provider="echo", ai_api_key="super-secret-key")
    dumped = s.model_dump_safe()
    assert dumped["ai_api_key"] == "***"
    assert "super-secret-key" not in str(dumped)


def test_safe_dump_is_json_serializable():
    import json

    s = Settings(ai_provider="echo", ai_api_key="super-secret-key")
    dumped = s.model_dump_safe()
    assert json.dumps(dumped)
    assert "super-secret-key" not in json.dumps(dumped)


def test_env_override(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "echo")
    monkeypatch.setenv("AI_MODEL", "env-model")
    s = reload_settings()
    assert s.ai_model == "env-model"
    reload_settings()
    assert get_settings() is not None
