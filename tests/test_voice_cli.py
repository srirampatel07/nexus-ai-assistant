"""Voice CLI tests (Phase 4). Headless script mode: no mic, speaker, or Groq."""

import run
from app.core.config import reload_settings


def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AI_PROVIDER", "echo")
    monkeypatch.setenv("AI_MODEL", "test-model")
    monkeypatch.setenv("MEMORY_DATABASE_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("VOICE_TTS_BACKEND", "null")
    reload_settings()


def test_voice_disabled_reports_cleanly(monkeypatch):
    monkeypatch.setenv("VOICE_ENABLED", "false")
    reload_settings()
    try:
        assert run.main(["--voice"]) == 2
    finally:
        reload_settings()


def test_voice_script_headless_pipeline(monkeypatch, tmp_path, capsys):
    _isolated_env(monkeypatch, tmp_path)
    try:
        assert run.main(["--voice", "--script", "hello | status"]) == 0
    finally:
        reload_settings()
    out = capsys.readouterr().out
    assert "NEXUS Voice Mode" in out
    assert "STT: script" in out
    assert "Microphone: available (scripted" in out
    assert "NEXUS (voice)> Hello. How can I help?" in out


def test_voice_script_stop_phrase(monkeypatch, tmp_path, capsys):
    _isolated_env(monkeypatch, tmp_path)
    try:
        assert run.main(["--voice", "--script", "stop listening"]) == 0
    finally:
        reload_settings()
    assert "Goodbye." in capsys.readouterr().out


def test_text_mode_still_works(monkeypatch, tmp_path, capsys):
    _isolated_env(monkeypatch, tmp_path)
    monkeypatch.setattr("builtins.input", lambda _prompt="": "/exit")
    try:
        assert run.main(["--text"]) == 0
    finally:
        reload_settings()
    assert "text mode" in capsys.readouterr().out
