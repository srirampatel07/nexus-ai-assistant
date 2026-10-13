"""NVIDIA-only launcher tests (mocked; no processes, ports, or network).

The launcher starts no model servers: there is no server lifecycle module
anymore (`app/launcher/servers.py` and `handoff.py` were removed with local
model operation). These tests assert the cloud menu behavior, the
read-only health report, and the absence of any local-server machinery.
"""

from app.launcher import menu as M


def test_menu_has_no_local_model_choice():
    assert "Text Mode" not in M.MENU
    assert "Vision Mode" not in M.MENU
    assert "Qwen" not in M.MENU
    assert "llama-server" not in M.MENU
    assert "8080" not in M.MENU and "8081" not in M.MENU
    assert "Start NEXUS" in M.MENU


def test_menu_starts_no_local_server_in_normal_path():
    """The cloud launcher must not invoke server startup/shutdown."""
    import inspect

    source = inspect.getsource(M)
    assert "ensure_server(" not in source
    assert "stop_server(" not in source
    assert "switch-model" not in source
    assert "ensure_mode" not in source
    assert "run_text_ui" not in source
    assert "llama-server" not in source.replace(
        "No llama-server.exe is started", "")


def test_local_server_modules_are_gone():
    """Local-model server lifecycle must not exist to be (mis)used."""
    import importlib.util

    assert importlib.util.find_spec("app.launcher.servers") is None
    assert importlib.util.find_spec("app.launcher.handoff") is None


def test_no_local_model_switcher_script():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    assert not (root / "switch-model.ps1").exists()


def test_menu_exit_and_invalid_choice(monkeypatch, capsys):
    inputs = iter(["bogus", "3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert M.main() == 0
    assert "choose 1, 2, or 3" in capsys.readouterr().out


def test_menu_ctrl_c_exits_cleanly(monkeypatch, capsys):
    def boom(prompt=""):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", boom)
    assert M.main() == 0
    assert "Goodbye" in capsys.readouterr().out


def test_menu_health_is_readonly(monkeypatch, capsys):
    monkeypatch.setattr(
        M, "collect_health",
        lambda: [("NVIDIA text", "CONFIGURED", "model=m endpoint=e")])
    inputs = iter(["2", "3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert M.main() == 0
    out = capsys.readouterr().out
    assert "CONFIGURED" in out


def test_menu_start_delegates_to_cloud_ui(monkeypatch):
    seen = []
    monkeypatch.setattr(M, "run_cloud", lambda: seen.append("cloud"))
    inputs = iter(["1", "3"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert M.main() == 0
    assert seen == ["cloud"]


def test_run_cloud_builds_assistant_without_servers(monkeypatch):
    """Cloud start builds the standard assistant and starts no server."""
    import subprocess

    import app.main as main_mod
    import run as run_mod

    def no_server(*a, **k):
        raise AssertionError("cloud path must not start servers")

    monkeypatch.setattr(subprocess, "Popen", no_server)
    captured = {}

    def fake_build(settings=None, conversation_id=None):
        captured["settings"] = settings
        return object()

    monkeypatch.setattr(main_mod, "build_assistant", fake_build)
    seen = {}

    def fake_cmd_text(assistant=None):
        seen["assistant"] = assistant
        return 0

    monkeypatch.setattr(run_mod, "cmd_text", fake_cmd_text)
    M.run_cloud()
    assert captured["settings"] is not None
    assert seen["assistant"] is not None


def test_collect_health_ignores_legacy_local_leftovers(monkeypatch):
    """Loopback URLs / local model ids in .env must not affect cloud health."""
    from app.core.config import Settings

    settings = Settings(
        ai_provider="nvidia",
        ai_model="nexus-default",
        ai_api_key="",
        ai_base_url="",
        nvidia_api_key="test-nvapi-key",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_vision_model="meta/llama-3.2-11b-vision-instruct",
        vision_enabled=True,
        vision_provider="nvidia",
        vision_base_url="http://127.0.0.1:8081/v1",
        vision_model="qwen2.5-vl-3b-local",
        vision_api_key="",
    )
    monkeypatch.setattr(M, "get_settings", lambda: settings)
    rows = M.collect_health()
    labels = [label for label, _, _ in rows]
    assert "NVIDIA text" in labels
    assert "NVIDIA vision" in labels
    assert not any(":8080" in label for label in labels)
    assert not any(":8081" in label for label in labels)
    assert not any("llama-server" in label.lower() for label in labels)
    assert not any("Model files" in label for label in labels)
    by_label = {label: (state, detail) for label, state, detail in rows}
    assert by_label["NVIDIA text"][0] == "CONFIGURED"
    assert by_label["NVIDIA vision"][0] == "CONFIGURED"
    assert "meta/llama-3.2-11b-vision-instruct" in by_label["NVIDIA vision"][1]
    assert "test-nvapi-key" not in str(rows)
    assert by_label["Local servers"][0] == "NOT USED"


def test_collect_health_flags_retired_vision_override(monkeypatch):
    from app.core.config import Settings

    settings = Settings(
        ai_provider="nvidia",
        ai_model="nexus-default",
        ai_api_key="",
        ai_base_url="",
        nvidia_api_key="test-nvapi-key",
        nvidia_base_url="https://integrate.api.nvidia.com/v1",
        nvidia_model="nvidia/nemotron-3.5-lightning-30b-a3b",
        nvidia_vision_model="nvidia/nemotron-nano-12b-v2-vl",
        vision_enabled=True,
    )
    monkeypatch.setattr(M, "get_settings", lambda: settings)
    rows = M.collect_health()
    by_label = {label: (state, detail) for label, state, detail in rows}
    assert by_label["NVIDIA vision"][0] == "CONFIGURED"
    assert "meta/llama-3.2-11b-vision-instruct" in by_label["NVIDIA vision"][1]
    assert "stale override" in by_label["NVIDIA vision"][1]
    assert "test-nvapi-key" not in str(rows)


def test_launcher_does_not_import_assistant_eagerly():
    import app.launcher.menu  # noqa: F401 - must import without side effects
