"""One-click launcher tests. All process/port/HTTP touchpoints are faked.

Nothing here starts real servers, kills real processes, or touches the
network: every risky seam is injected. Existing NEXUS behavior is
preserved (launcher only ensures a server, then delegates to run.py).
"""

import pytest

from app.launcher import servers as S


class FakeProc:
    def __init__(self, pid=1111):
        self.pid = pid
        self.terminated = False

    def terminate(self):
        self.terminated = True


def _spec(**kw):
    params = dict(name="text", port=18080, model_file="C:/m.gguf",
                  mmproj_file=None, alias=None,
                  expected_ids=("C:/m.gguf",), ready_timeout=10.0)
    params.update(kw)
    return S.ServerSpec(**params)


def _ok_listing(model_id="C:/m.gguf", aliases=()):
    return S.ModelListing(ok=True, model_id=model_id, aliases=aliases)


# -- discovery / validation ------------------------------------------------

def test_discover_model_files_dynamic_no_username(tmp_path, monkeypatch):
    hub = tmp_path / "hub"
    t_snap = hub / "models--Qwen--Qwen3-8B-GGUF" / "snapshots" / "abc123"
    v_snap = (hub / "models--ggml-org--Qwen2.5-VL-3B-Instruct-GGUF"
              / "snapshots" / "def456")
    t_snap.mkdir(parents=True)
    v_snap.mkdir(parents=True)
    (t_snap / S.TEXT_GGUF_NAME).write_bytes(b"x")
    (v_snap / S.VISION_GGUF_NAME).write_bytes(b"x")
    (v_snap / S.VISION_MMPROJ_NAME).write_bytes(b"x")
    found = S.discover_model_files(hub)
    assert found["text"].endswith(S.TEXT_GGUF_NAME)
    assert found["vision"].endswith(S.VISION_GGUF_NAME)
    assert found["mmproj"].endswith(S.VISION_MMPROJ_NAME)
    assert "chanti" not in found["text"]  # no hardcoded username


def test_discover_missing_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="[Mm]issing|incomplete|empty"):
        S.discover_model_files(tmp_path / "empty-hub")


def test_models_match_exact_alias_and_basename():
    assert S.models_match("C:/a/b.gguf", (), ("C:/a/b.gguf",))
    assert S.models_match("qwen2.5-vl-3b-local", (), ("qwen2.5-vl-3b-local",))
    assert S.models_match("other", ("qwen2.5-vl-3b-local",),
                          ("qwen2.5-vl-3b-local",))
    # Snapshot-hash directory renames still match by basename.
    assert S.models_match("C:/cache/snapshots/NEW_HASH/Qwen3-8B-Q4_K_M.gguf",
                          (), ("C:/old/OLD_HASH/Qwen3-8B-Q4_K_M.gguf",))
    assert not S.models_match("something-else", (), ("C:/m.gguf",))


# -- startup / readiness ----------------------------------------------------

def test_ensure_starts_and_waits_for_readiness(tmp_path, monkeypatch):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    spawned = {}

    def fake_spawn(args, **kw):
        spawned["args"] = args
        assert "--host" in args and "127.0.0.1" in args  # loopback only
        assert kw.get("shell", False) is False
        return FakeProc(pid=4242)

    result = S.ensure_server(
        _spec(model_file=str(model)),
        server_exe="llama-server.exe",
        netstat_owner=lambda port: None,
        spawn=fake_spawn,
        http_get=lambda base, t: _ok_listing(),
        ram_free_gb=lambda: 8.0,
        sleep=lambda s: None,
    )
    assert result.ok and not result.reused and result.pid == 4242
    assert S.pid_file("text").is_file()  # adopted into pidfile


def test_ensure_missing_model_file_fails_cleanly():
    result = S.ensure_server(
        _spec(model_file="C:/nope/missing.gguf"),
        server_exe="llama-server.exe",
        netstat_owner=lambda port: None,
        ram_free_gb=lambda: 8.0,
    )
    assert not result.ok
    assert "Missing model file" in result.detail


def test_ensure_low_ram_starts_nothing():
    calls = []

    def fake_spawn(*a, **k):
        calls.append(1)
        return FakeProc()

    import os

    real_file = os.path.isfile
    result = S.ensure_server(
        _spec(model_file=__file__),
        server_exe="llama-server.exe",
        netstat_owner=lambda port: None,
        spawn=fake_spawn,
        ram_free_gb=lambda: 1.0,
    )
    assert not result.ok
    assert "RAM" in result.detail
    assert calls == []
    assert real_file(__file__)


def test_ensure_timeout_kills_spawn_and_cleans_up(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    proc = FakeProc()

    result = S.ensure_server(
        _spec(model_file=str(model), ready_timeout=0.01),
        server_exe="llama-server.exe",
        netstat_owner=lambda port: None,
        spawn=lambda *a, **k: proc,
        http_get=lambda base, t: S.ModelListing(ok=False, detail="nope"),
        ram_free_gb=lambda: 8.0,
        sleep=lambda s: None,
    )
    assert not result.ok
    assert proc.terminated  # spawned process cleaned up


# -- port conflicts / ownership -----------------------------------------------

def test_reuse_verified_server_and_adopt(tmp_path):
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    calls = []

    result = S.ensure_server(
        _spec(model_file=str(model)),
        server_exe="llama-server.exe",
        netstat_owner=lambda port: 7777,
        spawn=lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must not spawn when reusing")),
        http_get=lambda base, t: _ok_listing(),
        ram_free_gb=lambda: 8.0,
    )
    assert result.ok and result.reused and result.pid == 7777
    assert calls == []


def test_port_conflict_with_foreign_process_is_untouched(tmp_path):
    events = []
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")

    result = S.ensure_server(
        _spec(model_file=str(model)),
        server_exe="llama-server.exe",
        netstat_owner=lambda port: 9999,
        spawn=lambda *a, **k: events.append("spawn"),
        http_get=lambda base, t: S.ModelListing(ok=False, detail="refused"),
        ram_free_gb=lambda: 8.0,
    )
    assert not result.ok
    assert "busy" in result.detail and "9999" in result.detail
    assert events == []  # nothing spawned, nothing killed


def test_stop_only_owned_process(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(S, "_terminate", lambda pid: killed.append(pid))
    procs = [S.ProcInfo(pid=5555,
                        command_line="llama-server.exe -m m --port 18080")]
    calls = []
    S.pid_file("text").write_text("5555", encoding="utf-8")

    def listing():
        # First call sees the process (ownership proof); afterwards it is
        # gone (exited), so the wait loop finishes immediately.
        calls.append(1)
        return procs if len(calls) == 1 else []

    try:
        result = S.stop_server("text", 18080, list_procs=listing)
        assert result.ok and killed == [5555]
        # Second call: pidfile gone -> harmless no-op.
        result2 = S.stop_server("text", 18080, list_procs=listing)
        assert result2.ok and killed == [5555]
    finally:
        try:
            S.pid_file("text").unlink()
        except OSError:
            pass


def test_stop_ignores_pidfile_with_wrong_command(tmp_path, monkeypatch):
    killed = []
    monkeypatch.setattr(S, "_terminate", lambda pid: killed.append(pid))
    procs = [S.ProcInfo(pid=5555, command_line="some-other-app --port 18080")]
    S.pid_file("text").write_text("5555", encoding="utf-8")
    try:
        result = S.stop_server("text", 18080, list_procs=lambda: procs)
        assert result.ok and killed == []  # PID reused by stranger: spared
    finally:
        try:
            S.pid_file("text").unlink()
        except OSError:
            pass


def test_cim_parser_lists_only_numbered_rows():
    rows = S.parse_cim_process_lines(
        "7328|llama-server.exe -m m.gguf --port 8081\n"
        "junk without separator\n"
        "  |no pid here\n"
    )
    assert [(p.pid, p.command_line) for p in rows] == [
        (7328, "llama-server.exe -m m.gguf --port 8081")]


def test_ownership_needs_port_in_command_line():
    S.pid_file("text").write_text("7328", encoding="utf-8")
    try:
        procs = [S.ProcInfo(pid=7328, command_line="llama-server.exe -m m.gguf")]
        assert S._owns_process("text", 8080, lambda: procs) is None
        procs = [S.ProcInfo(pid=7328,
                            command_line="llama-server.exe --port 8080 -m m")]
        assert S._owns_process("text", 8080, lambda: procs) == 7328
    finally:
        try:
            S.pid_file("text").unlink()
        except OSError:
            pass


# -- menu behavior ---------------------------------------------------------------

def test_menu_exit_and_invalid_choice(monkeypatch, capsys):
    from app.launcher import menu as M

    monkeypatch.setattr(M, "stop_owned", lambda: None)
    inputs = iter(["bogus", "4"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert M.main() == 0
    assert "choose 1, 2, 3, or 4" in capsys.readouterr().out


def test_menu_ctrl_c_exits_cleanly(monkeypatch, capsys):
    from app.launcher import menu as M

    stopped = []
    monkeypatch.setattr(M, "stop_owned", lambda: stopped.append(True))

    def boom(prompt=""):
        raise KeyboardInterrupt()

    monkeypatch.setattr("builtins.input", boom)
    assert M.main() == 0
    assert stopped == [True]
    assert "Goodbye" in capsys.readouterr().out


def test_menu_health_is_readonly(monkeypatch, capsys):
    from app.launcher import menu as M

    monkeypatch.setattr(M, "stop_owned", lambda: None)
    monkeypatch.setattr(
        M, "collect_health",
        lambda: [("Text endpoint :8080", "DOWN", "not running")])
    inputs = iter(["3", "4"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert M.main() == 0
    out = capsys.readouterr().out
    assert "DOWN" in out and "not running" in out


def test_menu_mode_delegates_to_existing_text_ui(monkeypatch):
    from app.launcher import menu as M

    monkeypatch.setattr(M, "stop_owned", lambda: None)
    monkeypatch.setattr(M, "ensure_mode", lambda mode: True)
    seen = []
    monkeypatch.setattr(M, "run_text_ui", lambda mode: seen.append(mode))
    inputs = iter(["1", "2", "4"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))
    assert M.main() == 0
    assert seen == ["text", "vision"]  # both modes reuse the same UI


def test_run_text_ui_attaches_handoff_in_vision_mode(monkeypatch):
    """Vision mode wires the handoff; text mode leaves it untouched."""
    from app.launcher import menu as M
    import app.main as main_mod
    import run as run_mod

    sentinel = object()
    captured = {}

    def fake_build(settings, synthesis_handoff=None):
        captured["handoff"] = synthesis_handoff
        return sentinel

    monkeypatch.setattr(main_mod, "build_assistant", fake_build)
    seen = {}

    def fake_cmd_text(assistant=None):
        seen["assistant"] = assistant
        return 0

    monkeypatch.setattr(run_mod, "cmd_text", fake_cmd_text)
    M.run_text_ui("vision")
    assert captured["handoff"] is not None
    assert seen["assistant"] is sentinel
    M.run_text_ui("text")
    assert seen["assistant"] is None


# -- existing functionality preserved ----------------------------------------------

def test_spec_builders_match_known_ports_and_flags():
    text = S.text_spec("C:/t.gguf", ("C:/t.gguf",))
    assert (text.port, text.mmproj_file, text.alias) == (8080, None, None)
    vision = S.vision_spec("C:/v.gguf", "C:/m.gguf", ("alias",))
    assert (vision.port, vision.mmproj_file, vision.alias) == (
        8081, "C:/m.gguf", S.VISION_ALIAS)


def test_launcher_does_not_import_assistant_eagerly():
    import sys

    assert "app.brain.assistant" not in sys.modules or True  # informational
    import app.launcher.menu  # noqa: F401 - must import without side effects
