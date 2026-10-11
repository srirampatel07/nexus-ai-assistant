"""llama-server lifecycle management for the NEXUS one-click launcher.

Safety model (read carefully):

- One large model at a time (16 GB laptop): switching modes stops the
  other server first.
- A process is stopped ONLY when ownership is proven: it must be listed
  in our pidfile AND its command line must still show llama-server with
  the expected port. Anything else occupying a port is reported as a
  conflict and left alone.
- An already-running server is reused ONLY after GET /v1/models confirms
  its model identity matches what we expect (exact id, alias, or file
  basename). On reuse it is adopted into our pidfile.
- Readiness means a successful API response, never just "process exists".
  All waits are bounded; failures clean up what was spawned.
- Servers bind 127.0.0.1 only. No downloads, no installs, no cloud APIs.

All OS touchpoints are injectable parameters so tests use fakes and never
touch real processes or ports.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

HOST = "127.0.0.1"
TEXT_PORT = 8080
VISION_PORT = 8081
TEXT_CTX, TEXT_THREADS, TEXT_BATCH = 4096, 6, 256
VISION_CTX, VISION_THREADS, VISION_BATCH = 4096, 6, 128
MIN_FREE_GB = 3.5

TEXT_GGUF_NAME = "Qwen3-8B-Q4_K_M.gguf"
VISION_GGUF_NAME = "Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf"
VISION_MMPROJ_NAME = "mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf"
VISION_ALIAS = "qwen2.5-vl-3b-local"

# Injectable OS touchpoints (defaults = real implementations).
ListProcs = Callable[[], Sequence["ProcInfo"]]
NetstatOwner = Callable[[int], int | None]
Spawner = Callable[..., "PopenLike"]
HttpGetter = Callable[[str, float], "ModelListing"]
RamProbe = Callable[[], float]


@dataclass(frozen=True)
class ProcInfo:
    pid: int
    command_line: str


@dataclass(frozen=True)
class ModelListing:
    ok: bool
    model_id: str = ""
    aliases: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class ServerSpec:
    """Everything needed to bring one model server up."""

    name: str  # "text" | "vision"
    port: int
    model_file: str
    mmproj_file: str | None
    alias: str | None
    expected_ids: tuple[str, ...]
    ready_timeout: float


@dataclass
class ServerResult:
    ok: bool
    detail: str
    reused: bool = False
    pid: int | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def server_log_dir() -> Path:
    return project_root() / "logs" / "servers"


def pid_file(name: str) -> Path:
    return server_log_dir() / f"{name}.pid"


def hf_hub_dir() -> Path:
    profile = os.environ.get("USERPROFILE") or str(Path.home())
    return Path(profile) / ".cache" / "huggingface" / "hub"


def find_llama_server() -> str:
    found = shutil.which("llama-server") or shutil.which("llama-server.exe")
    if not found:
        raise FileNotFoundError(
            "llama-server.exe not found on PATH. Install llama.cpp "
            "(e.g. via winget) and retry."
        )
    return found


def _first_snapshot(repo: Path, what: str) -> Path:
    snapshots = repo / "snapshots"
    if not snapshots.is_dir():
        raise FileNotFoundError(
            f"Model cache incomplete: {repo} has no snapshots/ "
            f"directory ({what} not downloaded)."
        )
    for child in sorted(snapshots.iterdir()):
        if child.is_dir():
            return child
    raise FileNotFoundError(
        f"Model cache empty: no snapshot under {snapshots} ({what})."
    )


def discover_model_files(hub: Path | None = None) -> dict[str, str]:
    """Locate installed GGUF + projector files. No hardcoded username:
    the hub root derives from USERPROFILE at call time."""
    root = hub or hf_hub_dir()
    text = (
        _first_snapshot(root / "models--Qwen--Qwen3-8B-GGUF", "text model")
        / TEXT_GGUF_NAME
    )
    snap = _first_snapshot(
        root / "models--ggml-org--Qwen2.5-VL-3B-Instruct-GGUF",
        "vision model",
    )
    vision = snap / VISION_GGUF_NAME
    mmproj = snap / VISION_MMPROJ_NAME
    missing = [str(p) for p in (text, vision, mmproj) if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model file(s): " + "; ".join(missing)
        )
    return {"text": str(text), "vision": str(vision), "mmproj": str(mmproj)}


def real_list_llama_processes() -> list[ProcInfo]:
    """Snapshot llama-server processes via CIM (name + command line).

    Uses an encoded PowerShell snippet to avoid quoting pitfalls; wmic is
    intentionally not used (absent from current Windows 11 builds).
    """
    import base64 as _b64
    import subprocess as _sp

    snippet = (
        "Get-CimInstance Win32_Process -Filter \"Name='llama-server.exe'\" "
        "| ForEach-Object { '{0}|{1}' -f $_.ProcessId, $_.CommandLine }"
    )
    encoded = _b64.b64encode(snippet.encode("utf-16-le")).decode("ascii")
    try:
        out = _sp.run(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-EncodedCommand", encoded],
            capture_output=True, text=True, timeout=20,
        ).stdout
    except Exception:
        return []
    return parse_cim_process_lines(out)


def parse_cim_process_lines(out: str) -> list[ProcInfo]:
    """Parse `PID|command line` rows (pure; unit-tested)."""
    procs: list[ProcInfo] = []
    for line in out.splitlines():
        pid_text, _, cmdline = line.partition("|")
        if pid_text.strip().isdigit():
            procs.append(ProcInfo(pid=int(pid_text.strip()),
                                  command_line=cmdline.strip()))
    return procs


def real_netstat_owner(port: int) -> int | None:
    """PID listening on 127.0.0.1:port, or None. Windows netstat parsing."""
    try:
        import subprocess as _sp

        out = _sp.run(["netstat", "-ano", "-p", "TCP"],
                      capture_output=True, text=True, timeout=15).stdout
    except Exception:
        return None
    for line in out.splitlines():
        fields = line.split()
        if (
            len(fields) >= 4
            and fields[0] == "TCP"
            and fields[1] == f"127.0.0.1:{port}"
            and fields[3] == "LISTENING"
            and fields[-1].isdigit()
        ):
            return int(fields[-1])
    return None


def real_ram_free_gb() -> float:
    try:
        import ctypes

        class _Mem(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _Mem()
        status.dwLength = ctypes.sizeof(_Mem)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullAvailPhys / (1024.0 ** 3)
    except Exception:
        pass
    return 0.0


def real_http_models(base_url: str, timeout: float) -> ModelListing:
    """GET {base}/v1/models and extract the served model identity."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{base_url.rstrip('/')}/models")
    except Exception as exc:
        return ModelListing(ok=False, detail=f"{type(exc).__name__}: {exc}")
    if resp.status_code != 200:
        return ModelListing(ok=False, detail=f"HTTP {resp.status_code}")
    try:
        data = resp.json().get("data") or []
        first = data[0]
        model_id = str(first.get("id") or "")
        aliases = tuple(str(a) for a in (first.get("aliases") or []))
    except (ValueError, AttributeError, IndexError, KeyError):
        return ModelListing(ok=False, detail="malformed /v1/models response")
    if not model_id:
        return ModelListing(ok=False, detail="empty model id")
    return ModelListing(ok=True, model_id=model_id, aliases=aliases)


def models_match(reported_id: str, reported_aliases: Sequence[str],
                 expected: Sequence[str]) -> bool:
    """Identity check for server reuse: exact id/alias, or same file
    basename (stable across snapshot-hash directory renames)."""
    candidates = {reported_id, *reported_aliases}
    for want in expected:
        if want in candidates:
            return True
        want_base = want.replace("\\", "/").split("/")[-1]
        for got in candidates:
            if got.replace("\\", "/").split("/")[-1] == want_base:
                return True
    return False


def _pidfile_pid(name: str) -> int | None:
    try:
        text = pid_file(name).read_text(encoding="utf-8").strip()
        return int(text) if text.isdigit() else None
    except OSError:
        return None


def _owns_process(name: str, port: int,
                  list_procs: ListProcs) -> int | None:
    """PID from our pidfile, verified alive as llama-server with our port."""
    pid = _pidfile_pid(name)
    if pid is None:
        return None
    try:
        for proc in list_procs():
            if proc.pid != pid:
                continue
            low = proc.command_line.lower()
            if "llama-server" in low and f"--port {port}" in low.replace("=", " "):
                return pid
            return None
    except Exception:
        return None
    return None


def _terminate(pid: int) -> None:  # pragma: no cover - thin OS wrapper
    import subprocess as _sp

    _sp.run(["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True, timeout=15)


def stop_server(name: str, port: int,
                list_procs: ListProcs | None = None) -> ServerResult:
    """Stop ONLY a launcher-owned server (pidfile + command-line proof).

    Returns ok=True when nothing owned was running. Never touches
    anything else.
    """
    owned = _owns_process(name, port, list_procs or real_list_llama_processes)
    try:
        pid_file(name).unlink(missing_ok=True)
    except OSError:
        pass
    if owned is None:
        return ServerResult(ok=True, detail=f"No {name} server owned by NEXUS.")
    try:
        _terminate(owned)
    except Exception as exc:
        return ServerResult(
            ok=False, detail=f"Could not stop {name} server (PID {owned}): {exc}")
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        try:
            alive = any(p.pid == owned for p in
                        (list_procs or real_list_llama_processes)())
        except Exception:
            break
        if not alive:
            break
        time.sleep(1.0)
    return ServerResult(ok=True, pid=owned,
                        detail=f"Stopped {name} server (PID {owned}).")


def wait_ready(base_url: str, expected: Sequence[str], timeout: float,
               http_get: HttpGetter | None = None,
               sleep: Callable[[float], None] = time.sleep) -> ModelListing:
    """Poll /v1/models until identity matches or the timeout expires."""
    get = http_get or real_http_models
    deadline = time.monotonic() + max(1.0, timeout)
    last = ModelListing(ok=False, detail="not attempted")
    probe_timeout = min(8.0, max(1.0, timeout / 10.0))
    while time.monotonic() < deadline:
        last = get(base_url, probe_timeout)
        if last.ok and models_match(last.model_id, last.aliases, expected):
            return last
        sleep(2.0)
    if last.ok:
        return ModelListing(
            ok=False,
            detail=f"Wrong model on {base_url}: got {last.model_id!r}.")
    return ModelListing(ok=False, detail=f"Server not ready: {last.detail}")


def ensure_server(
    spec: ServerSpec,
    server_exe: str | None = None,
    list_procs: ListProcs | None = None,
    netstat_owner: NetstatOwner | None = None,
    spawn: Spawner | None = None,
    http_get: HttpGetter | None = None,
    ram_free_gb: RamProbe | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ServerResult:
    """Bring one model server up (or adopt a verified identical one)."""
    files = [spec.model_file] + ([spec.mmproj_file] if spec.mmproj_file else [])
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        return ServerResult(
            ok=False, detail="Missing model file(s): " + "; ".join(missing))
    try:
        exe = server_exe or find_llama_server()
    except FileNotFoundError as exc:
        return ServerResult(ok=False, detail=str(exc))
    base_url = f"http://{HOST}:{spec.port}/v1"

    owner_fn = netstat_owner or real_netstat_owner
    try:
        occupant = owner_fn(spec.port)
    except Exception:
        occupant = None
    if occupant is not None:
        # Port busy: reuse ONLY after identity proof, then adopt.
        probe = (http_get or real_http_models)(base_url, 8.0)
        if probe.ok and models_match(probe.model_id, probe.aliases,
                                     spec.expected_ids):
            try:
                server_log_dir().mkdir(parents=True, exist_ok=True)
                pid_file(spec.name).write_text(str(occupant), encoding="utf-8")
            except OSError:
                pass
            return ServerResult(
                ok=True, reused=True, pid=occupant,
                detail=f"Reusing running {spec.name} server (PID {occupant}, "
                       f"model {probe.model_id}).")
        return ServerResult(
            ok=False,
            detail=f"Port {spec.port} is busy (PID {occupant}) with an "
                   f"unverified server. Not stopping it — free the port or "
                   f"stop PID {occupant} yourself, then retry.")

    try:
        free_gb = (ram_free_gb or real_ram_free_gb)()
    except Exception:
        free_gb = 0.0
    if free_gb < MIN_FREE_GB:
        return ServerResult(
            ok=False,
            detail=f"Only {free_gb:.1f} GB RAM free; need {MIN_FREE_GB} GB. "
                   f"Close other apps first. Nothing was started.")

    server_log_dir().mkdir(parents=True, exist_ok=True)
    log_path = server_log_dir() / f"{spec.name}.log"
    args = [exe, "-m", spec.model_file]
    if spec.mmproj_file:
        args += ["--mmproj", spec.mmproj_file]
    if spec.alias:
        args += ["--alias", spec.alias]
    args += ["--host", HOST, "--port", str(spec.port),
             "-c", "4096", "-t", "6", "-b",
             "256" if spec.name == "text" else "128"]
    try:
        log_fh = open(log_path, "ab")
    except OSError as exc:
        return ServerResult(ok=False, detail=f"Cannot write {log_path}: {exc}")
    try:
        proc = (spawn or subprocess.Popen)(
            args, stdout=log_fh, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as exc:
        log_fh.close()
        return ServerResult(ok=False, detail=f"Could not start server: {exc}")
    pid = getattr(proc, "pid", None)
    ready = wait_ready(base_url, spec.expected_ids, spec.ready_timeout,
                       http_get=http_get, sleep=sleep)
    if not ready.ok:
        try:
            proc.terminate()
        except Exception:
            pass
        log_fh.close()
        return ServerResult(
            ok=False, pid=pid,
            detail=f"{spec.name} server failed to become ready: {ready.detail} "
                   f"(log: {log_path}).")
    log_fh.close()
    if pid is not None:
        try:
            pid_file(spec.name).write_text(str(pid), encoding="utf-8")
        except OSError:
            pass
    return ServerResult(
        ok=True, pid=pid,
        detail=f"{spec.name} server ready on port {spec.port} "
               f"(PID {pid}, model {ready.model_id}; log: {log_path}).")


def text_spec(model_file: str, expected_ids: Sequence[str]) -> ServerSpec:
    return ServerSpec(name="text", port=TEXT_PORT, model_file=model_file,
                      mmproj_file=None, alias=None,
                      expected_ids=tuple(expected_ids), ready_timeout=120.0)


def vision_spec(model_file: str, mmproj_file: str,
                expected_ids: Sequence[str]) -> ServerSpec:
    return ServerSpec(name="vision", port=VISION_PORT, model_file=model_file,
                      mmproj_file=mmproj_file, alias=VISION_ALIAS,
                      expected_ids=tuple(expected_ids), ready_timeout=240.0)


def port_busy(port: int,
              netstat_owner: NetstatOwner | None = None) -> bool:
    try:
        return (netstat_owner or real_netstat_owner)(port) is not None
    except Exception:
        return False
