"""Safe read-only system information tools (Phase 2).

All tools here are SAFE: they expose no secrets, tokens, environment
variables, or credentials — only platform facts and NEXUS metadata.
"""

from __future__ import annotations

import os
import platform
import socket
import sys

from pydantic import BaseModel

from app import __version__
from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult


class _NoInput(BaseModel):
    pass


def _base_facts() -> dict:
    return {
        "os": platform.system(),
        "os_release": platform.release(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "cpu_count": os.cpu_count(),
        "working_directory": os.getcwd(),
        "nexus_version": __version__,
    }


def _memory_facts() -> dict:
    """Total/available RAM via ctypes on Windows; graceful fallback elsewhere."""
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
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

        status = _MemStatus()
        status.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {
                "memory_load_percent": status.dwMemoryLoad,
                "total_ram_bytes": status.ullTotalPhys,
                "available_ram_bytes": status.ullAvailPhys,
            }
    except Exception:
        pass
    return {"memory": "unavailable on this platform"}


class SystemInfoTool(Tool):
    name = "system.info"
    description = "Get OS, Python, hostname, CPU count and NEXUS version."
    category = "system"
    risk_level = PermissionLevel.SAFE
    input_model = _NoInput

    def execute(self, validated: BaseModel) -> ToolResult:
        return ToolResult(ok=True, output=_base_facts())


class SystemCpuTool(Tool):
    name = "system.cpu"
    description = "Get CPU facts (architecture, logical core count)."
    category = "system"
    risk_level = PermissionLevel.SAFE
    input_model = _NoInput

    def execute(self, validated: BaseModel) -> ToolResult:
        return ToolResult(
            ok=True,
            output={
                "processor": platform.processor(),
                "machine": platform.machine(),
                "cpu_count": os.cpu_count(),
            },
        )


class SystemMemoryTool(Tool):
    name = "system.memory"
    description = "Get memory statistics (total/available RAM)."
    category = "system"
    risk_level = PermissionLevel.SAFE
    input_model = _NoInput

    def execute(self, validated: BaseModel) -> ToolResult:
        return ToolResult(ok=True, output=_memory_facts())


class SystemPlatformTool(Tool):
    name = "system.platform"
    description = "Get detailed platform identification."
    category = "system"
    risk_level = PermissionLevel.SAFE
    input_model = _NoInput

    def execute(self, validated: BaseModel) -> ToolResult:
        return ToolResult(
            ok=True,
            output={
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "platform": platform.platform(),
            },
        )
