"""Controlled filesystem tools (Phase 2).

Access is confined to configurable allowed roots (default: C:\\chanti\\nexus).
Every path is resolved and must stay inside a root — `..` traversal and
absolute escapes are rejected. Sensitive files (`.env`, private keys) are
never readable, even inside allowed roots, so secrets can't leak into AI
responses. Sizes are capped to avoid runaway reads/writes.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from app.core.exceptions import ToolDenied
from app.security.policy import PermissionLevel
from app.tools.base import Tool, ToolResult

# Basenames (or suffixes) that are never readable/writable through tools.
_SENSITIVE_NAMES = {".env", "id_rsa", "id_ed25519"}
_SENSITIVE_SUFFIXES = (".pem", ".pfx", ".p12", ".key")


def _is_sensitive(path: Path) -> bool:
    name = path.name.lower()
    if name in _SENSITIVE_NAMES or name.startswith(".env."):
        return True
    return name.endswith(_SENSITIVE_SUFFIXES)


class FilesystemContext:
    """Shared confinement config injected into every filesystem tool."""

    def __init__(
        self,
        allowed_roots: list[str],
        max_read_bytes: int = 1_048_576,
        max_write_bytes: int = 1_048_576,
    ) -> None:
        roots = [Path(r).resolve() for r in allowed_roots]
        if not roots:
            raise ValueError("At least one allowed filesystem root is required.")
        self.roots = roots
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes

    def confine(self, raw: str) -> Path:
        """Resolve `raw` and ensure it stays inside an allowed root."""
        candidate = (Path(raw).expanduser().resolve())
        for root in self.roots:
            try:
                if candidate == root or root in candidate.parents:
                    return candidate
            except OSError:
                continue
        raise ToolDenied(
            f"Path '{raw}' is outside the allowed roots: "
            + ", ".join(str(r) for r in self.roots)
        )


class ListInput(BaseModel):
    path: str = Field(description="Directory to list (must be inside allowed roots).")


class ReadInput(BaseModel):
    path: str = Field(description="File to read (must be inside allowed roots).")
    max_chars: int = Field(default=20000, ge=1, le=200000)


class WriteInput(BaseModel):
    path: str = Field(description="File to write (must be inside allowed roots).")
    content: str = Field(description="UTF-8 text content to write.")
    overwrite: bool = Field(default=False)


class CreateDirInput(BaseModel):
    path: str = Field(description="Directory to create (must be inside allowed roots).")


class DeleteInput(BaseModel):
    path: str = Field(description="File or empty directory to delete.")


class FilesystemListTool(Tool):
    name = "filesystem.list"
    description = "List directory contents inside allowed roots."
    category = "filesystem"
    risk_level = PermissionLevel.SAFE
    input_model = ListInput

    def __init__(self, ctx: FilesystemContext) -> None:
        self.ctx = ctx

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, ListInput)
        try:
            target = self.ctx.confine(validated.path)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        if not target.exists():
            return ToolResult(ok=False, error=f"Path does not exist: {target}")
        if not target.is_dir():
            return ToolResult(ok=False, error=f"Not a directory: {target}")
        try:
            entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
        except OSError as exc:
            return ToolResult(ok=False, error=f"Cannot list directory: {exc}")
        return ToolResult(ok=True, output={"path": str(target), "entries": entries[:500]})


class FilesystemReadTool(Tool):
    name = "filesystem.read"
    description = "Read a UTF-8 text file inside allowed roots."
    category = "filesystem"
    risk_level = PermissionLevel.LOW
    input_model = ReadInput

    def __init__(self, ctx: FilesystemContext) -> None:
        self.ctx = ctx

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, ReadInput)
        try:
            target = self.ctx.confine(validated.path)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        if _is_sensitive(target):
            return ToolResult(ok=False, error="Refused: sensitive file (secrets are never exposed).")
        if not target.is_file():
            return ToolResult(ok=False, error=f"Not a file: {target}")
        try:
            if target.stat().st_size > self.ctx.max_read_bytes:
                return ToolResult(ok=False, error="Refused: file exceeds max read size.")
            text = target.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError:
            return ToolResult(ok=False, error="Refused: file is not UTF-8 text.")
        except OSError as exc:
            return ToolResult(ok=False, error=f"Cannot read file: {exc}")
        truncated = len(text) > validated.max_chars
        return ToolResult(
            ok=True,
            output={"path": str(target), "content": text[: validated.max_chars], "truncated": truncated},
        )


class FilesystemWriteTool(Tool):
    name = "filesystem.write"
    description = "Write UTF-8 text to a file inside allowed roots. Needs approval."
    category = "filesystem"
    risk_level = PermissionLevel.MEDIUM
    requires_confirmation = True
    input_model = WriteInput

    def __init__(self, ctx: FilesystemContext) -> None:
        self.ctx = ctx

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, WriteInput)
        try:
            target = self.ctx.confine(validated.path)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        if _is_sensitive(target):
            return ToolResult(ok=False, error="Refused: sensitive file (secrets are never modified).")
        data = validated.content.encode("utf-8")
        if len(data) > self.ctx.max_write_bytes:
            return ToolResult(ok=False, error="Refused: content exceeds max write size.")
        if target.exists() and not validated.overwrite:
            return ToolResult(ok=False, error="File exists. Set overwrite=true to replace it.")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as exc:
            return ToolResult(ok=False, error=f"Cannot write file: {exc}")
        return ToolResult(ok=True, output={"path": str(target), "bytes_written": len(data)})


class FilesystemCreateDirectoryTool(Tool):
    name = "filesystem.create_directory"
    description = "Create a directory (parents included) inside allowed roots. Needs approval."
    category = "filesystem"
    risk_level = PermissionLevel.MEDIUM
    requires_confirmation = True
    input_model = CreateDirInput

    def __init__(self, ctx: FilesystemContext) -> None:
        self.ctx = ctx

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, CreateDirInput)
        try:
            target = self.ctx.confine(validated.path)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            return ToolResult(ok=False, error=f"Already exists: {target}")
        except OSError as exc:
            return ToolResult(ok=False, error=f"Cannot create directory: {exc}")
        return ToolResult(ok=True, output={"path": str(target)})


class FilesystemDeleteTool(Tool):
    name = "filesystem.delete"
    description = "Delete a file or empty directory inside allowed roots. HIGH risk, approval mandatory."
    category = "filesystem"
    risk_level = PermissionLevel.HIGH
    requires_confirmation = True
    input_model = DeleteInput

    def __init__(self, ctx: FilesystemContext) -> None:
        self.ctx = ctx

    def execute(self, validated: BaseModel) -> ToolResult:
        assert isinstance(validated, DeleteInput)
        try:
            target = self.ctx.confine(validated.path)
        except ToolDenied as exc:
            return ToolResult(ok=False, error=str(exc))
        if _is_sensitive(target):
            return ToolResult(ok=False, error="Refused: sensitive file (secrets are never deleted).")
        if target in self.ctx.roots:
            return ToolResult(ok=False, error="Refused: cannot delete an allowed root itself.")
        if not target.exists() and not target.is_symlink():
            return ToolResult(ok=False, error=f"Path does not exist: {target}")
        try:
            if target.is_dir() and not target.is_symlink():
                target.rmdir()  # only removes EMPTY directories
            else:
                target.unlink()
        except OSError as exc:
            return ToolResult(ok=False, error=f"Cannot delete (directory must be empty): {exc}")
        return ToolResult(ok=True, output={"deleted": str(target)})
