"""Filesystem tool tests (Phase 2). Confined to tmp_path only."""

import pytest

from app.core.exceptions import ToolDenied
from app.tools.filesystem import (
    CreateDirInput,
    DeleteInput,
    FilesystemContext,
    FilesystemCreateDirectoryTool,
    FilesystemDeleteTool,
    FilesystemListTool,
    FilesystemReadTool,
    FilesystemWriteTool,
    ListInput,
    ReadInput,
    WriteInput,
)


@pytest.fixture
def ctx(tmp_path):
    return FilesystemContext(allowed_roots=[str(tmp_path)])


def test_confine_allows_inside(ctx, tmp_path):
    assert ctx.confine(str(tmp_path / "sub" / "f.txt")).parent.name == "sub"


def test_confine_blocks_traversal(ctx, tmp_path):
    with pytest.raises(ToolDenied):
        ctx.confine(str(tmp_path / ".." / "outside.txt"))


def test_confine_blocks_absolute_outside(ctx):
    import os

    outside = os.path.abspath(os.sep)
    with pytest.raises(ToolDenied):
        ctx.confine(outside)


def test_list_directory(ctx, tmp_path):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    tool = FilesystemListTool(ctx)
    result = tool.execute(ListInput(path=str(tmp_path)))
    assert result.ok
    assert "a.txt" in result.output["entries"]
    assert "sub/" in result.output["entries"]


def test_list_missing_dir(ctx, tmp_path):
    result = FilesystemListTool(ctx).execute(ListInput(path=str(tmp_path / "nope")))
    assert not result.ok


def test_read_write_roundtrip(ctx, tmp_path):
    write = FilesystemWriteTool(ctx)
    target = str(tmp_path / "note.txt")
    assert write.execute(WriteInput(path=target, content="hello")).ok
    read = FilesystemReadTool(ctx).execute(ReadInput(path=target))
    assert read.ok
    assert read.output["content"] == "hello"


def test_write_refuses_overwrite_without_flag(ctx, tmp_path):
    target = tmp_path / "note.txt"
    target.write_text("old")
    result = FilesystemWriteTool(ctx).execute(
        WriteInput(path=str(target), content="new")
    )
    assert not result.ok
    assert target.read_text() == "old"


def test_read_refuses_env(ctx, tmp_path):
    secret = tmp_path / ".env"
    secret.write_text("AI_API_KEY=real-key")
    result = FilesystemReadTool(ctx).execute(ReadInput(path=str(secret)))
    assert not result.ok
    assert "sensitive" in result.error.lower()


def test_write_refuses_env(ctx, tmp_path):
    result = FilesystemWriteTool(ctx).execute(
        WriteInput(path=str(tmp_path / ".env"), content="x")
    )
    assert not result.ok


def test_read_refuses_oversize(tmp_path):
    small = FilesystemContext(allowed_roots=[str(tmp_path)], max_read_bytes=4)
    big = tmp_path / "big.txt"
    big.write_text("way too long")
    result = FilesystemReadTool(small).execute(ReadInput(path=str(big)))
    assert not result.ok


def test_read_refuses_binary(ctx, tmp_path):
    raw = tmp_path / "bin.dat"
    raw.write_bytes(bytes(range(256)))
    result = FilesystemReadTool(ctx).execute(ReadInput(path=str(raw)))
    assert not result.ok


def test_create_directory(ctx, tmp_path):
    result = FilesystemCreateDirectoryTool(ctx).execute(
        CreateDirInput(path=str(tmp_path / "a" / "b"))
    )
    assert result.ok
    assert (tmp_path / "a" / "b").is_dir()


def test_delete_file(ctx, tmp_path):
    target = tmp_path / "gone.txt"
    target.write_text("x")
    result = FilesystemDeleteTool(ctx).execute(DeleteInput(path=str(target)))
    assert result.ok
    assert not target.exists()


def test_delete_refuses_root(ctx, tmp_path):
    result = FilesystemDeleteTool(ctx).execute(DeleteInput(path=str(tmp_path)))
    assert not result.ok


def test_delete_refuses_nonempty_dir(ctx, tmp_path):
    full = tmp_path / "full"
    full.mkdir()
    (full / "f.txt").write_text("x")
    result = FilesystemDeleteTool(ctx).execute(DeleteInput(path=str(full)))
    assert not result.ok
    assert full.exists()


def test_tools_reject_outside_root(ctx):
    for tool, payload in [
        (FilesystemListTool(ctx), ListInput(path="C:\\Windows")),
        (FilesystemReadTool(ctx), ReadInput(path="C:\\Windows\\win.ini")),
    ]:
        result = tool.execute(payload)
        assert not result.ok, tool.name
