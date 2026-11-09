"""Replaced files must be on disk before the rename, or a power loss can empty them."""
from __future__ import annotations

import os
import sys

import pytest

from rag.vector_store import atomic_write_bytes, atomic_write_text


@pytest.fixture
def disk_calls(monkeypatch):
    """Records fsync and replace in order, then runs the real calls. An fsync entry
    says what was synced: "file" holding `content`, or "directory"."""
    calls: list[tuple[str, str]] = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        if os.path.isdir(f"/dev/fd/{fd}"):
            calls.append(("fsync", "directory"))
        else:
            calls.append(("fsync", "file"))
        real_fsync(fd)

    def replace(src, dst):
        calls.append(("replace", "file"))
        real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    return calls


def _write_text(path):
    atomic_write_text(path, "new contents")


def _write_bytes(path):
    atomic_write_bytes(path, lambda tmp: tmp.write_text("new contents"))


@pytest.mark.parametrize("write", [_write_text, _write_bytes])
def test_data_is_synced_before_the_rename(tmp_path, disk_calls, write):
    write(tmp_path / "f")

    assert disk_calls[:2] == [("fsync", "file"), ("replace", "file")]


@pytest.mark.skipif(sys.platform == "win32", reason="Windows can't fsync a directory")
@pytest.mark.parametrize("write", [_write_text, _write_bytes])
def test_directory_is_synced_after_the_rename(tmp_path, disk_calls, write):
    write(tmp_path / "f")

    assert disk_calls[2:] == [("fsync", "directory")]


@pytest.mark.parametrize("write", [_write_text, _write_bytes])
def test_contents_are_written(tmp_path, write):
    path = tmp_path / "f"

    write(path)

    assert path.read_text() == "new contents"
    assert [p.name for p in tmp_path.iterdir()] == ["f"]
