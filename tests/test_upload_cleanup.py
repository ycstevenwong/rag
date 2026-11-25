"""Startup clears abandoned uploads without touching anything else."""
from __future__ import annotations

import os
import time
from pathlib import Path


def _make_old(path: Path) -> None:
    long_ago = time.time() - 100_000
    os.utime(path, (long_ago, long_ago))


def test_startup_keeps_gitkeep_in_uploads(app_module, data_dirs):
    keep = data_dirs["UPLOAD_DIR"] / ".gitkeep"
    keep.touch()
    _make_old(keep)

    app_module.create_app()

    assert keep.exists()


def test_startup_removes_old_orphan_uploads(app_module, data_dirs):
    orphan = data_dirs["UPLOAD_DIR"] / "abc123.pdf"
    orphan.write_bytes(b"left behind")
    _make_old(orphan)

    app_module.create_app()

    assert not orphan.exists()


def test_startup_keeps_recent_uploads(app_module, data_dirs):
    recent = data_dirs["UPLOAD_DIR"] / "abc123.pdf"
    recent.write_bytes(b"still in progress")

    app_module.create_app()

    assert recent.exists()
