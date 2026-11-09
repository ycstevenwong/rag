"""Uploaded documents and local IDE files must stay out of git."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git isn't installed")


def _is_ignored(relative_path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", relative_path], cwd=ROOT, capture_output=True
    )
    assert result.returncode in (0, 1), result.stderr
    return result.returncode == 0


@pytest.mark.parametrize(
    "path",
    [
        "data/pending/upload.pdf",
        "data/pending/upload.json",
        "data/corpus/manual/guide.pdf",
        "data/uploads/upload.pdf",
        "data/sessions/s.json",
        "data/index/chunks.json",
        "data/index/.lock",
        ".idea/project.iml",
    ],
)
def test_private_and_local_files_are_ignored(path):
    assert _is_ignored(path)


@pytest.mark.parametrize(
    "path",
    ["data/eval/eval.json", "data/uploads/.gitkeep", "data/index/.gitkeep", "data/sessions/.gitkeep"],
)
def test_tracked_files_are_not_ignored(path):
    assert not _is_ignored(path)
