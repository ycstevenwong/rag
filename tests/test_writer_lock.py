"""Only one process may write the index at a time."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

import config
from rag.writer_lock import WriterLockHeld, acquire_writer_lock

from .conftest import pdf_bytes

ROOT = Path(__file__).resolve().parent.parent

_HOLDER = """
import sys
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from rag.writer_lock import acquire_writer_lock
acquire_writer_lock(Path(sys.argv[2]))
print("held", flush=True)
sys.stdin.read()
"""


@pytest.fixture
def lock_holder():
    """Start another process holding the lock on a path. Returns the starter."""
    started = []

    def start(path: Path) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, "-c", _HOLDER, str(ROOT), str(path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        started.append(proc)
        assert proc.stdout.readline().strip() == "held"
        return proc

    yield start
    for proc in started:
        proc.kill()
        proc.wait()


def test_second_process_is_refused(tmp_path, lock_holder):
    lock = tmp_path / ".lock"
    lock_holder(lock)

    with pytest.raises(WriterLockHeld):
        acquire_writer_lock(lock)


def test_same_process_can_acquire_again(tmp_path):
    lock = tmp_path / ".lock"

    acquire_writer_lock(lock)
    acquire_writer_lock(lock)


def test_lock_clears_when_the_holder_is_killed(tmp_path, lock_holder):
    lock = tmp_path / ".lock"
    holder = lock_holder(lock)
    holder.kill()
    holder.wait()

    acquire_writer_lock(lock)


def test_app_refuses_to_start_when_lock_is_held(app_module, data_dirs, lock_holder):
    lock_holder(data_dirs["INDEX_DIR"] / ".lock")

    with pytest.raises(WriterLockHeld):
        app_module.create_app()


def test_reloader_parent_does_not_take_the_lock(app_module, data_dirs, lock_holder, monkeypatch):
    """With FLASK_DEBUG the parent only watches files; the child process serves."""
    lock_holder(data_dirs["INDEX_DIR"] / ".lock")
    monkeypatch.setattr(config, "FLASK_DEBUG", True)
    monkeypatch.delenv("WERKZEUG_RUN_MAIN", raising=False)

    app_module.create_app()


def test_reloader_child_takes_the_lock(app_module, data_dirs, lock_holder, monkeypatch):
    lock_holder(data_dirs["INDEX_DIR"] / ".lock")
    monkeypatch.setattr(config, "FLASK_DEBUG", True)
    monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true")

    with pytest.raises(WriterLockHeld):
        app_module.create_app()


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ingest_script_exits_when_lock_is_held(data_dirs, lock_holder, monkeypatch, tmp_path, capsys):
    lock_holder(data_dirs["INDEX_DIR"] / ".lock")
    doc = tmp_path / "doc.pdf"
    doc.write_bytes(pdf_bytes("alpha bravo"))
    script = _load_script("ingest_corpus")
    monkeypatch.setattr(sys, "argv", ["ingest_corpus.py", str(doc)])

    assert script.main() == 3
    assert "another process" in capsys.readouterr().err.lower()
    assert not config.CHUNKS_PATH.exists()
