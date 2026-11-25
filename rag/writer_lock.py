"""Single-writer guard for the index directory.

Each process keeps its own in-memory copy of the index and rewrites the files
on every change, so a second writer would silently discard the first one's
work. The lock is an OS file lock, so the OS releases it when the holder exits,
even after a crash, and a leftover lock file is harmless.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import IO


class WriterLockHeld(RuntimeError):
    """Another process is already writing to this index."""


# Open handles keep the locks alive for the life of the process.
_held: dict[Path, IO] = {}


def _try_lock(handle: IO) -> bool:
    if sys.platform == "win32":
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True

    import fcntl

    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def acquire_writer_lock(path: Path) -> None:
    """Hold the lock at `path` until this process exits. Raises WriterLockHeld
    if another process has it. Calling it again from the holder is a no-op."""
    key = path.resolve()
    if key in _held:
        return
    key.parent.mkdir(parents=True, exist_ok=True)
    handle = key.open("a+")
    if not _try_lock(handle):
        handle.close()
        raise WriterLockHeld(
            f"Another process is using the index in {key.parent} (lock: {key.name}). "
            "Only one process may write it at a time: stop the running app or ingest "
            "script, and run gunicorn with 1 worker (threads are fine)."
        )
    _held[key] = handle
