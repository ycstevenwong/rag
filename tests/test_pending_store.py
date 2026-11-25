"""The pending queue must not lose or permanently strand files."""
from __future__ import annotations

import os
import time

import pytest

from rag.pending import PendingStore


@pytest.fixture
def store(tmp_path):
    return PendingStore(tmp_path / "pending")


def _upload(tmp_path, name: str = "up.txt", body: bytes = b"hello"):
    path = tmp_path / name
    path.write_bytes(body)
    return path


def _age(path, seconds: float = 7200) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_failed_metadata_write_keeps_no_half_written_metadata(store, tmp_path, monkeypatch):
    def crash(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", crash)
    with pytest.raises(OSError):
        store.add(_upload(tmp_path), "up.txt")
    monkeypatch.undo()

    assert list(store.dir.glob("*.json")) == []


def test_remove_deletes_the_file_when_metadata_is_corrupt(store, tmp_path):
    item = store.add(_upload(tmp_path), "up.txt")
    (store.dir / f"{item.pending_id}.json").write_text("{not json", encoding="utf-8")

    store.remove(item.pending_id)

    assert list(store.dir.iterdir()) == []


def test_remove_only_touches_its_own_files(store, tmp_path):
    keep = store.add(_upload(tmp_path, "a.txt"), "a.txt")
    gone = store.add(_upload(tmp_path, "b.txt"), "b.txt")

    store.remove(gone.pending_id)
    store.remove("*")

    assert store.get(keep.pending_id) is not None
    assert store.file_path(keep).exists()


def test_cleanup_removes_old_files_with_no_usable_metadata(store, tmp_path):
    orphan = store.dir / "abc123.txt"
    orphan.write_bytes(b"orphan")
    corrupt_file = store.dir / "def456.txt"
    corrupt_file.write_bytes(b"corrupt")
    corrupt_meta = store.dir / "def456.json"
    corrupt_meta.write_text("{not json", encoding="utf-8")
    for path in (orphan, corrupt_file, corrupt_meta):
        _age(path)

    removed = store.cleanup_orphans(max_age_seconds=3600)

    assert removed == 3
    assert list(store.dir.iterdir()) == []


def test_cleanup_keeps_recent_files_that_may_still_be_arriving(store, tmp_path):
    (store.dir / "abc123.txt").write_bytes(b"just moved here")

    assert store.cleanup_orphans(max_age_seconds=3600) == 0
    assert (store.dir / "abc123.txt").exists()


def test_cleanup_keeps_valid_items_of_any_age(store, tmp_path):
    item = store.add(_upload(tmp_path), "up.txt")
    _age(store.file_path(item))
    _age(store.dir / f"{item.pending_id}.json")

    assert store.cleanup_orphans(max_age_seconds=3600) == 0
    assert store.get(item.pending_id) is not None


def test_app_startup_clears_pending_orphans(app_module, data_dirs):
    orphan = data_dirs["PENDING_DIR"] / "abc123.txt"
    orphan.write_bytes(b"orphan")
    _age(orphan)

    app_module.create_app()

    assert not orphan.exists()
