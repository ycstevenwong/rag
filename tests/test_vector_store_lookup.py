"""get_chunk is called many times per query, so it can't scan the chunk list."""
from __future__ import annotations

import numpy as np

from rag.vector_store import ChunkRecord, DocRecord, FaissStore, FileRecord


def _store(root) -> FaissStore:
    return FaissStore(root / "v", root / "c", root / "d", root / "f")


def _add(store: FaissStore, chunk_id: int, file_id: str, text: str) -> None:
    vec = np.zeros((1, 8), dtype=np.float32)
    vec[0, chunk_id] = 1.0
    store.add(vec, [ChunkRecord(chunk_id, file_id, text, 1)])
    store.add_file(FileRecord(file_id, file_id, file_id, 0.0, 1))
    store.add_doc(DocRecord(f"doc-{file_id}", file_id))


class _NoScanning(list):
    def __iter__(self):
        raise AssertionError("get_chunk scanned the chunk list")


def test_get_chunk_does_not_scan_the_chunk_list(tmp_path):
    store = _store(tmp_path)
    _add(store, 0, "a", "first")
    store.chunks = _NoScanning(store.chunks)

    assert store.get_chunk(0).text == "first"


def test_get_chunk_finds_added_chunks_and_misses_unknown_ids(tmp_path):
    store = _store(tmp_path)
    _add(store, 3, "a", "three")

    assert store.get_chunk(3).text == "three"
    assert store.get_chunk(4) is None


def test_get_chunk_forgets_deleted_chunks(tmp_path):
    store = _store(tmp_path)
    _add(store, 0, "a", "gone")
    _add(store, 1, "b", "kept")

    store.delete_doc("doc-a")

    assert store.get_chunk(0) is None
    assert store.get_chunk(1).text == "kept"


def test_get_chunk_works_after_a_reload(tmp_path):
    store = _store(tmp_path)
    _add(store, 2, "a", "two")
    store.persist()

    assert _store(tmp_path).get_chunk(2).text == "two"
