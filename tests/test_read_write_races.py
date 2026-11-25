"""Searches running during a write must see the old state or the new one, never a mix."""
from __future__ import annotations

import threading

import numpy as np

from rag.vector_store import ChunkRecord, DocRecord, FaissStore, FileRecord


def _unit(i: int, dim: int = 4) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1.0
    return v


class _IndexThatDeletesDuringSearch:
    """Wraps the FAISS index: while a search runs, another thread deletes a doc."""

    def __init__(self, real, on_search):
        self._real = real
        self._on_search = on_search

    def __getattr__(self, name):
        return getattr(self._real, name)

    def search(self, *args, **kwargs):
        self._on_search()
        return self._real.search(*args, **kwargs)


def test_search_during_delete_never_maps_a_vector_to_the_wrong_chunk(tmp_path):
    store = FaissStore(tmp_path / "v", tmp_path / "c", tmp_path / "d", tmp_path / "f")
    for pos, (file_id, text) in enumerate([("gone", "deleted text"), ("kept", "kept text")]):
        store.add(_unit(pos).reshape(1, -1), [ChunkRecord(pos, file_id, text, 2)])
        store.add_file(FileRecord(file_id, file_id, file_id, 0.0, 1))
        store.add_doc(DocRecord(f"doc-{file_id}", file_id))

    deleter = threading.Thread(target=store.delete_doc, args=("doc-gone",))

    def delete_while_searching():
        deleter.start()
        deleter.join(timeout=0.3)  # blocks for the full timeout if search holds the lock

    store._index = _IndexThatDeletesDuringSearch(store._index, delete_while_searching)

    hits = store.search(_unit(0), 1)  # query matches the "gone" vector exactly
    deleter.join()

    assert [c.text for c, _ in hits] == ["deleted text"]
