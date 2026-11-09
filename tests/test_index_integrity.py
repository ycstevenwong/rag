"""The FAISS index and chunk list must stay position-aligned across restarts."""
from __future__ import annotations

import numpy as np
import pytest

from rag.vector_store import ChunkRecord, FaissStore, IndexIntegrityError


def _store(root) -> FaissStore:
    return FaissStore(root / "v.faiss", root / "c.json", root / "d.json", root / "f.json")


def _unit(i: int) -> np.ndarray:
    v = np.zeros((1, 4), dtype=np.float32)
    v[0, i] = 1.0
    return v


def _persisted_store(root) -> FaissStore:
    store = _store(root)
    store.add(_unit(0), [ChunkRecord(id=0, file_id="f", text="zero", token_count=1)])
    store.persist()
    return store


def test_load_refuses_more_vectors_than_chunks(tmp_path, monkeypatch):
    """The crash window: vectors.faiss replaced, chunks.json not yet written."""
    import rag.vector_store as vs

    store = _persisted_store(tmp_path)
    store.add(_unit(1), [ChunkRecord(id=1, file_id="f", text="one", token_count=1)])

    def crash(*_args, **_kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(vs, "_atomic_write_text", crash)
    with pytest.raises(OSError):
        store.persist()
    monkeypatch.undo()

    with pytest.raises(IndexIntegrityError, match=r"2 vectors.*1 chunks"):
        _store(tmp_path)


def test_load_refuses_more_chunks_than_vectors(tmp_path):
    store = _persisted_store(tmp_path)
    store.vectors_path.unlink()

    with pytest.raises(IndexIntegrityError, match=r"0 vectors.*1 chunks"):
        _store(tmp_path)


def test_load_accepts_an_aligned_index(tmp_path):
    _persisted_store(tmp_path)

    reloaded = _store(tmp_path)

    hits = reloaded.search(_unit(0), 1)
    assert [c.text for c, _ in hits] == ["zero"]


def test_load_accepts_an_empty_directory(tmp_path):
    assert _store(tmp_path).chunks == []
