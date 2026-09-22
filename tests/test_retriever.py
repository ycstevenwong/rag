"""End-to-end retriever test using stubbed embeddings (no model download)."""
from __future__ import annotations

import numpy as np
import pytest

from rag.bm25_store import BM25Store
from rag.retriever import HybridRetriever
from rag.vector_store import ChunkRecord, DocRecord, FaissStore, FileRecord


class StubEmbedder:
    dim = 8

    def encode(self, texts):
        return np.array([_text_vec(t) for t in texts], dtype=np.float32)

    def encode_query(self, text):
        return _text_vec(text)


def _text_vec(text: str) -> np.ndarray:
    """Deterministic 8-d bag-of-character-class vector, normalized."""
    v = np.zeros(8, dtype=np.float32)
    for ch in text.lower():
        if ch.isalpha():
            v[ord(ch) % 8] += 1
        elif ch.isdigit():
            v[ord(ch) % 8] += 0.5
    norm = np.linalg.norm(v)
    if norm > 0:
        v /= norm
    return v


def test_hybrid_retrieves_keyword_match(tmp_path):
    pytest.importorskip("faiss")
    vec_store = FaissStore(
        vectors_path=tmp_path / "v.faiss",
        chunks_path=tmp_path / "c.json",
        docs_path=tmp_path / "d.json",
        files_path=tmp_path / "f.json",
    )
    bm25_store = BM25Store(tmp_path / "bm25.pkl")
    embedder = StubEmbedder()

    texts = [
        "Install dependencies via npm install.",
        "Configure the database connection string.",
        "The error code XK-204 indicates an authentication timeout.",
        "Run unit tests with pytest -q.",
    ]
    file_id = "test-file"
    records = [
        ChunkRecord(id=i, file_id=file_id, text=t, token_count=10, meta={"pages": [i + 1]})
        for i, t in enumerate(texts)
    ]
    vectors = embedder.encode(texts)
    vec_store.add(vectors, records)
    vec_store.add_file(FileRecord(
        file_id=file_id, sha256="x", filename="manual.pdf",
        uploaded_at=0.0, n_chunks=len(records),
    ))
    vec_store.add_doc(DocRecord(doc_id="doc-1", file_id=file_id))
    bm25_store.rebuild((r.id, r.text) for r in records)

    retriever = HybridRetriever(vec_store, bm25_store, embedder)
    results = retriever.search("What does error code XK-204 mean?", k_vector=4, k_bm25=4, top_n=2)
    assert results
    assert any("XK-204" in r.chunk.text for r in results)


class _FakeVecStore:
    def __init__(self, vecs: dict[int, np.ndarray]):
        self._vecs = vecs

    def get_vectors(self, ids):
        kept = [i for i in ids if i in self._vecs]
        return kept, np.array([self._vecs[i] for i in kept], dtype=np.float32)


def test_mmr_keeps_bm25_only_hit():
    """A chunk ranked #1 by fusion (via BM25) must survive MMR even when its
    vector is far from the others — MMR relevance is the fused score, not
    query cosine."""
    from rag.retriever import _mmr_select

    near = np.array([1.0, 0.0], dtype=np.float32)
    far = np.array([0.0, 1.0], dtype=np.float32)
    vecs = {0: far, 1: near, 2: near, 3: near}
    fused = {0: 0.033, 1: 0.016, 2: 0.015, 3: 0.014}

    picked = _mmr_select([0, 1, 2, 3], fused, _FakeVecStore(vecs), top_n=2)
    assert picked[0] == 0
