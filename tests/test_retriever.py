"""End-to-end retriever test using stubbed embeddings (no model download)."""
from __future__ import annotations

import numpy as np
import pytest

from rag.bm25_store import BM25Store
from rag.retriever import HybridRetriever
from rag.vector_store import ChunkRecord, FaissStore


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
        dim=StubEmbedder.dim,
    )
    bm25_store = BM25Store(tmp_path / "bm25.pkl")
    embedder = StubEmbedder()

    texts = [
        "Install dependencies via npm install.",
        "Configure the database connection string.",
        "The error code XK-204 indicates an authentication timeout.",
        "Run unit tests with pytest -q.",
    ]
    records = [
        ChunkRecord(id=i, doc_id="doc", filename="manual.pdf", text=t, token_count=10, meta={"pages": [i + 1]})
        for i, t in enumerate(texts)
    ]
    vectors = embedder.encode(texts)
    vec_store.add(vectors, records)
    bm25_store.rebuild((r.id, r.text) for r in records)

    retriever = HybridRetriever(vec_store, bm25_store, embedder)
    results = retriever.search("What does error code XK-204 mean?", k_vector=4, k_bm25=4, top_n=2)
    assert results
    assert any("XK-204" in r.chunk.text for r in results)
