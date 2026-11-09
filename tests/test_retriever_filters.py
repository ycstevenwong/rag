"""Filtering and RRF fusion, using real stores and the stub embedder."""
from __future__ import annotations

import numpy as np
import pytest

from rag.bm25_store import BM25Store
from rag.retriever import HybridRetriever
from rag.vector_store import ChunkRecord, DocRecord, FaissStore, FileRecord

from .conftest import StubEmbedder


def _build(root, files: dict[str, dict]) -> HybridRetriever:
    """files: {file_id: {"texts": [...], "doc": DocRecord kwargs}}"""
    store = FaissStore(root / "v", root / "c", root / "d", root / "f")
    embedder = StubEmbedder()
    next_id = 0
    for file_id, spec in files.items():
        records = []
        for text in spec["texts"]:
            records.append(ChunkRecord(next_id, file_id, text, 5))
            next_id += 1
        store.add(np.stack([embedder.encode_query(r.text) for r in records]), records)
        store.add_file(FileRecord(file_id, file_id, f"{file_id}.pdf", 0.0, len(records)))
        store.add_doc(DocRecord(doc_id=f"doc-{file_id}", file_id=file_id, **spec["doc"]))
    return HybridRetriever(store, BM25Store.from_chunks(store.chunks), embedder)


def test_filter_finds_matches_ranked_below_the_unfiltered_top(tmp_path):
    retriever = _build(tmp_path, {
        "other": {
            "texts": [f"refund policy overview section {i} refund refund" for i in range(500)],
            "doc": {"app_code": "app-a"},
        },
        "target": {
            "texts": [f"refund handled by the billing team note {i}" for i in range(5)],
            "doc": {"app_code": "app-b"},
        },
    })

    results = retriever.search(
        "refund policy overview", k_vector=20, k_bm25=20, top_n=10, filters={"app_code": "app-b"}
    )

    assert sorted(r.chunk.text for r in results) == [
        f"refund handled by the billing team note {i}" for i in range(5)
    ]
    assert {r.doc.doc_id for r in results} == {"doc-target"}


def test_filter_excludes_non_matching_documents(tmp_path):
    retriever = _build(tmp_path, {
        "a": {"texts": ["alpha token"], "doc": {"app_code": "app-a"}},
        "b": {"texts": ["alpha token again"], "doc": {"app_code": "app-b"}},
        "c": {"texts": ["unrelated words"], "doc": {"app_code": "app-c"}},
    })

    results = retriever.search("alpha token", filters={"app_code": "app-a"})

    assert [r.chunk.text for r in results] == ["alpha token"]


def test_filtered_ranks_stop_at_the_k_limits(tmp_path):
    retriever = _build(tmp_path, {
        "a": {
            "texts": [f"shared words number {i}" for i in range(50)],
            "doc": {"app_code": "app-a"},
        },
    })

    results = retriever.search(
        "shared words", k_vector=3, k_bm25=3, top_n=50, filters={"app_code": "app-a"}
    )

    assert 3 <= len(results) <= 6
    assert all(r.vector_rank is None or r.vector_rank < 3 for r in results)
    assert all(r.bm25_rank is None or r.bm25_rank < 3 for r in results)


def test_rrf_scores_sum_reciprocal_ranks_across_both_lists(tmp_path):
    retriever = _build(tmp_path, {
        "a": {
            "texts": ["kilo lima mike", "november oscar papa", "quebec romeo sierra"],
            "doc": {},
        },
    })

    results = retriever.search("kilo lima mike", k_vector=3, k_bm25=3, top_n=3, rrf_k=60)

    top = results[0]
    assert top.chunk.text == "kilo lima mike"
    assert (top.vector_rank, top.bm25_rank) == (0, 0)
    assert top.score == pytest.approx(1 / 61 + 1 / 61)
    others = results[1:]
    assert all(r.bm25_rank is None for r in others)
    assert {r.vector_rank for r in others} == {1, 2}
    assert {round(r.score, 6) for r in others} == {round(1 / 62, 6), round(1 / 63, 6)}
