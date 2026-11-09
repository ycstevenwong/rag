"""Ingests running at the same time must not corrupt each other.

Generators are advanced by hand to reproduce the interleaving that threaded SSE
requests produce: both past their dedupe check and embedding before either one
commits.
"""
from __future__ import annotations

import pytest

from rag.bm25_store import BM25Store
from rag.ingest import IngestPipeline
from rag.vector_store import FaissStore

from .conftest import StubEmbedder


def _advance_to_embedding(stream) -> None:
    for event in stream:
        if event.get("stage") == "Embedding":
            return
    raise AssertionError("stream ended before the Embedding stage")


def _drain(stream) -> dict:
    last = {}
    for event in stream:
        last = event
    return last


@pytest.fixture
def pipeline(tmp_path):
    store = FaissStore(
        tmp_path / "v.faiss", tmp_path / "c.json", tmp_path / "d.json", tmp_path / "f.json"
    )
    return IngestPipeline(store, BM25Store(tmp_path / "b.pkl"), StubEmbedder())


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _interleave(pipeline, tmp_path):
    first = pipeline.ingest_stream(_write(tmp_path, "x.txt", "xray document text"), "x.txt")
    second = pipeline.ingest_stream(_write(tmp_path, "y.txt", "yankee document text"), "y.txt")
    _advance_to_embedding(first)
    _advance_to_embedding(second)
    _drain(first)
    _drain(second)


def test_interleaved_ingests_get_distinct_chunk_ids(pipeline, tmp_path):
    _interleave(pipeline, tmp_path)

    store = pipeline.vector_store
    assert sorted(c.id for c in store.chunks) == [0, 1]
    assert {store.get_chunk(c.id).text for c in store.chunks} == {
        "xray document text",
        "yankee document text",
    }


def test_deleting_one_interleaved_doc_keeps_the_others_vector(pipeline, tmp_path):
    _interleave(pipeline, tmp_path)
    store = pipeline.vector_store
    yankee_doc = next(
        d.doc_id for d in store.docs if store.get_file(d.file_id).filename == "y.txt"
    )

    pipeline.delete(yankee_doc)

    hits = store.search(StubEmbedder().encode_query("xray document text"), 1)
    assert [(c.text, round(score, 3)) for c, score in hits] == [("xray document text", 1.0)]


def test_same_file_ingested_concurrently_is_stored_once(pipeline, tmp_path):
    path = _write(tmp_path, "same.txt", "shared document text")
    first = pipeline.ingest_stream(path, "same.txt", version="v1")
    second = pipeline.ingest_stream(path, "same.txt", version="v2")
    _advance_to_embedding(first)
    _advance_to_embedding(second)
    _drain(first)
    result = _drain(second)["result"]

    store = pipeline.vector_store
    assert len(store.files) == 1
    assert len(store.chunks) == 1
    assert sorted(d.version for d in store.docs) == ["v1", "v2"]
    assert result["linked"] is True
