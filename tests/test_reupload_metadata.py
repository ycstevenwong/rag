"""Uploading known content again must not lose what's new about it."""
from __future__ import annotations

import io
import json

import pytest

import config
from rag.bm25_store import BM25Store
from rag.citations import build_context_block
from rag.ingest import IngestPipeline
from rag.retriever import HybridRetriever
from rag.vector_store import DocRecord, FaissStore

from .conftest import StubEmbedder


def _open_store(root) -> FaissStore:
    return FaissStore(root / "v", root / "c", root / "d", root / "f")


@pytest.fixture
def pipeline(tmp_path):
    return IngestPipeline(_open_store(tmp_path), BM25Store(), StubEmbedder())


@pytest.fixture
def doc_path(tmp_path):
    path = tmp_path / "bytes.txt"
    path.write_text("shared alpha content", encoding="utf-8")
    return path


def _ingest(pipeline, path, name, **kwargs) -> dict:
    last = {}
    for event in pipeline.ingest_stream(path, name, **kwargs):
        last = event
    return last["result"]


def test_new_tags_are_merged_into_a_duplicate(pipeline, doc_path, tmp_path):
    _ingest(pipeline, doc_path, "a.txt", tags=["old", "shared"], version="v1")

    result = _ingest(pipeline, doc_path, "a.txt", tags=["shared", "new"], version="v1")

    assert result["duplicate"] is True
    assert len(pipeline.vector_store.docs) == 1
    assert pipeline.vector_store.docs[0].tags == ["old", "shared", "new"]
    assert _open_store(tmp_path).docs[0].tags == ["old", "shared", "new"]


def test_duplicate_with_no_new_tags_does_not_rewrite_the_index(pipeline, doc_path, monkeypatch):
    _ingest(pipeline, doc_path, "a.txt", tags=["same"])
    writes = []
    monkeypatch.setattr(pipeline.vector_store, "persist", lambda: writes.append(1))

    _ingest(pipeline, doc_path, "a.txt", tags=["same"])

    assert writes == []


def test_linked_upload_keeps_its_own_filename(pipeline, doc_path, tmp_path):
    _ingest(pipeline, doc_path, "guide_v1.txt", version="v1")

    _ingest(pipeline, doc_path, "guide_v2.txt", version="v2")

    names = {d.version: d.filename for d in _open_store(tmp_path).docs}
    assert names == {"v1": "guide_v1.txt", "v2": "guide_v2.txt"}


def test_citation_uses_the_filename_of_the_doc_that_matched(pipeline, doc_path):
    _ingest(pipeline, doc_path, "guide_v1.txt", version="v1", app_code="app-a")
    _ingest(pipeline, doc_path, "guide_v2.txt", version="v2", app_code="app-b")
    retriever = HybridRetriever(pipeline.vector_store, pipeline.bm25_store, StubEmbedder())

    results = retriever.search("shared alpha content", filters={"app_code": "app-b"})

    _, sources = build_context_block(results, token_budget=1000)
    assert sources[0]["filename"] == "guide_v2.txt"
    assert sources[0]["locator"].startswith("guide_v2.txt")


def test_doc_list_shows_each_docs_own_filename(flask_app, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", "unused-hash")
    client = flask_app.test_client()
    with client.session_transaction() as session:
        session["is_admin"] = True
    for name, version in (("one.txt", "v1"), ("two.txt", "v2")):
        client.post(
            "/upload",
            data={
                "file": (io.BytesIO(b"same bytes"), name),
                "source_type": "other",
                "version": version,
            },
            content_type="multipart/form-data",
        ).get_data()

    items = client.get("/docs").json["items"]

    assert {i["version"]: i["filename"] for i in items} == {"v1": "one.txt", "v2": "two.txt"}


def test_docs_json_without_filename_still_loads_and_falls_back(tmp_path, pipeline, doc_path):
    _ingest(pipeline, doc_path, "original.txt", version="v1")
    docs_file = tmp_path / "d"
    stored = json.loads(docs_file.read_text(encoding="utf-8"))
    for doc in stored:
        del doc["filename"]
    docs_file.write_text(json.dumps(stored), encoding="utf-8")

    store = _open_store(tmp_path)
    retriever = HybridRetriever(store, BM25Store.from_chunks(store.chunks), StubEmbedder())
    results = retriever.search("shared alpha content")

    assert isinstance(store.docs[0], DocRecord)
    assert results[0].filename == "original.txt"
