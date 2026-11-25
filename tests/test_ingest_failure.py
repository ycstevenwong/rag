"""A failed commit must not leave the failed document behind."""
from __future__ import annotations

import pytest

from rag.ingest import IngestPipeline
from rag.vector_store import FaissStore

from .conftest import StubEmbedder


def _open_store(root) -> FaissStore:
    return FaissStore(root / "v", root / "c", root / "d", root / "f")


@pytest.fixture
def pipeline(tmp_path):
    store = _open_store(tmp_path)
    return IngestPipeline(store, StubEmbedder())


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _drain(stream) -> dict:
    last = {}
    for event in stream:
        last = event
    return last


def _fail_next_persist(store, monkeypatch) -> None:
    real_persist = store.persist
    state = {"failed": False}

    def persist():
        if not state["failed"]:
            state["failed"] = True
            raise OSError("disk full")
        real_persist()

    monkeypatch.setattr(store, "persist", persist)


def test_failed_commit_leaves_no_trace_in_memory(pipeline, tmp_path, monkeypatch):
    store = pipeline.vector_store
    _drain(pipeline.ingest_stream(_write(tmp_path, "kept.txt", "kept alpha text"), "kept.txt"))
    _fail_next_persist(store, monkeypatch)

    with pytest.raises(OSError):
        _drain(pipeline.ingest_stream(_write(tmp_path, "bad.txt", "failed bravo text"), "bad.txt"))

    assert [c.text for c in store.chunks] == ["kept alpha text"]
    assert [f.filename for f in store.files] == ["kept.txt"]
    assert len(store.docs) == 1


def test_failed_commit_is_not_written_by_the_next_persist(pipeline, tmp_path, monkeypatch):
    store = pipeline.vector_store
    _fail_next_persist(store, monkeypatch)
    with pytest.raises(OSError):
        _drain(pipeline.ingest_stream(_write(tmp_path, "bad.txt", "failed bravo text"), "bad.txt"))

    _drain(pipeline.ingest_stream(_write(tmp_path, "good.txt", "good charlie text"), "good.txt"))

    reloaded = _open_store(tmp_path)
    assert [c.text for c in reloaded.chunks] == ["good charlie text"]


def test_same_file_can_be_retried_after_a_failed_commit(pipeline, tmp_path, monkeypatch):
    store = pipeline.vector_store
    path = _write(tmp_path, "retry.txt", "retry delta text")
    _fail_next_persist(store, monkeypatch)
    with pytest.raises(OSError):
        _drain(pipeline.ingest_stream(path, "retry.txt"))

    result = _drain(pipeline.ingest_stream(path, "retry.txt"))["result"]

    assert result["duplicate"] is False
    assert result["n_chunks"] == 1
    hits = store.search(StubEmbedder().encode_query("retry delta text"), 1)
    assert [(c.text, round(score, 3)) for c, score in hits] == [("retry delta text", 1.0)]


def test_failed_link_leaves_no_doc_record(pipeline, tmp_path, monkeypatch):
    store = pipeline.vector_store
    path = _write(tmp_path, "shared.txt", "shared echo text")
    _drain(pipeline.ingest_stream(path, "shared.txt", version="v1"))
    _fail_next_persist(store, monkeypatch)

    with pytest.raises(OSError):
        _drain(pipeline.ingest_stream(path, "shared.txt", version="v2"))

    assert [d.version for d in store.docs] == ["v1"]
    assert [c.text for c in store.chunks] == ["shared echo text"]
