"""A deleted chunk's ID must never come back: saved citations refer to it."""
from __future__ import annotations

import pytest

from rag.ingest import IngestPipeline
from rag.vector_store import FaissStore

from .conftest import StubEmbedder


def _open_pipeline(root) -> IngestPipeline:
    store = FaissStore(root / "v", root / "c", root / "d", root / "f")
    return IngestPipeline(store, StubEmbedder())


def _ingest(pipeline, root, name: str, text: str) -> None:
    path = root / name
    path.write_text(text, encoding="utf-8")
    for _ in pipeline.ingest_stream(path, name):
        pass


def _delete_newest(pipeline) -> int:
    store = pipeline.vector_store
    deleted_id = store.chunks[-1].id
    pipeline.delete(store.docs[-1].doc_id)
    return deleted_id


def test_deleted_newest_id_is_not_reused(tmp_path):
    pipeline = _open_pipeline(tmp_path)
    _ingest(pipeline, tmp_path, "a.txt", "alpha text")
    _ingest(pipeline, tmp_path, "b.txt", "bravo text")
    deleted_id = _delete_newest(pipeline)

    _ingest(pipeline, tmp_path, "c.txt", "charlie text")

    assert pipeline.vector_store.get_chunk(deleted_id) is None
    assert pipeline.vector_store.chunks[-1].id == deleted_id + 1


def test_deleted_id_is_not_reused_after_a_restart(tmp_path):
    pipeline = _open_pipeline(tmp_path)
    _ingest(pipeline, tmp_path, "a.txt", "alpha text")
    _ingest(pipeline, tmp_path, "b.txt", "bravo text")
    deleted_id = _delete_newest(pipeline)

    restarted = _open_pipeline(tmp_path)
    _ingest(restarted, tmp_path, "c.txt", "charlie text")

    assert restarted.vector_store.get_chunk(deleted_id) is None


def test_ids_are_not_reused_after_deleting_everything(tmp_path):
    pipeline = _open_pipeline(tmp_path)
    _ingest(pipeline, tmp_path, "a.txt", "alpha text")
    deleted_id = _delete_newest(pipeline)

    restarted = _open_pipeline(tmp_path)
    _ingest(restarted, tmp_path, "b.txt", "bravo text")

    assert restarted.vector_store.chunks[0].id > deleted_id


@pytest.mark.parametrize("contents", [None, "not json", '{"next_chunk_id": "x"}', "[]"])
def test_existing_index_without_a_usable_meta_file_continues_from_its_chunks(tmp_path, contents):
    pipeline = _open_pipeline(tmp_path)
    _ingest(pipeline, tmp_path, "a.txt", "alpha text")
    meta = tmp_path / "index_meta.json"
    if contents is None:
        meta.unlink()
    else:
        meta.write_text(contents, encoding="utf-8")

    restarted = _open_pipeline(tmp_path)
    _ingest(restarted, tmp_path, "b.txt", "bravo text")

    assert [c.id for c in restarted.vector_store.chunks] == [0, 1]
