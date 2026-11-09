"""BM25 is derived from the chunks at load, never read from a pickle."""
from __future__ import annotations

import pickle

import config
from rag.bm25_store import BM25Store
from rag.vector_store import ChunkRecord


class _WritesSentinelWhenUnpickled:
    def __init__(self, sentinel: str):
        self.sentinel = sentinel

    def __reduce__(self):
        return (open, (self.sentinel, "w"))


def test_starting_the_app_does_not_unpickle_an_existing_bm25_file(app_module, data_dirs):
    sentinel = data_dirs["DATA_DIR"] / "code-ran"
    legacy_file = data_dirs["INDEX_DIR"] / "bm25.pkl"
    legacy_file.write_bytes(pickle.dumps(_WritesSentinelWhenUnpickled(str(sentinel))))

    app_module.create_app()

    assert not sentinel.exists()


def test_store_built_from_chunks_finds_their_text():
    chunks = [
        ChunkRecord(id=7, file_id="f", text="error code XK204 means timeout", token_count=5),
        ChunkRecord(id=9, file_id="f", text="install with pip", token_count=3),
        ChunkRecord(id=11, file_id="f", text="run the tests", token_count=3),
    ]

    store = BM25Store.from_chunks(chunks)

    assert [cid for cid, _ in store.search("XK204", 5)] == [7]


def test_store_built_from_no_chunks_returns_nothing():
    assert BM25Store.from_chunks([]).search("anything", 5) == []


def test_ingest_writes_no_bm25_file(client, flask_app, app_module, data_dirs, monkeypatch):
    import io

    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", "unused-hash")
    with client.session_transaction() as session:
        session["is_admin"] = True

    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"alpha bravo"), "a.txt"), "source_type": "other"},
        content_type="multipart/form-data",
    )

    assert '"type": "done"' in response.get_data(as_text=True)
    assert not list(data_dirs["INDEX_DIR"].glob("bm25*"))
