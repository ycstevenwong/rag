"""Failures during a chat turn must reach the client as an SSE error event."""
from __future__ import annotations

import json

import pytest
import requests

import config

from .conftest import docx_bytes


def _events(response) -> list[dict]:
    lines = response.get_data(as_text=True).splitlines()
    return [json.loads(line.removeprefix("data: ")) for line in lines if line]


@pytest.fixture
def indexed_client(flask_app, app_module, monkeypatch):
    """A client with one document indexed, so chat actually reaches retrieval."""
    import io

    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", "unused-hash")
    client = flask_app.test_client()
    with client.session_transaction() as session:
        session["is_admin"] = True
    client.post(
        "/upload",
        data={"file": (io.BytesIO(docx_bytes("alpha bravo charlie")), "a.docx"), "source_type": "other"},
        content_type="multipart/form-data",
    ).get_data()
    return client


def _embedding_down(monkeypatch, app_module) -> None:
    def fail(self, _text):
        raise requests.ConnectionError("embedding endpoint down")

    monkeypatch.setattr(app_module.Embedder, "encode_query", fail)


def test_retrieval_failure_sends_an_error_event(indexed_client, app_module, monkeypatch):
    _embedding_down(monkeypatch, app_module)

    events = _events(indexed_client.post("/chat", json={"session_id": "s", "message": "alpha?"}))

    assert [e["type"] for e in events] == ["session", "error"]
    assert "embedding endpoint down" in events[1]["error"]


def test_failed_turn_is_not_saved_to_the_session(indexed_client, app_module, monkeypatch, data_dirs):
    _embedding_down(monkeypatch, app_module)

    indexed_client.post("/chat", json={"session_id": "s", "message": "alpha?"}).get_data()

    assert not (data_dirs["SESSION_DIR"] / "s.json").exists()


def test_successful_turn_still_streams_sources_then_answer(indexed_client):
    events = _events(indexed_client.post("/chat", json={"session_id": "s", "message": "alpha?"}))

    assert [e["type"] for e in events] == ["session", "sources", "delta", "done"]
