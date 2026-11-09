"""Corpus text and chat transcripts aren't for anonymous callers."""
from __future__ import annotations

import io
import json

import pytest

import config


@pytest.fixture
def admin(flask_app, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", "unused-hash")
    client = flask_app.test_client()
    with client.session_transaction() as session:
        session["is_admin"] = True
    client.post(
        "/upload",
        data={"file": (io.BytesIO(b"managed secret text"), "a.txt"), "source_type": "other"},
        content_type="multipart/form-data",
    ).get_data()
    return client


def test_anonymous_cannot_read_chunks(admin, client):
    assert client.get("/chunks/0").status_code == 403


def test_admin_can_read_chunks(admin):
    assert admin.get("/chunks/0").json["text"] == "managed secret text"


def test_anonymous_cannot_read_sessions(admin, client):
    client.post("/chat", json={"session_id": "s1", "message": "managed?"}).get_data()

    response = client.get("/sessions/s1")

    assert response.status_code == 403
    assert "managed?" not in response.get_data(as_text=True)


def test_admin_can_read_sessions(admin, client):
    client.post("/chat", json={"session_id": "s1", "message": "managed?"}).get_data()

    assert admin.get("/sessions/s1").json[0]["content"] == "managed?"


def test_anonymous_chat_still_shows_source_snippets(admin, client):
    body = client.post("/chat", json={"session_id": "s1", "message": "managed"}).get_data(as_text=True)

    sources = next(
        event["sources"]
        for event in (json.loads(line.removeprefix("data: ")) for line in body.splitlines() if line)
        if event["type"] == "sources"
    )
    assert sources[0]["snippet"] == "managed secret text"


@pytest.mark.parametrize("session_id", ["!!!", "a/b", "a" * 65, "with space", 123, ["x"]])
def test_chat_rejects_an_invalid_session_id(client, session_id, data_dirs):
    response = client.post("/chat", json={"session_id": session_id, "message": "hi"})

    assert response.status_code == 400
    assert list(data_dirs["SESSION_DIR"].iterdir()) == []


@pytest.mark.parametrize("session_id", ["abc123", "a-b_c", "a" * 64])
def test_chat_accepts_a_valid_session_id(client, session_id):
    response = client.post("/chat", json={"session_id": session_id, "message": "hi"})

    assert response.status_code == 200
    assert f'"session_id": "{session_id}"' in response.get_data(as_text=True)


def test_chat_generates_a_session_id_when_none_is_given(client):
    body = client.post("/chat", json={"message": "hi"}).get_data(as_text=True)

    first_event = json.loads(body.splitlines()[0].removeprefix("data: "))
    assert len(first_event["session_id"]) == 32
