"""A pending upload may only leave the queue once its content is indexed."""
from __future__ import annotations

import io
import json

import pytest
import requests

import config

from .conftest import docx_bytes


@pytest.fixture
def admin(flask_app, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", "unused-hash")
    client = flask_app.test_client()
    with client.session_transaction() as session:
        session["is_admin"] = True
    return client


DOC = docx_bytes("alpha bravo charlie")


def _queue_upload(client, name: str = "doc.docx", body: bytes = DOC) -> str:
    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(body), name), "source_type": "other"},
        content_type="multipart/form-data",
    )
    return json.loads(response.get_data(as_text=True).removeprefix("data: "))["pending_id"]


def _events(response) -> list[dict]:
    lines = response.get_data(as_text=True).splitlines()
    return [json.loads(line.removeprefix("data: ")) for line in lines if line]


def _pending_ids(admin) -> list[str]:
    return [item["pending_id"] for item in admin.get("/admin/pending").json["items"]]


def _embedding_endpoint_down(monkeypatch, app_module):
    def fail(self, texts, *, batch_size=32):
        raise requests.ConnectionError("embedding endpoint down")
        yield  # pragma: no cover

    monkeypatch.setattr(app_module.Embedder, "encode_batched", fail)


def test_successful_approval_removes_the_item(admin, client):
    pending_id = _queue_upload(client)

    events = _events(admin.post(f"/admin/pending/{pending_id}/approve", json={}))

    assert events[-1]["type"] == "done"
    assert _pending_ids(admin) == []


def test_failed_approval_keeps_the_item_and_its_file(admin, client, app_module, monkeypatch):
    pending_id = _queue_upload(client)
    _embedding_endpoint_down(monkeypatch, app_module)

    events = _events(admin.post(f"/admin/pending/{pending_id}/approve", json={}))

    assert events[-1] == {"type": "error", "error": "embedding endpoint down"}
    assert _pending_ids(admin) == [pending_id]
    assert (config.PENDING_DIR / f"{pending_id}.docx").read_bytes() == DOC


def test_failed_approval_can_be_retried(admin, client, app_module, monkeypatch):
    pending_id = _queue_upload(client)
    with monkeypatch.context() as outage:
        _embedding_endpoint_down(outage, app_module)
        admin.post(f"/admin/pending/{pending_id}/approve", json={}).get_data()

    events = _events(admin.post(f"/admin/pending/{pending_id}/approve", json={}))

    assert events[-1]["type"] == "done"
    assert _pending_ids(admin) == []


def test_client_disconnect_mid_approval_keeps_the_item(admin, client):
    pending_id = _queue_upload(client)

    response = admin.post(f"/admin/pending/{pending_id}/approve", json={}, buffered=False)
    next(response.response)  # the client reads one event, then goes away
    response.close()

    assert _pending_ids(admin) == [pending_id]
