"""Search is vector-only. rank_bm25 couldn't be installed where the app runs, so
the app must start, index and answer without it."""
from __future__ import annotations

import io
import json
import sys

import config

from .conftest import docx_bytes


def _events(response) -> list[dict]:
    return [json.loads(line[5:]) for line in response.get_data(as_text=True).splitlines() if line.startswith("data:")]


def test_the_app_uploads_and_answers_without_rank_bm25(app_module, monkeypatch):
    # A None entry makes "import rank_bm25" raise ImportError, as if it weren't installed.
    monkeypatch.setitem(sys.modules, "rank_bm25", None)
    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", "unused-hash")
    client = app_module.create_app().test_client()
    with client.session_transaction() as session:
        session["is_admin"] = True

    upload = client.post(
        "/upload",
        data={"file": (io.BytesIO(docx_bytes("The refund window is 30 days.")), "a.docx"), "source_type": "other"},
        content_type="multipart/form-data",
    )
    assert _events(upload)[-1]["type"] == "done"

    chat = client.post("/chat", json={"session_id": "s1", "message": "How long is the refund window?",
                                      "filters": {"app_code": "SG"}})
    events = _events(chat)
    sources = next(e for e in events if e["type"] == "sources")["sources"]
    assert [s["filename"] for s in sources] == ["a.docx"]
    assert events[-1]["type"] == "done"
