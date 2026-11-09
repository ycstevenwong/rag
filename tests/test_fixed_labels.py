"""Functionality and version come from fixed lists, so a rule can't be missed by a typo."""
from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

import config
from rag.metadata import canonical_label

from .conftest import REAL_FUNCTIONALITIES, REAL_VERSIONS, StubEmbedder, docx_bytes, pdf_bytes

ROOT = Path(__file__).resolve().parent.parent
DOC = docx_bytes("alpha bravo charlie")


@pytest.fixture
def lists(monkeypatch):
    monkeypatch.setattr(config, "APP_CODES", ["SG", "MY", "ID"])
    monkeypatch.setattr(config, "FUNCTIONALITIES", ["loan", "other"])
    monkeypatch.setattr(config, "VERSIONS", ["v1", "v2"])


@pytest.fixture
def admin(flask_app, lists, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", "unused-hash")
    client = flask_app.test_client()
    with client.session_transaction() as session:
        session["is_admin"] = True
    return client


def _upload(client, body=DOC, **fields):
    data = dict(source_type="manual", version="v1", functionality="loan")
    data.update(fields)
    return client.post(
        "/upload",
        data={"file": (io.BytesIO(body), "a.docx"), **data},
        content_type="multipart/form-data",
    )


def _doc(client) -> dict:
    return client.get("/docs").json["items"][0]


def test_the_real_lists_cover_the_rules():
    assert REAL_VERSIONS == ["v1", "v2"]
    assert "loan" in REAL_FUNCTIONALITIES


@pytest.mark.parametrize(
    "value, expected",
    [("loan", "loan"), ("Loan", "loan"), (" LOAN ", "loan"), ("other", "other")],
)
def test_canonical_label_returns_the_listed_spelling(value, expected):
    assert canonical_label(value, ["loan", "other"], "functionality") == expected


@pytest.mark.parametrize("value", ["", "  ", "loans", "v3", None, 5])
def test_canonical_label_rejects_anything_else(value):
    with pytest.raises(ValueError, match="functionality"):
        canonical_label(value, ["loan", "other"], "functionality")


def test_canonical_label_allows_anything_when_there_is_no_list():
    assert canonical_label(" Free Text ", [], "functionality") == "Free Text"


@pytest.mark.parametrize(
    "fields",
    [{"version": "v3"}, {"version": ""}, {"functionality": "loans"}, {"functionality": ""}],
)
def test_upload_rejects_unlisted_labels(client, lists, data_dirs, fields):
    response = _upload(client, **fields)

    assert response.status_code == 400
    assert list(data_dirs["PENDING_DIR"].iterdir()) == []
    assert list(data_dirs["UPLOAD_DIR"].iterdir()) == []


@pytest.mark.parametrize("app_code", ["", "Thailand"])
def test_non_manual_upload_needs_a_known_app(client, lists, app_code):
    assert _upload(client, source_type="spec", app_code=app_code).status_code == 400


def test_manual_upload_needs_no_app(client, lists):
    assert _upload(client, source_type="manual", app_code="").status_code == 200


def test_upload_stores_the_listed_spelling(admin):
    response = _upload(admin, version="V2", functionality="LOAN")
    assert '"type": "done"' in response.get_data(as_text=True)

    doc = _doc(admin)
    assert (doc["version"], doc["functionality"]) == ("v2", "loan")


def _queue(client, **fields) -> str:
    response = _upload(client, **fields)
    return json.loads(response.get_data(as_text=True).removeprefix("data: "))["pending_id"]


def test_approval_rejects_an_unlisted_override_and_keeps_the_item(admin, client):
    pending_id = _queue(client)

    response = admin.post(f"/admin/pending/{pending_id}/approve", json={"version": "v9"})

    assert response.status_code == 400
    assert [i["pending_id"] for i in admin.get("/admin/pending").json["items"]] == [pending_id]


def test_approval_stores_the_listed_spelling(admin, client):
    pending_id = _queue(client)

    admin.post(f"/admin/pending/{pending_id}/approve", json={"version": "V2"}).get_data()

    assert _doc(admin)["version"] == "v2"


def test_edit_rejects_an_unlisted_label(admin):
    _upload(admin).get_data()
    doc_id = _doc(admin)["doc_id"]

    assert admin.patch(f"/docs/{doc_id}", json={"version": "v9"}).status_code == 400
    assert admin.patch(f"/docs/{doc_id}", json={"functionality": "loans"}).status_code == 400
    assert (_doc(admin)["version"], _doc(admin)["functionality"]) == ("v1", "loan")


def test_edit_keeps_an_older_label_that_is_not_listed(admin, flask_app, monkeypatch):
    """Documents indexed before the lists existed can still have other fields edited."""
    monkeypatch.setattr(config, "FUNCTIONALITIES", [])
    _upload(admin, functionality="reg-e").get_data()
    monkeypatch.setattr(config, "FUNCTIONALITIES", ["loan", "other"])
    doc_id = _doc(admin)["doc_id"]

    response = admin.patch(f"/docs/{doc_id}", json={"functionality": "reg-e", "version": "v2"})

    assert response.status_code == 200
    assert (_doc(admin)["functionality"], _doc(admin)["version"]) == ("reg-e", "v2")


def test_edit_can_move_an_older_label_onto_the_list(admin, monkeypatch):
    monkeypatch.setattr(config, "FUNCTIONALITIES", [])
    _upload(admin, functionality="reg-e").get_data()
    monkeypatch.setattr(config, "FUNCTIONALITIES", ["loan", "other"])
    doc_id = _doc(admin)["doc_id"]

    assert admin.patch(f"/docs/{doc_id}", json={"functionality": "Other"}).status_code == 200
    assert _doc(admin)["functionality"] == "other"


def test_upload_form_offers_the_lists_as_dropdowns(client, lists):
    page = client.get("/").get_data(as_text=True)

    assert '<select id="upload-version"' in page
    assert '<select id="upload-functionality"' in page
    assert '<option value="loan">loan</option>' in page
    assert '<option value="v2">v2</option>' in page


def _load_ingest_script():
    spec = importlib.util.spec_from_file_location("ingest_corpus", ROOT / "scripts" / "ingest_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bulk_ingest_skips_files_with_unlisted_labels(data_dirs, lists, tmp_path, monkeypatch, capsys):
    good = tmp_path / "Loan_V2_guide.pdf"
    good.write_bytes(pdf_bytes("loan guide text"))
    bad = tmp_path / "mortgage_v9_guide.pdf"
    bad.write_bytes(pdf_bytes("mortgage guide text"))
    script = _load_ingest_script()
    monkeypatch.setattr(script, "Embedder", StubEmbedder)
    monkeypatch.setattr(sys, "argv", ["ingest_corpus.py", str(tmp_path), "--by-filename"])

    assert script.main() == 2

    assert "Choose a version" in capsys.readouterr().err
    docs = json.loads(config.DOCS_PATH.read_text(encoding="utf-8"))
    assert [(d["filename"], d["functionality"], d["version"]) for d in docs] == [
        ("Loan_V2_guide.pdf", "loan", "v2")
    ]
