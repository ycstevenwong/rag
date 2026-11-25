"""Only PDF and DOCX documents can enter the index."""
from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path

import pytest

import config

from .conftest import docx_bytes, pdf_bytes

ROOT = Path(__file__).resolve().parent.parent


def _upload(client, name: str, body: bytes):
    return client.post(
        "/upload",
        data={"file": (io.BytesIO(body), name), "source_type": "other"},
        content_type="multipart/form-data",
    )


@pytest.mark.parametrize("name", ["a.pptx", "a.xlsx", "a.txt", "a.md", "a.doc", "a.xls", "a", "a.PPTX"])
def test_other_file_types_are_rejected(client, data_dirs, name):
    response = _upload(client, name, b"content")

    assert response.status_code == 400
    assert "docx, pdf" in response.json["error"]
    assert list(data_dirs["PENDING_DIR"].iterdir()) == []
    assert list(data_dirs["UPLOAD_DIR"].iterdir()) == []


@pytest.mark.parametrize(
    "name, body",
    [("a.pdf", pdf_bytes("alpha")), ("a.docx", docx_bytes("alpha")), ("A.PDF", pdf_bytes("alpha")), ("A.DocX", docx_bytes("alpha"))],
)
def test_pdf_and_docx_are_accepted_in_any_case(client, name, body):
    response = _upload(client, name, body)

    assert response.status_code == 200
    assert '"type": "queued"' in response.get_data(as_text=True)


def test_rejected_type_suggests_a_conversion(client):
    assert "save as .docx" in _upload(client, "a.doc", b"x").json["error"]
    assert "Convert to .docx or .pdf" in _upload(client, "a.pptx", b"x").json["error"]


def test_upload_form_only_offers_pdf_and_docx(client):
    page = client.get("/").get_data(as_text=True)

    assert 'accept=".pdf,.docx"' in page
    assert "pptx" not in page and "xlsx" not in page


def _load_ingest_script():
    spec = importlib.util.spec_from_file_location("ingest_corpus", ROOT / "scripts" / "ingest_corpus.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bulk_ingest_only_collects_pdf_and_docx(tmp_path):
    for name in ("a.pdf", "b.DOCX", "c.pptx", "d.xlsx", "e.txt", "f.md"):
        (tmp_path / name).write_bytes(b"x")

    found = _load_ingest_script().collect_files(tmp_path)

    assert sorted(p.name for p in found) == ["a.pdf", "b.DOCX"]


def test_a_file_over_the_size_limit_gets_a_readable_reason(app_module, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 1)
    client = app_module.create_app().test_client()

    response = _upload(client, "big.pdf", b"x" * (1024 * 1024 + 1))

    assert response.status_code == 413
    assert response.json["error"] == "The file is larger than the 1 MB upload limit."
