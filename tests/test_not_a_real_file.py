"""A file whose extension doesn't match its contents gets a plain reason, without
the server's file path or library internals."""
from __future__ import annotations

import pytest

from rag.parsers import parse


@pytest.mark.parametrize(
    "name, expected",
    [
        ("notes.pdf", "This file isn't a valid PDF. Open it and save it as a PDF again, then re-upload."),
        ("notes.docx", "This file isn't a valid Word document. Open it in Word and save it as .docx, then re-upload."),
    ],
)
def test_a_text_file_renamed_to_a_document_type_says_so(tmp_path, name, expected):
    path = tmp_path / name
    path.write_text("just some plain text, not a document")

    with pytest.raises(ValueError) as caught:
        parse(path)

    assert str(caught.value) == expected


@pytest.fixture
def admin(flask_app, monkeypatch):
    import config

    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", "unused-hash")
    client = flask_app.test_client()
    with client.session_transaction() as session:
        session["is_admin"] = True
    return client


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Fake_Statement.pdf", "This file isn't a valid PDF. Open it and save it as a PDF again, then re-upload."),
        ("Fake_Notes.docx", "This file isn't a valid Word document. Open it in Word and save it as .docx, then re-upload."),
    ],
)
def test_a_staff_suggestion_that_isnt_a_real_document_is_refused_before_the_queue(
    client, admin, data_dirs, name, expected
):
    """It used to be queued, and looked exactly like a real suggestion to the admin."""
    import io

    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(b"just some plain text, not a document"), name), "source_type": "other"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.json == {"error": expected}
    assert admin.get("/admin/pending").json["items"] == []
    assert not [p for p in data_dirs["UPLOAD_DIR"].iterdir() if p.name != ".gitkeep"]


def test_a_real_staff_suggestion_is_still_queued(client, admin):
    import io

    from .conftest import docx_bytes

    response = client.post(
        "/upload",
        data={"file": (io.BytesIO(docx_bytes("alpha bravo")), "real.docx"), "source_type": "other"},
        content_type="multipart/form-data",
    )

    assert '"type": "queued"' in response.get_data(as_text=True)
    assert [i["filename"] for i in admin.get("/admin/pending").json["items"]] == ["real.docx"]
