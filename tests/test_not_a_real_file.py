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
