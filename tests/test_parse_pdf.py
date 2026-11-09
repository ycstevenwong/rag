"""PDF parsing against real files generated with PyMuPDF."""
from __future__ import annotations

import pytest

fitz = pytest.importorskip("pymupdf")

from rag.parsers import parse_pdf  # noqa: E402

BODY = "Body sentence with enough words to make this the main font of the page."


def _pdf(tmp_path, build, name: str = "t.pdf"):
    doc = fitz.open()
    build(doc)
    path = tmp_path / name
    doc.save(path)
    doc.close()
    return path


def _texts(blocks) -> list[str]:
    return [b.text for b in blocks]


def test_numbered_chapter_headings_are_kept(tmp_path):
    def build(doc):
        for i in range(1, 5):
            page = doc.new_page()
            page.insert_text((72, 60), f"Chapter {i}", fontsize=24)
            page.insert_text((72, 120), f"{BODY} Unique {'abcd'[i - 1]}.", fontsize=11)

    blocks = parse_pdf(_pdf(tmp_path, build))

    assert [b.text for b in blocks if b.kind == "heading"] == [
        "Chapter 1", "Chapter 2", "Chapter 3", "Chapter 4",
    ]
    assert blocks[1].meta["heading_path"] == "Chapter 1"


def test_numbered_body_lines_at_the_top_of_pages_are_kept(tmp_path):
    def build(doc):
        for line in ("Step 1 is mandatory", "Step 2 is mandatory", "Step 3 is mandatory"):
            page = doc.new_page()
            page.insert_text((72, 60), line, fontsize=11)
            page.insert_text((72, 90), BODY, fontsize=11)

    texts = _texts(parse_pdf(_pdf(tmp_path, build)))

    assert "Step 1 is mandatory" in texts
    assert "Step 3 is mandatory" in texts


def test_two_line_footer_in_one_block_is_removed(tmp_path):
    def build(doc):
        for i in range(1, 5):
            page = doc.new_page()
            page.insert_text((72, 120), f"{BODY} Unique {'abcd'[i - 1]}.", fontsize=11)
            page.insert_textbox(
                fitz.Rect(72, 740, 400, 790), f"Company Confidential\nPage {i}", fontsize=8
            )

    joined = " ".join(_texts(parse_pdf(_pdf(tmp_path, build))))

    assert "Confidential" not in joined
    assert "Page" not in joined


@pytest.mark.parametrize("footer", ["Page {i}", "{i}", "Page {i} of 9", "{i} / 9", "Acme Corp Page {i}"])
def test_page_number_footers_are_removed(tmp_path, footer):
    def build(doc):
        for i in range(1, 5):
            page = doc.new_page()
            page.insert_text((72, 120), f"{BODY} Unique {'abcd'[i - 1]}.", fontsize=11)
            page.insert_text((72, 780), footer.format(i=i), fontsize=8)

    texts = _texts(parse_pdf(_pdf(tmp_path, build)))

    assert texts == [f"{BODY} Unique {c}." for c in "abcd"]


def test_identical_running_header_with_digits_is_removed(tmp_path):
    def build(doc):
        for i in range(4):
            page = doc.new_page()
            page.insert_text((72, 40), "Manual Rev 3 2024", fontsize=11)
            page.insert_text((72, 120), f"{BODY} Unique {'abcd'[i]}.", fontsize=11)

    assert "Manual Rev 3 2024" not in _texts(parse_pdf(_pdf(tmp_path, build)))


def test_other_lines_in_a_block_with_a_repeated_line_survive(tmp_path):
    def build(doc):
        for i in range(4):
            page = doc.new_page()
            page.insert_textbox(
                fitz.Rect(72, 40, 500, 110), f"Acme Manual\nFirst line unique {'abcd'[i]}", fontsize=11
            )
            page.insert_text((72, 200), f"{BODY}", fontsize=11)
            page.insert_text((72, 230), "x", fontsize=11)
            page.insert_text((72, 780), "y" * (i + 1), fontsize=11)

    joined = " ".join(_texts(parse_pdf(_pdf(tmp_path, build))))

    assert "Acme Manual" not in joined
    assert "First line unique a" in joined
