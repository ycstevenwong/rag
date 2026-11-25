"""Section names in citations come from real headings, not from body text,
formulas or glyph names. Cases taken from the FDIC, Basel and FINRA documents."""
from __future__ import annotations

import pytest

fitz = pytest.importorskip("pymupdf")
docx = pytest.importorskip("docx")

from docx import Document  # noqa: E402

from rag.parsers import parse_docx, parse_pdf  # noqa: E402

BODY = "Body sentence with enough words to make this the main font of the page."


def _pdf(tmp_path, build):
    doc = fitz.open()
    build(doc.new_page())
    path = tmp_path / "t.pdf"
    doc.save(path)
    doc.close()
    return path


def _headings(blocks) -> list[str]:
    return [b.text for b in blocks if b.kind == "heading"]


def _body_prose(page, y: float, size: float = 10, lines: int = 6) -> None:
    for i in range(lines):
        page.insert_text((72, y + i * 14), f"{BODY} Line {i}.", fontsize=size)


def test_title_sharing_a_block_with_body_text_is_split_from_it(tmp_path):
    def build(page):
        page.insert_text((72, 100), "SINGLE ACCOUNTS", fontsize=12)
        page.insert_text((72, 114), "A Single Account is a deposit owned by one person", fontsize=10)
        page.insert_text((72, 126), "with no beneficiaries. This category includes:", fontsize=10)
        _body_prose(page, 300)

    blocks = parse_pdf(_pdf(tmp_path, build))

    assert _headings(blocks) == ["SINGLE ACCOUNTS"]
    body = next(b for b in blocks if b.text.startswith("A Single Account"))
    assert body.kind == "paragraph"
    assert body.text.endswith("This category includes:")
    assert body.meta["heading_path"] == "SINGLE ACCOUNTS"


def test_large_formula_brackets_do_not_make_a_heading(tmp_path):
    def build(page):
        page.insert_text((72, 80), "Risk quantification", fontsize=14)
        page.insert_text((72, 120), "(", fontsize=14, fontname="symb")
        page.insert_text((80, 120), "E E LGD LGD", fontsize=10)
        page.insert_text((150, 120), ")", fontsize=14, fontname="symb")
        _body_prose(page, 300)

    blocks = parse_pdf(_pdf(tmp_path, build))

    assert _headings(blocks) == ["Risk quantification"]
    assert {b.meta["heading_path"] for b in blocks} == {"Risk quantification"}


def test_dingbat_bullets_become_bullet_characters(tmp_path):
    def build(page):
        page.insert_text((72, 80), "The FDIC does not insure:", fontsize=12)
        for i, item in enumerate(("Stock Investments", "Bond Investments")):
            page.insert_text((72, 94 + i * 12), "l", fontsize=10, fontname="zadb")
            page.insert_text((84, 94 + i * 12), item, fontsize=10)
        _body_prose(page, 300)

    blocks = parse_pdf(_pdf(tmp_path, build))

    assert _headings(blocks) == ["The FDIC does not insure:"]
    items = next(b for b in blocks if "Stock Investments" in b.text)
    assert items.text == "• Stock Investments • Bond Investments"


def test_bullet_glyph_named_in_a_text_font_becomes_a_bullet_character(tmp_path):
    def build(page):
        page.insert_text((72, 80), "Self-directed plans:", fontsize=12)
        page.insert_text((72, 94), "Bullet", fontsize=10, fontname="tiro")
        page.insert_text((110, 94), "Self-directed 401(k) plan", fontsize=10)
        _body_prose(page, 300)

    blocks = parse_pdf(_pdf(tmp_path, build))

    assert any(b.text == "• Self-directed 401(k) plan" for b in blocks)


def test_small_table_figures_do_not_set_the_body_size(tmp_path):
    """Table cells are short, so they shouldn't decide what size normal text is."""
    def build(page):
        page.insert_text((72, 60), "Insured Institution Performance", fontsize=14)
        _body_prose(page, 90, size=9, lines=4)
        for row in range(40):
            for col in range(6):
                page.insert_text((72 + col * 80, 200 + row * 9), f"{row},{col}38.5", fontsize=7)

    blocks = parse_pdf(_pdf(tmp_path, build))

    assert _headings(blocks) == ["Insured Institution Performance"]


def test_text_without_a_word_is_not_a_heading(tmp_path):
    def build(page):
        page.insert_text((72, 200), "Q2", fontsize=120)
        page.insert_text((72, 300), "Quarterly Banking Profile", fontsize=16)
        _body_prose(page, 400)

    assert _headings(parse_pdf(_pdf(tmp_path, build))) == ["Quarterly Banking Profile"]


def test_heading_wrapped_onto_a_line_of_digits_is_kept_whole(tmp_path):
    def build(page):
        page.insert_text((72, 80), "Documentation of Transfers - 12 CFR", fontsize=14)
        page.insert_text((72, 96), "1005.9", fontsize=14)
        _body_prose(page, 300)

    assert _headings(parse_pdf(_pdf(tmp_path, build))) == ["Documentation of Transfers - 12 CFR 1005.9"]


def test_docx_sentence_in_a_heading_style_is_body_text(tmp_path):
    doc = Document()
    doc.add_heading("5. Customer Identification Program", level=1)
    sentence = (
        "What this means for you: When you open an account, we will ask for your name, "
        "address, date of birth and other information."
    )
    doc.add_heading(sentence, level=2)
    path = tmp_path / "t.docx"
    doc.save(path)

    blocks = parse_docx(path)

    assert blocks[-1].kind == "paragraph"
    assert blocks[-1].meta["heading_path"] == "5. Customer Identification Program"


def test_docx_heading_whitespace_is_collapsed(tmp_path):
    doc = Document()
    doc.add_heading("5.\tCustomer Identification Program", level=1)
    doc.add_heading("c.\t Verifying  Information", level=2)
    doc.add_paragraph("Body text.")
    path = tmp_path / "t.docx"
    doc.save(path)

    blocks = parse_docx(path)

    assert blocks[-1].meta["heading_path"] == "5. Customer Identification Program > c. Verifying Information"
