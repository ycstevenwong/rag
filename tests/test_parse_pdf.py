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


def test_screen_block_keeps_its_line_breaks(tmp_path, monkeypatch):
    from rag import config_facade

    monkeypatch.setattr(config_facade, "PDF_SCREEN_FONTS", "cour")

    def build(doc):
        page = doc.new_page()
        page.insert_text((72, 100), f"{BODY} {BODY}", fontsize=11)
        page.insert_textbox(
            fitz.Rect(72, 200, 500, 300),
            "FIELD-A:  ____\nFIELD-B:  ____\nPF3=EXIT PF8=NEXT",
            fontname="cour",
            fontsize=10,
        )

    screens = [b for b in parse_pdf(_pdf(tmp_path, build)) if b.kind == "screen"]

    assert len(screens) == 1
    assert screens[0].text.splitlines() == ["FIELD-A:  ____", "FIELD-B:  ____", "PF3=EXIT PF8=NEXT"]


def test_paragraph_lines_are_still_joined_with_spaces(tmp_path):
    def build(doc):
        page = doc.new_page()
        page.insert_textbox(fitz.Rect(72, 100, 500, 200), f"{BODY}\nSecond line.", fontsize=11)

    assert parse_pdf(_pdf(tmp_path, build))[0].text == f"{BODY} Second line."


def _heading_paths(tmp_path, sizes_and_titles) -> list[str]:
    def build(doc):
        page = doc.new_page()
        y = 60
        for size, title in sizes_and_titles:
            page.insert_text((72, y), title, fontsize=size)
            page.insert_text((72, y + 25), f"{BODY} {title}", fontsize=11)
            y += 60

    blocks = parse_pdf(_pdf(tmp_path, build))
    return [b.meta["heading_path"] for b in blocks if b.kind == "heading"]


def test_larger_heading_does_not_nest_under_a_smaller_one(tmp_path):
    assert _heading_paths(tmp_path, [(13, "Minor"), (18, "Bigger")]) == ["Minor", "Bigger"]


def test_same_size_headings_are_siblings(tmp_path):
    assert _heading_paths(tmp_path, [(18, "One"), (18, "Two")]) == ["One", "Two"]


def test_smaller_heading_nests_under_a_larger_one(tmp_path):
    assert _heading_paths(tmp_path, [(24, "Top"), (18, "Mid"), (13, "Low"), (18, "Mid2")]) == [
        "Top",
        "Top > Mid",
        "Top > Mid > Low",
        "Top > Mid2",
    ]


def test_encrypted_pdf_gives_a_clear_error(tmp_path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "secret", fontsize=11)
    path = tmp_path / "locked.pdf"
    doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, user_pw="u", owner_pw="o")
    doc.close()

    with pytest.raises(ValueError, match="password"):
        parse_pdf(path)


def test_pdf_with_no_text_gives_no_blocks(tmp_path):
    assert parse_pdf(_pdf(tmp_path, lambda doc: doc.new_page())) == []
