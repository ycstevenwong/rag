"""PPTX parsing against real presentations."""
from __future__ import annotations

import pytest

pptx = pytest.importorskip("pptx")

from pptx import Presentation  # noqa: E402
from pptx.util import Inches  # noqa: E402

from rag.parsers import parse_pptx  # noqa: E402

TITLE_ONLY = 5
BLANK = 6


def _textbox(shapes, text: str):
    box = shapes.add_textbox(Inches(1), Inches(1), Inches(3), Inches(1))
    box.text_frame.text = text
    return box


def _save(prs, tmp_path):
    path = tmp_path / "t.pptx"
    prs.save(path)
    return path


def test_real_title_is_used_even_when_another_shape_comes_first(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[TITLE_ONLY])
    box = _textbox(slide.shapes, "Body text box")
    tree = slide.shapes._spTree
    tree.remove(box._element)
    tree.insert(2, box._element)  # before the title in z-order
    slide.shapes.title.text = "Real Title"

    block = parse_pptx(_save(prs, tmp_path))[0]

    assert block.meta["slide_title"] == "Real Title"
    assert block.text == "Real Title\n\nBody text box"


def test_slide_without_a_title_has_no_title(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[BLANK])
    _textbox(slide.shapes, "Just some body text")

    block = parse_pptx(_save(prs, tmp_path))[0]

    assert block.meta["slide_title"] == ""
    assert block.text == "Just some body text"


def test_table_text_is_kept(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[BLANK])
    table = slide.shapes.add_table(2, 2, Inches(1), Inches(1), Inches(3), Inches(1)).table
    table.cell(0, 0).text = "Code"
    table.cell(0, 1).text = "Meaning"
    table.cell(1, 0).text = "XK-204"
    table.cell(1, 1).text = "Timeout"

    block = parse_pptx(_save(prs, tmp_path))[0]

    assert block.text == "Code | Meaning\nXK-204 | Timeout"


def test_text_in_nested_groups_is_kept(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[BLANK])
    outer = slide.shapes.add_group_shape()
    _textbox(outer.shapes, "In the group")
    inner = outer.shapes.add_group_shape()
    _textbox(inner.shapes, "In the nested group")

    block = parse_pptx(_save(prs, tmp_path))[0]

    assert block.text == "In the group\n\nIn the nested group"


def test_notes_are_kept(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[TITLE_ONLY])
    slide.shapes.title.text = "T"
    slide.notes_slide.notes_text_frame.text = "Speaker notes"

    assert parse_pptx(_save(prs, tmp_path))[0].text == "T\n\nSpeaker notes"


def test_notes_slide_without_a_body_placeholder_does_not_fail(tmp_path):
    prs = Presentation()
    slide = prs.slides.add_slide(prs.slide_layouts[TITLE_ONLY])
    slide.shapes.title.text = "T"
    for placeholder in list(slide.notes_slide.placeholders):
        placeholder._element.getparent().remove(placeholder._element)

    assert parse_pptx(_save(prs, tmp_path))[0].text == "T"
