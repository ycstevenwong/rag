"""DOCX parsing against real documents."""
from __future__ import annotations

import pytest

docx = pytest.importorskip("docx")

from docx import Document  # noqa: E402
from docx.oxml import parse_xml  # noqa: E402
from docx.oxml.ns import nsdecls  # noqa: E402

from rag.parsers import parse_docx  # noqa: E402


def _save(doc, tmp_path):
    path = tmp_path / "t.docx"
    doc.save(path)
    return path


def _insert_before_end(doc, element) -> None:
    body = doc.element.body
    body.insert(len(body) - 1, element)  # sectPr has to stay last


def _content_control(inner_xml: str, properties: str = ""):
    return parse_xml(
        f'<w:sdt {nsdecls("w")}><w:sdtPr>{properties}</w:sdtPr>'
        f"<w:sdtContent>{inner_xml}</w:sdtContent></w:sdt>"
    )


def _texts(blocks) -> list[str]:
    return [b.text for b in blocks]


def test_horizontally_merged_cell_appears_once(tmp_path):
    doc = Document()
    table = doc.add_table(2, 3)
    table.cell(0, 0).merge(table.cell(0, 2)).text = "MERGED"
    table.cell(1, 0).text = "a"
    table.cell(1, 2).text = "c"

    assert _texts(parse_docx(_save(doc, tmp_path))) == ["MERGED", "a | c"]


def test_nested_table_rows_are_kept_in_order(tmp_path):
    doc = Document()
    outer = doc.add_table(1, 1)
    cell = outer.cell(0, 0)
    cell.text = "Outer cell"
    inner = cell.add_table(1, 2)
    inner.cell(0, 0).text = "Inner one"
    inner.cell(0, 1).text = "Inner two"

    assert _texts(parse_docx(_save(doc, tmp_path))) == ["Outer cell", "Inner one | Inner two"]


def test_paragraph_inside_a_content_control_is_kept_in_order(tmp_path):
    doc = Document()
    doc.add_paragraph("Before")
    _insert_before_end(doc, _content_control("<w:p><w:r><w:t>Inside control</w:t></w:r></w:p>"))
    doc.add_paragraph("After")

    assert _texts(parse_docx(_save(doc, tmp_path))) == ["Before", "Inside control", "After"]


def test_heading_inside_a_content_control_sets_the_section(tmp_path):
    doc = Document()
    heading_xml = (
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Setup</w:t></w:r></w:p>'
    )
    _insert_before_end(doc, _content_control(heading_xml))
    doc.add_paragraph("Install it.")

    blocks = parse_docx(_save(doc, tmp_path))

    assert blocks[0].kind == "heading"
    assert blocks[1].meta["heading_path"] == "Setup"


def test_table_inside_a_content_control_is_kept(tmp_path):
    doc = Document()
    table = doc.add_table(1, 2)
    table.cell(0, 0).text = "k"
    table.cell(0, 1).text = "v"
    element = table._element
    element.getparent().remove(element)
    control = _content_control("")
    control[-1].append(element)
    _insert_before_end(doc, control)

    assert _texts(parse_docx(_save(doc, tmp_path))) == ["k | v"]


def test_table_of_contents_control_is_skipped(tmp_path):
    doc = Document()
    toc_properties = '<w:docPartObj><w:docPartGallery w:val="Table of Contents"/></w:docPartObj>'
    _insert_before_end(
        doc,
        _content_control("<w:p><w:r><w:t>Setup ..... 3</w:t></w:r></w:p>", toc_properties),
    )
    doc.add_paragraph("Real content")

    assert _texts(parse_docx(_save(doc, tmp_path))) == ["Real content"]
