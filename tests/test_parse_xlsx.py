"""XLSX parsing and chunking against real workbooks."""
from __future__ import annotations

import pytest

openpyxl = pytest.importorskip("openpyxl")

from rag.chunker import chunk_blocks  # noqa: E402
from rag.citations import _format_locator  # noqa: E402
from rag.parsers import parse_xlsx  # noqa: E402


def _workbook(tmp_path, sheets: dict[str, list[list]]):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for title, rows in sheets.items():
        sheet = wb.create_sheet(title)
        for row in rows:
            sheet.append(row)
    path = tmp_path / "t.xlsx"
    wb.save(path)
    return path


def test_blank_rows_are_skipped(tmp_path):
    path = _workbook(tmp_path, {"S": [
        ["id", "name"], [1, "one"], [None, None], [None, None], [2, "two"],
    ]})

    blocks = parse_xlsx(path)

    assert blocks[0].text == "Headers: id | name\n1 | one\n2 | two"
    assert (blocks[0].meta["row_start"], blocks[0].meta["row_end"]) == (2, 5)


def test_groups_count_rows_with_data_only(tmp_path):
    rows = [["id", "v"]]
    for i in range(20):
        rows += [[i, "x"], [None, None]]
    path = _workbook(tmp_path, {"S": rows})

    blocks = parse_xlsx(path)

    assert len(blocks) == 1
    assert blocks[0].text.count("\n") == 20


def test_sheet_with_a_single_row_is_kept(tmp_path):
    path = _workbook(tmp_path, {"Only": [["solo fact", "only row"]]})

    blocks = parse_xlsx(path)

    assert [b.text for b in blocks] == ["solo fact | only row"]
    assert (blocks[0].meta["row_start"], blocks[0].meta["row_end"]) == (1, 1)


def test_chunks_do_not_span_sheets(tmp_path):
    path = _workbook(tmp_path, {
        "Zeta": [["id"], [1], [2], [3], [4]],
        "Alpha": [["k"], [100], [101]],
    })

    chunks = chunk_blocks(parse_xlsx(path), target_tokens=600, overlap_tokens=0)

    assert [_format_locator("t.xlsx", c.meta) for c in chunks] == [
        "t.xlsx, sheet Zeta rows 2–5",
        "t.xlsx, sheet Alpha rows 2–3",
    ]


def test_oversized_group_splits_by_row_keeping_headers_and_rows(tmp_path):
    long_cell = "lorem ipsum. " * 30
    rows = [["id", "text"]] + [[i, long_cell] for i in range(20)]
    path = _workbook(tmp_path, {"Wide": rows})

    chunks = chunk_blocks(parse_xlsx(path), target_tokens=300, overlap_tokens=0)

    assert len(chunks) > 1
    assert all(c.text.startswith("Headers: id | text\n") for c in chunks)
    # Each chunk's row_range covers exactly the rows it holds, with no gaps.
    covered = []
    for c in chunks:
        first, last = c.meta["row_range"]
        assert c.text.count("\n") == last - first + 1
        covered.extend(range(first, last + 1))
    assert covered == list(range(2, 22))


def test_single_row_larger_than_the_target_still_gets_its_row_number(tmp_path):
    rows = [["id", "text"], [1, "short"], [2, "word " * 2000], [3, "short"]]
    path = _workbook(tmp_path, {"S": rows})

    chunks = chunk_blocks(parse_xlsx(path), target_tokens=300, overlap_tokens=0)

    joined = "\n".join(c.text for c in chunks)
    assert "1 | short" in joined
    assert "3 | short" in joined
    assert all(c.text.startswith("Headers: id | text\n") for c in chunks)
    long_row_chunks = [c for c in chunks if "word word" in c.text]
    assert {tuple(c.meta["row_range"]) for c in long_row_chunks} == {(3, 3)}
