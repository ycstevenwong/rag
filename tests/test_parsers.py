from pathlib import Path

from rag.parsers import parse_text


def test_parse_text_paragraphs(tmp_path: Path):
    p = tmp_path / "demo.md"
    p.write_text("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.\n")
    blocks = parse_text(p)
    assert len(blocks) == 3
    assert blocks[0].text == "First paragraph."
    assert blocks[2].text == "Third paragraph."
    assert all(b.kind == "paragraph" for b in blocks)


def _write_bytes(tmp_path: Path, data: bytes) -> Path:
    p = tmp_path / "demo.txt"
    p.write_bytes(data)
    return p


def test_parse_text_reads_utf16_with_a_bom(tmp_path: Path):
    p = _write_bytes(tmp_path, "Refund policy.\n\nSecond.".encode("utf-16"))

    assert [b.text for b in parse_text(p)] == ["Refund policy.", "Second."]


def test_parse_text_drops_a_utf8_bom(tmp_path: Path):
    p = _write_bytes(tmp_path, "﻿First paragraph.".encode("utf-8"))

    assert [b.text for b in parse_text(p)] == ["First paragraph."]


def test_parse_text_falls_back_to_windows_1252(tmp_path: Path):
    p = _write_bytes(tmp_path, "Café crème — “quoted”".encode("cp1252"))

    assert [b.text for b in parse_text(p)] == ["Café crème — “quoted”"]


def test_parse_text_keeps_utf8(tmp_path: Path):
    p = _write_bytes(tmp_path, "退款 Ångström".encode("utf-8"))

    assert [b.text for b in parse_text(p)] == ["退款 Ångström"]


def test_parse_text_splits_on_lines_holding_only_whitespace(tmp_path: Path):
    p = _write_bytes(tmp_path, b"one\r\n\r\ntwo\r\n \t\r\nthree")

    assert [b.text for b in parse_text(p)] == ["one", "two", "three"]
