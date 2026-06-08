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
