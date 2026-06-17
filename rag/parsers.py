"""Format-specific parsers that produce a normalized stream of Blocks.

Each parser returns List[Block]. A Block is the smallest unit of text the
chunker will consider for grouping; it carries provenance metadata so we can
build citations later (page number, slide number, heading path, sheet name).
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Block:
    text: str
    kind: str  # "paragraph" | "heading" | "table_row" | "slide" | "sheet_row"
    meta: dict[str, Any] = field(default_factory=dict)


def parse(path: Path) -> list[Block]:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext == ".pptx":
        return parse_pptx(path)
    if ext == ".xlsx":
        return parse_xlsx(path)
    if ext in (".txt", ".md"):
        return parse_text(path)
    raise ValueError(f"Unsupported file type: {ext}")


def parse_text(path: Path) -> list[Block]:
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks: list[Block] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            blocks.append(Block(text=para, kind="paragraph", meta={}))
    return blocks


def parse_pdf(path: Path) -> list[Block]:
    import fitz  # PyMuPDF

    pages_text: list[str] = []
    doc = fitz.open(str(path))
    try:
        for page in doc:
            try:
                pages_text.append(page.get_text("text") or "")
            except Exception:
                pages_text.append("")
    finally:
        doc.close()

    pages_text = _strip_repeating_lines(pages_text, min_frac=0.5)

    blocks: list[Block] = []
    for page_num, text in enumerate(pages_text, start=1):
        for para in _split_paragraphs(text):
            blocks.append(Block(text=para, kind="paragraph", meta={"page": page_num}))
    return blocks


def parse_docx(path: Path) -> list[Block]:
    from docx import Document

    doc = Document(str(path))
    blocks: list[Block] = []
    heading_stack: list[str] = []  # [h1, h2, h3, ...]

    for para in doc.paragraphs:
        text = (para.text or "").strip()
        if not text:
            continue
        style = (para.style.name or "").lower() if para.style else ""
        if style.startswith("heading"):
            try:
                level = int(style.split()[-1])
            except ValueError:
                level = 1
            heading_stack = heading_stack[: max(0, level - 1)]
            heading_stack.append(text)
            blocks.append(Block(
                text=text,
                kind="heading",
                meta={"heading_path": " > ".join(heading_stack), "level": level},
            ))
        else:
            blocks.append(Block(
                text=text,
                kind="paragraph",
                meta={"heading_path": " > ".join(heading_stack) if heading_stack else ""},
            ))

    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            row_text = " | ".join(c for c in cells if c)
            if row_text:
                blocks.append(Block(
                    text=row_text,
                    kind="table_row",
                    meta={"heading_path": " > ".join(heading_stack) if heading_stack else ""},
                ))
    return blocks


def parse_pptx(path: Path) -> list[Block]:
    from pptx import Presentation

    prs = Presentation(str(path))
    blocks: list[Block] = []
    for slide_num, slide in enumerate(prs.slides, start=1):
        title = ""
        body_parts: list[str] = []
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            text = shape.text_frame.text.strip()
            if not text:
                continue
            if shape == slide.shapes.title or (
                shape.has_text_frame and shape.text_frame.text and not title
            ):
                if not title:
                    title = text
                    continue
            body_parts.append(text)

        notes = ""
        if slide.has_notes_slide:
            notes = (slide.notes_slide.notes_text_frame.text or "").strip()

        combined = "\n\n".join(part for part in [title, *body_parts, notes] if part)
        if combined:
            blocks.append(Block(
                text=combined,
                kind="slide",
                meta={"slide_num": slide_num, "slide_title": title},
            ))
    return blocks


def parse_xlsx(path: Path) -> list[Block]:
    from openpyxl import load_workbook

    wb = load_workbook(str(path), data_only=True, read_only=True)
    blocks: list[Block] = []
    for sheet in wb.worksheets:
        rows_iter = sheet.iter_rows(values_only=True)
        try:
            header = next(rows_iter)
        except StopIteration:
            continue
        header_str = " | ".join(str(c) if c is not None else "" for c in header)
        buffer: list[str] = []
        first_row_in_buffer = 2
        row_idx = 2
        for row in rows_iter:
            cells = " | ".join(str(c) if c is not None else "" for c in row)
            if cells.strip():
                buffer.append(cells)
            if len(buffer) >= 20:
                blocks.append(Block(
                    text=f"Headers: {header_str}\n" + "\n".join(buffer),
                    kind="sheet_row",
                    meta={
                        "sheet": sheet.title,
                        "row_start": first_row_in_buffer,
                        "row_end": row_idx,
                    },
                ))
                buffer = []
                first_row_in_buffer = row_idx + 1
            row_idx += 1
        if buffer:
            blocks.append(Block(
                text=f"Headers: {header_str}\n" + "\n".join(buffer),
                kind="sheet_row",
                meta={
                    "sheet": sheet.title,
                    "row_start": first_row_in_buffer,
                    "row_end": row_idx - 1,
                },
            ))
    wb.close()
    return blocks


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    parts = [p for p in parts if p]
    if not parts:
        parts = [line.strip() for line in text.splitlines() if line.strip()]
    return parts


def _strip_repeating_lines(pages_text: list[str], min_frac: float = 0.5) -> list[str]:
    """Remove running headers/footers that repeat at the top or bottom of pages.

    A line is treated as a header/footer if a digit-normalized version of it
    appears among the first or last non-empty lines of at least `min_frac` of
    the pages. Needs at least 3 pages to engage.
    """
    if len(pages_text) < 3:
        return pages_text

    def norm(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip())

    page_lines = [p.splitlines() for p in pages_text]
    top_counts: Counter[str] = Counter()
    bot_counts: Counter[str] = Counter()
    for lines in page_lines:
        non_empty = [l for l in lines if l.strip()]
        for l in non_empty[:2]:
            top_counts[norm(l)] += 1
        for l in non_empty[-2:]:
            bot_counts[norm(l)] += 1

    threshold = max(2, int(min_frac * len(pages_text)))
    top_repeats = {k for k, v in top_counts.items() if v >= threshold}
    bot_repeats = {k for k, v in bot_counts.items() if v >= threshold}
    if not top_repeats and not bot_repeats:
        return pages_text

    cleaned: list[str] = []
    for lines in page_lines:
        i, j = 0, len(lines)
        while i < j and (not lines[i].strip() or norm(lines[i]) in top_repeats):
            i += 1
        while j > i and (not lines[j - 1].strip() or norm(lines[j - 1]) in bot_repeats):
            j -= 1
        cleaned.append("\n".join(lines[i:j]))
    return cleaned
