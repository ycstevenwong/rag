"""Format-specific parsers that produce a normalized stream of Blocks.

Each parser returns List[Block]. A Block is the smallest unit of text the
chunker will consider for grouping; it carries provenance metadata so we can
build citations later (page number, slide number, heading path, sheet name).
"""
from __future__ import annotations

import codecs
import re
import zipfile
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


def check_readable(path: Path) -> None:
    """Raise ValueError with a plain reason if the file can't be opened as its
    type. Cheap enough to run on every upload, before a suggestion is queued."""
    ext = path.suffix.lower()
    if ext == ".pdf":
        _open_pdf(path).close()
    elif ext == ".docx":
        _open_docx(path)


def _open_pdf(path: Path):
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(str(path), filetype="pdf")
    except (fitz.FileDataError, RuntimeError) as exc:
        raise ValueError(
            "This file isn't a valid PDF. Open it and save it as a PDF again, then re-upload."
        ) from exc
    if doc.needs_pass:
        doc.close()
        raise ValueError("This PDF is password-protected. Remove the password and re-upload.")
    return doc


def _open_docx(path: Path):
    from docx import Document
    from docx.opc.exceptions import PackageNotFoundError

    try:
        return Document(str(path))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError(
            "This file isn't a valid Word document. Open it in Word and save it as .docx, then re-upload."
        ) from exc


def _push_heading(stack: list[tuple[int, str]], level: int, text: str) -> None:
    """Close every open section at this level or deeper, then open this one.
    Compares levels rather than stack depth, since documents can skip levels."""
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, _collapse_spaces(text)))


def _collapse_spaces(text: str) -> str:
    """Tabs after heading numbers ("5.\\tScope") would otherwise show in citations."""
    return " ".join(text.split())


# Fonts whose glyphs are symbols, not words. Their sizes say nothing about headings:
# formulas use large brackets. Dingbat glyphs are bullets, and some PDFs name them
# ("Bullet") in the extracted text.
_SYMBOL_FONTS = ("symbol", "math", "wingding", "webding", "dingbat")
_DINGBAT_FONTS = ("wingding", "webding", "dingbat")
_LETTER_RE = re.compile(r"[^\W\d_]")
# Lines with this many words are sentences, so their size is the body size.
_PROSE_LINE_WORDS = 6


def _is_font(span: dict, kinds: tuple[str, ...]) -> bool:
    font = (span.get("font") or "").lower()
    return any(k in font for k in kinds)


def _span_text(span: dict) -> str:
    text = span.get("text", "")
    # Some text fonts name the glyph too, in a span of its own.
    if text.strip() == "Bullet" or (text.strip() and _is_font(span, _DINGBAT_FONTS)):
        return "•"
    return text


_BULLET_GAP_RE = re.compile(r"•(?=\S)")


def _heading_size(spans: list[dict]) -> float:
    """Largest size among spans in a text font, so a formula's big brackets in a
    symbol font don't count."""
    sizes = [
        s.get("size") or 0.0
        for s in spans
        if s.get("text", "").strip() and not _is_font(s, _SYMBOL_FONTS)
    ]
    return max(sizes, default=0.0)


# Headings are rarely this long and almost never end like a sentence.
_SENTENCE_WORDS = 12
_SENTENCE_END_RE = re.compile(r"[.!?][\"'”’)]*$")


def _looks_like_heading(text: str) -> bool:
    """Has a word (not just a figure like "Q2") and isn't a full sentence."""
    if len(_LETTER_RE.findall(text)) < 2:
        return False
    return not (len(text.split()) >= _SENTENCE_WORDS and _SENTENCE_END_RE.search(text))


def _heading_path(stack: list[tuple[int, str]]) -> str:
    return " > ".join(text for _, text in stack)


def parse_text(path: Path) -> list[Block]:
    text = _decode_text(path.read_bytes()).replace("\r\n", "\n").replace("\r", "\n")
    blocks: list[Block] = []
    for para in re.split(r"\n[ \t]*\n", text):
        para = para.strip()
        if para:
            blocks.append(Block(text=para, kind="paragraph", meta={}))
    return blocks


def _decode_text(data: bytes) -> str:
    """A byte order mark names the encoding. Without one, try UTF-8, then assume
    Windows-1252, the likeliest source of text that isn't valid UTF-8."""
    if data.startswith(codecs.BOM_UTF8):
        return data.decode("utf-8-sig", errors="replace")
    if data.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return data.decode("utf-32", errors="replace")
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def parse_pdf(path: Path) -> list[Block]:
    """Section-aware PDF parser.

    Walks each page's structured text (PyMuPDF's "dict" output) so we can read
    font sizes per block. Blocks whose max font size is meaningfully larger
    than the document's body font are treated as headings; we maintain a
    heading_stack so every emitted paragraph carries the section context it
    belongs to (same shape as parse_docx). The chunker then keeps each chunk
    within one section via heading_path boundary tracking.

    Falls back to legacy paragraph-mode parsing if the structural extraction
    can't determine a body font (very small / image-only / odd PDFs).
    """
    from . import config_facade as cfg

    max_font = float(cfg.PDF_MAX_FONT_SIZE or 0)  # 0 = disabled
    screen_fonts = [
        f.strip().lower() for f in (cfg.PDF_SCREEN_FONTS or "").split(",") if f.strip()
    ]
    field_desc_labels = [
        l.strip().lower() for l in (cfg.PDF_FIELD_DESC_LABELS or "").split(",") if l.strip()
    ]

    def _block_is_screen(sizes_spans_text: list[tuple[float, str]]) -> bool:
        """True if every span in the block matches a configured screen font."""
        if not screen_fonts or not sizes_spans_text:
            return False
        for _, font_name in sizes_spans_text:
            fn = (font_name or "").lower()
            if not any(sf in fn for sf in screen_fonts):
                return False
        return True

    def _matches_field_desc(text: str) -> bool:
        """True if the block text matches a configured Field Descriptions label."""
        if not field_desc_labels:
            return False
        lower = text.strip().lower()
        return any(label in lower for label in field_desc_labels)

    doc = _open_pdf(path)
    try:
        # First pass — plain text per page, used to identify repeating
        # headers/footers we should drop from the structured pass.
        pages_text: list[str] = []
        for page in doc:
            try:
                pages_text.append(page.get_text("text") or "")
            except Exception:
                pages_text.append("")

        top_repeats, bot_repeats = _find_repeating_lines(pages_text, min_frac=0.5)
        repeating = top_repeats | bot_repeats

        # Estimate the document's body font size. If we can't, fall back.
        body_size = _estimate_body_font_size(doc)
        if body_size is None:
            return _parse_pdf_legacy(pages_text)

        heading_threshold = body_size * 1.15  # >= 15% larger than body = heading
        heading_stack: list[tuple[int, str]] = []
        blocks: list[Block] = []
        # Buffer for consecutive same-page screen blocks so they merge into one.
        screen_buf: list[str] = []
        screen_buf_page: int | None = None
        # Track the most recent screen so Field Descriptions and their
        # subsequent content can point back to it. Reset by real (level ≤ 3)
        # headings.
        last_screen_page: int | None = None

        def _flush_screen() -> None:
            nonlocal screen_buf, screen_buf_page, last_screen_page
            if not screen_buf or screen_buf_page is None:
                return
            joined = "\n".join(screen_buf)
            blocks.append(Block(
                text=joined,
                kind="screen",
                meta={
                    "page": screen_buf_page,
                    "heading_path": _heading_path(heading_stack),
                },
            ))
            last_screen_page = screen_buf_page
            screen_buf = []
            screen_buf_page = None

        for page_num, page in enumerate(doc, start=1):
            if screen_buf_page is not None and screen_buf_page != page_num:
                _flush_screen()
            try:
                page_dict = page.get_text("dict")
            except Exception:
                continue
            for blk in page_dict.get("blocks", []):
                if blk.get("type", 0) != 0:  # 0 = text block
                    continue
                texts: list[str] = []
                line_heading_sizes: list[float] = []
                sizes: list[float] = []
                span_fonts: list[tuple[float, str]] = []
                for line in blk.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = _BULLET_GAP_RE.sub("• ", "".join(_span_text(s) for s in spans)).strip()
                    # Per line, because a block can join a repeated line with
                    # others (a two-line footer, say).
                    if _is_repeated_line(line_text, repeating):
                        continue
                    if line_text:
                        texts.append(line_text)
                        line_heading_sizes.append(_heading_size(spans))
                    for s in spans:
                        sz = s.get("size")
                        if sz:
                            sizes.append(sz)
                        span_fonts.append((sz or 0.0, s.get("font") or ""))
                text = " ".join(texts).strip()
                if not text:
                    continue
                max_size = max(sizes) if sizes else body_size
                if max_font and max_size >= max_font:
                    continue

                # Screen detection: all spans in a configured screen font.
                # Consecutive same-page screen blocks buffer into one Block.
                if _block_is_screen(span_fonts):
                    if screen_buf_page is not None and screen_buf_page != page_num:
                        _flush_screen()
                    screen_buf_page = page_num
                    # Line structure is what makes a screen readable, so unlike
                    # prose its lines stay separate.
                    screen_buf.append("\n".join(texts))
                    continue
                # Non-screen block — flush any pending screen first.
                if screen_buf:
                    _flush_screen()

                # Heading heuristic: the block's leading lines are visibly larger
                # than body text, and short. Body lines after them in the same
                # block (a title directly above its first paragraph) stay body.
                n_title = 0
                while n_title < len(texts) and line_heading_sizes[n_title] >= heading_threshold:
                    n_title += 1
                title = _collapse_spaces(" ".join(texts[:n_title]))
                is_heading = n_title > 0 and len(title) <= 200 and _looks_like_heading(title)
                rest = " ".join(texts[n_title:]) if is_heading else ""
                if is_heading:
                    text = title
                    heading_size = max(line_heading_sizes[:n_title])
                # OR the block text matches a configured Field Descriptions
                # label — treated as a level-4 sub-heading so its contents nest
                # under whatever section is above it.
                is_field_desc = _matches_field_desc(text) and len(text) <= 100
                if is_field_desc and not is_heading:
                    is_heading = True
                    forced_level = 4
                    text = _collapse_spaces(text)
                else:
                    forced_level = None

                if is_heading:
                    level = forced_level or _heading_level_from_size(heading_size, body_size)
                    # A major heading (level ≤ 3) resets the screen scope.
                    # Level-4 Field Descriptions inherits the current tag so
                    # it stays anchored to the screen it describes.
                    if level <= 3:
                        last_screen_page = None
                    _push_heading(heading_stack, level, text)
                    meta = {
                        "page": page_num,
                        "heading_path": _heading_path(heading_stack),
                        "level": level,
                    }
                    if last_screen_page is not None:
                        meta["related_screen_page"] = last_screen_page
                    blocks.append(Block(text=text, kind="heading", meta=meta))
                    if rest:
                        body_meta = {"page": page_num, "heading_path": meta["heading_path"]}
                        if last_screen_page is not None:
                            body_meta["related_screen_page"] = last_screen_page
                        blocks.append(Block(text=rest, kind="paragraph", meta=body_meta))
                else:
                    meta = {
                        "page": page_num,
                        "heading_path": _heading_path(heading_stack),
                    }
                    if last_screen_page is not None:
                        meta["related_screen_page"] = last_screen_page
                    blocks.append(Block(text=text, kind="paragraph", meta=meta))
        _flush_screen()  # emit any trailing screen block
        return blocks
    finally:
        doc.close()


def _parse_pdf_legacy(pages_text: list[str]) -> list[Block]:
    """Paragraph-mode fallback (the pre-section-aware behavior)."""
    pages_text = _strip_repeating_lines(pages_text, min_frac=0.5)
    blocks: list[Block] = []
    for page_num, text in enumerate(pages_text, start=1):
        for para in _split_paragraphs(text):
            blocks.append(Block(text=para, kind="paragraph", meta={"page": page_num}))
    return blocks


def _estimate_body_font_size(doc) -> float | None:
    """Most-common font size weighted by character count. Returns None if
    no usable text/font info is available."""
    sizes: Counter[float] = Counter()
    # Sentences only, when there are any: in a report full of tables, the small
    # figures in the cells can outweigh the text and set the body size too low.
    prose_sizes: Counter[float] = Counter()
    for page in doc:
        try:
            page_dict = page.get_text("dict")
        except Exception:
            continue
        for blk in page_dict.get("blocks", []):
            if blk.get("type", 0) != 0:
                continue
            for line in blk.get("lines", []):
                spans = line.get("spans", [])
                is_prose = len("".join(s.get("text", "") for s in spans).split()) >= _PROSE_LINE_WORDS
                for span in spans:
                    sz = span.get("size")
                    txt = (span.get("text") or "").strip()
                    if sz and txt:
                        # Round to nearest 0.5pt so near-identical sizes group.
                        key = round(sz * 2) / 2
                        sizes[key] += len(txt)
                        if is_prose:
                            prose_sizes[key] += len(txt)
    counts = prose_sizes or sizes
    if not counts:
        return None
    return counts.most_common(1)[0][0]


def _heading_level_from_size(size: float, body_size: float) -> int:
    """Map font-size ratio to a heading level (1 = biggest)."""
    ratio = size / body_size if body_size else 1.0
    if ratio >= 2.0:
        return 1
    if ratio >= 1.5:
        return 2
    return 3


def parse_docx(path: Path) -> list[Block]:
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    from . import config_facade as cfg

    doc = _open_docx(path)
    blocks: list[Block] = []
    heading_stack: list[tuple[int, str]] = []
    stop_keywords = [k.lower() for k in cfg.DOCX_STOP_HEADINGS if k.strip()]

    for child in _docx_block_elements(doc.element.body):
        tag = _xml_local_name(child)

        if tag == "p":
            para = Paragraph(child, doc._body)
            text = (para.text or "").strip()
            if not text:
                continue
            style = (para.style.name or "").lower() if para.style else ""
            # Some documents put body sentences in a heading style.
            is_heading = style.startswith("heading") and _looks_like_heading(text)

            if is_heading and stop_keywords:
                heading_lower = text.lower()
                if any(kw in heading_lower for kw in stop_keywords):
                    break  # drop the matching heading and everything after

            if is_heading:
                text = _collapse_spaces(text)
                try:
                    level = int(style.split()[-1])
                except ValueError:
                    level = 1
                _push_heading(heading_stack, level, text)
                blocks.append(Block(
                    text=text,
                    kind="heading",
                    meta={"heading_path": _heading_path(heading_stack), "level": level},
                ))
            else:
                blocks.append(Block(
                    text=text,
                    kind="paragraph",
                    meta={"heading_path": _heading_path(heading_stack)},
                ))

        elif tag == "tbl":
            for row_text in _docx_table_rows(Table(child, doc._body)):
                blocks.append(Block(
                    text=row_text,
                    kind="table_row",
                    meta={"heading_path": _heading_path(heading_stack)},
                ))

    return blocks


def _xml_local_name(element) -> str:
    return element.tag.split("}", 1)[-1]


def _docx_block_elements(parent):
    """Paragraphs and tables in document order, including those inside content
    controls. A generated table of contents is skipped: it only repeats headings."""
    for child in parent.iterchildren():
        if _xml_local_name(child) != "sdt":
            yield child
            continue
        gallery = [
            e.get(e.tag.split("}", 1)[0] + "}val", "")
            for e in child.iter()
            if _xml_local_name(e) == "docPartGallery"
        ]
        if any(value.startswith("Table of Contents") for value in gallery):
            continue
        for content in child:
            if _xml_local_name(content) == "sdtContent":
                yield from _docx_block_elements(content)


def _docx_table_rows(table):
    """One line per row, followed by the rows of any tables nested in its cells."""
    for row in table.rows:
        # row.cells repeats a merged cell once per column it spans.
        cells = []
        seen = set()
        for cell in row.cells:
            if id(cell._tc) not in seen:
                seen.add(id(cell._tc))
                cells.append(cell)
        row_text = " | ".join(text for text in (c.text.strip() for c in cells) if text)
        if row_text:
            yield row_text
        for cell in cells:
            for nested in cell.tables:
                yield from _docx_table_rows(nested)


def parse_pptx(path: Path) -> list[Block]:
    from pptx import Presentation

    prs = Presentation(str(path))
    blocks: list[Block] = []
    for slide_num, slide in enumerate(prs.slides, start=1):
        title_shape = slide.shapes.title
        title = title_shape.text_frame.text.strip() if title_shape is not None else ""
        body_parts = list(_pptx_shape_texts(slide.shapes, skip=title_shape))

        notes = ""
        if slide.has_notes_slide:
            # None when the notes slide has no body placeholder.
            notes_frame = slide.notes_slide.notes_text_frame
            notes = (notes_frame.text or "").strip() if notes_frame is not None else ""

        combined = "\n\n".join(part for part in [title, *body_parts, notes] if part)
        if combined:
            blocks.append(Block(
                text=combined,
                kind="slide",
                meta={"slide_num": slide_num, "slide_title": title},
            ))
    return blocks


def _pptx_shape_texts(shapes, skip=None):
    """Text of each shape in z-order, descending into groups and reading tables."""
    from pptx.shapes.group import GroupShape

    for shape in shapes:
        if skip is not None and shape == skip:
            continue
        if isinstance(shape, GroupShape):
            yield from _pptx_shape_texts(shape.shapes)
        elif shape.has_table:
            rows = (" | ".join(cell.text.strip() for cell in row.cells) for row in shape.table.rows)
            text = "\n".join(row for row in rows if row.strip(" |"))
            if text:
                yield text
        elif shape.has_text_frame:
            text = shape.text_frame.text.strip()
            if text:
                yield text


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
        header_str = _sheet_row_text(header)
        rows: list[tuple[int, str]] = []  # (row number, text) with data only

        def flush() -> None:
            blocks.append(Block(
                text=f"Headers: {header_str}\n" + "\n".join(text for _, text in rows),
                kind="sheet_row",
                meta={
                    "sheet": sheet.title,
                    "row_start": rows[0][0],
                    "row_end": rows[-1][0],
                    "rows": [number for number, _ in rows],
                },
            ))
            rows.clear()

        n_data_rows = 0
        for row_idx, row in enumerate(rows_iter, start=2):
            if _row_has_data(row):
                rows.append((row_idx, _sheet_row_text(row)))
                n_data_rows += 1
            if len(rows) >= 20:
                flush()
        if rows:
            flush()
        # With nothing below it, the first row is the content, not a header.
        if n_data_rows == 0 and _row_has_data(header):
            blocks.append(Block(
                text=header_str,
                kind="sheet_row",
                meta={"sheet": sheet.title, "row_start": 1, "row_end": 1, "rows": [1]},
            ))
    wb.close()
    return blocks


def _row_has_data(row: tuple) -> bool:
    return any(c is not None and str(c).strip() for c in row)


def _sheet_row_text(row: tuple) -> str:
    """One line per row: a newline inside a cell would break the row numbering."""
    return " | ".join(
        re.sub(r"[\r\n]+", " ", str(c)) if c is not None else "" for c in row
    )


def _split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n\n")]
    parts = [p for p in parts if p]
    if not parts:
        parts = [line.strip() for line in text.splitlines() if line.strip()]
    return parts


def _mask_digits(line: str) -> str:
    return re.sub(r"\d+", "#", line)


# After _mask_digits: "#", "Page #", "Acme Corp Page #", "# of #", "Page # / #".
_PAGE_NUMBER_LINE_RE = re.compile(r"(?:^|\bpage\s+)#(?:\s*(?:of|/)\s*#)?$", re.IGNORECASE)


def _find_repeating_lines(
    pages_text: list[str], min_frac: float = 0.5
) -> tuple[set[str], set[str]]:
    """Identify header/footer lines that recur on most pages.

    A line qualifies if it repeats exactly, or if its digits vary and it reads as
    a page number. Other lines with varying digits ("Chapter 3", "Step 2 ...") are
    content, even when they sit at the top of most pages.

    Returns (top_repeats, bot_repeats) for use with _is_repeated_line. Needs at
    least 3 pages to engage; returns (set(), set()) otherwise.
    """
    if len(pages_text) < 3:
        return set(), set()

    page_lines = [p.splitlines() for p in pages_text]
    top_counts: Counter[str] = Counter()
    bot_counts: Counter[str] = Counter()
    for lines in page_lines:
        non_empty = [l.strip() for l in lines if l.strip()]
        for counts, edge in ((top_counts, non_empty[:2]), (bot_counts, non_empty[-2:])):
            for line in edge:
                counts[line] += 1
                digits_masked = _mask_digits(line)
                if digits_masked != line and _PAGE_NUMBER_LINE_RE.search(digits_masked):
                    counts[digits_masked] += 1

    threshold = max(2, int(min_frac * len(pages_text)))
    top_repeats = {k for k, v in top_counts.items() if v >= threshold}
    bot_repeats = {k for k, v in bot_counts.items() if v >= threshold}
    return top_repeats, bot_repeats


def _is_repeated_line(line: str, repeating: set[str]) -> bool:
    line = line.strip()
    return line in repeating or _mask_digits(line) in repeating


def _strip_repeating_lines(pages_text: list[str], min_frac: float = 0.5) -> list[str]:
    """Remove running headers/footers that repeat at the top or bottom of pages."""
    top_repeats, bot_repeats = _find_repeating_lines(pages_text, min_frac)
    if not top_repeats and not bot_repeats:
        return pages_text

    page_lines = [p.splitlines() for p in pages_text]
    cleaned: list[str] = []
    for lines in page_lines:
        i, j = 0, len(lines)
        while i < j and (not lines[i].strip() or _is_repeated_line(lines[i], top_repeats)):
            i += 1
        while j > i and (not lines[j - 1].strip() or _is_repeated_line(lines[j - 1], bot_repeats)):
            j -= 1
        cleaned.append("\n".join(lines[i:j]))
    return cleaned
