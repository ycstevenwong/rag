"""Inspect a PDF and report what parse_pdf would extract.

Prints the body font size estimate, font-size distribution, detected
repeating headers/footers, the heading list, and whether the parser would
fall back to legacy paragraph mode. Useful for tuning section-aware
chunking or debugging why a particular doc parses weirdly.

Usage:
    python scripts/inspect_pdf.py path/to/file.pdf
    python scripts/inspect_pdf.py path/to/file.pdf --max-headings 50
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

# Force UTF-8 stdout so glyphs don't crash on Windows cp1252 terminals.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag.parsers import (  # noqa: E402
    _estimate_body_font_size,
    _find_repeating_lines,
    _heading_level_from_size,
)


def inspect(pdf_path: Path, max_headings: int) -> int:
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        print(f"Could not open {pdf_path}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    try:
        n_pages = doc.page_count
        print(f"=== PDF Inspector: {pdf_path.name} ===")
        print(f"Pages: {n_pages}")

        # First pass: plain text per page (drives repeating-line detection).
        pages_text: list[str] = []
        for page in doc:
            try:
                pages_text.append(page.get_text("text") or "")
            except Exception:
                pages_text.append("")

        top_repeats, bot_repeats = _find_repeating_lines(pages_text, min_frac=0.5)

        # Body font size estimate (same code path the real parser uses).
        body_size = _estimate_body_font_size(doc)

        print()
        if body_size is None:
            print("Body font size: could not be determined.")
            print("Parser mode:    LEGACY (fallback) — would split by paragraph only,")
            print("                no heading detection, no section boundaries.")
            print()
            print("Common causes: image-only PDF (scanned), encrypted, or unusual encoding.")
            return 0

        threshold = body_size * 1.15
        print(f"Body font size:    {body_size:.1f} pt  (most common, character-weighted)")
        print(f"Heading threshold: ≥ {threshold:.2f} pt  AND  ≤ 200 chars")
        print("Parser mode:       structural (section-aware)")
        print()

        # Font-size distribution.
        size_dist: Counter[float] = Counter()
        for page in doc:
            try:
                page_dict = page.get_text("dict")
            except Exception:
                continue
            for blk in page_dict.get("blocks", []):
                if blk.get("type", 0) != 0:
                    continue
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        sz = span.get("size")
                        txt = (span.get("text") or "").strip()
                        if sz and txt:
                            size_dist[round(sz * 2) / 2] += len(txt)

        print("Font size distribution (top 8, by character count):")
        for size, count in size_dist.most_common(8):
            tags = []
            if size == body_size:
                tags.append("body")
            if size >= threshold:
                tags.append("heading-eligible")
            tag_str = f"  ({', '.join(tags)})" if tags else ""
            print(f"  {size:5.1f} pt : {count:>9,} chars{tag_str}")
        print()

        # Repeating headers/footers.
        if top_repeats or bot_repeats:
            print("Repeating headers/footers (will be stripped):")
            for r in sorted(top_repeats):
                print(f"  top:    {r!r}")
            for r in sorted(bot_repeats):
                print(f"  bottom: {r!r}")
        else:
            print("Repeating headers/footers: none detected.")
        print()

        # Walk again and identify headings the real parser would emit.
        headings: list[tuple[int, int, float, str]] = []
        n_blocks = 0
        n_paragraphs = 0
        repeating = top_repeats | bot_repeats

        def is_repeating(text: str) -> bool:
            if not repeating:
                return False
            import re
            return re.sub(r"\d+", "#", text.strip()) in repeating

        for page_num, page in enumerate(doc, start=1):
            try:
                page_dict = page.get_text("dict")
            except Exception:
                continue
            for blk in page_dict.get("blocks", []):
                if blk.get("type", 0) != 0:
                    continue
                texts: list[str] = []
                sizes: list[float] = []
                for line in blk.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(s.get("text", "") for s in spans).strip()
                    if line_text:
                        texts.append(line_text)
                    for s in spans:
                        sz = s.get("size")
                        if sz:
                            sizes.append(sz)
                text = " ".join(texts).strip()
                if not text or is_repeating(text):
                    continue
                n_blocks += 1
                max_size = max(sizes) if sizes else body_size
                if max_size >= threshold and len(text) <= 200:
                    level = _heading_level_from_size(max_size, body_size)
                    headings.append((page_num, level, max_size, text))
                else:
                    n_paragraphs += 1

        # Summary.
        print(f"Block totals (after stripping repeating lines):")
        print(f"  total blocks:      {n_blocks}")
        print(f"  headings:          {len(headings)}")
        print(f"  paragraphs:        {n_paragraphs}")
        print()

        if headings:
            level_counts = Counter(h[1] for h in headings)
            level_summary = ", ".join(f"H{l}={n}" for l, n in sorted(level_counts.items()))
            print(f"Heading levels: {level_summary}")
            print()
            shown = min(len(headings), max_headings)
            print(f"First {shown} headings:")
            for page_num, level, size, text in headings[:shown]:
                snippet = text if len(text) <= 70 else text[:67] + "..."
                print(f"  H{level} [page {page_num:>3}] {size:5.1f}pt  {snippet}")
            if len(headings) > shown:
                print(f"  ... {len(headings) - shown} more")
        else:
            print("No headings detected. Section-aware chunking won't split this PDF.")
            print("Likely causes:")
            print("  - all body text uses the same font size (no visual heading hierarchy)")
            print("  - headings are >200 chars (rare — paragraphs masquerading as headings)")
            print("  - PDF was generated without preserving font metadata")

        return 0
    finally:
        doc.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a PDF's section-aware parse output.")
    parser.add_argument("path", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--max-headings", type=int, default=30,
        help="Maximum number of headings to print (default: 30)",
    )
    args = parser.parse_args()

    pdf_path: Path = args.path.resolve()
    if not pdf_path.exists():
        print(f"Not found: {pdf_path}", file=sys.stderr)
        return 1
    if pdf_path.suffix.lower() != ".pdf":
        print(f"Not a PDF: {pdf_path}", file=sys.stderr)
        return 1

    return inspect(pdf_path, max_headings=args.max_headings)


if __name__ == "__main__":
    sys.exit(main())
