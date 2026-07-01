"""Inspect a PDF and report what parse_pdf would extract.

Prints the body font size estimate, font-size distribution, detected
repeating headers/footers, the heading list, and whether the parser would
fall back to legacy paragraph mode. Useful for tuning section-aware
chunking or debugging why a particular doc parses weirdly.

Usage:
    python scripts/inspect_pdf.py path/to/file.pdf
    python scripts/inspect_pdf.py path/to/file.pdf --max-headings 50
    python scripts/inspect_pdf.py path/to/file.pdf --full
    python scripts/inspect_pdf.py path/to/file.pdf --full -o report.txt
    python scripts/inspect_pdf.py path/to/file.pdf --find "Section 1"
    python scripts/inspect_pdf.py path/to/file.pdf --find "REVERSE-TRANS" --full
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


def inspect(pdf_path: Path, max_headings: int, full: bool = False, find: str | None = None) -> int:
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

        # Substring search — runs early so a user hunting for one specific
        # phrase sees the matches before the rest of the report scrolls by.
        if find:
            print()
            print(f"Search results for {find!r} (case-insensitive substring):")
            find_lower = find.lower()
            matches: list[tuple[int, float, str, int, str]] = []
            for page_num, page in enumerate(doc, start=1):
                try:
                    page_dict = page.get_text("dict")
                except Exception:
                    continue
                for blk in page_dict.get("blocks", []):
                    if blk.get("type", 0) != 0:
                        continue
                    for line in blk.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text") or ""
                            if find_lower in text.lower():
                                matches.append((
                                    page_num,
                                    float(span.get("size") or 0),
                                    span.get("font") or "(unknown)",
                                    int(span.get("flags") or 0),
                                    text,
                                ))
            if not matches:
                print("  No matches found.")
            else:
                cap = len(matches) if full else min(len(matches), 30)
                print(f"  Found {len(matches)} match{'es' if len(matches) != 1 else ''}"
                      f"{'; showing first ' + str(cap) if cap < len(matches) else ''}:")
                print()
                for i, (page_num, sz, font_name, flags, text) in enumerate(matches[:cap], start=1):
                    flag_tags = []
                    if flags & 16: flag_tags.append("bold")
                    if flags & 8:  flag_tags.append("mono")
                    if flags & 2:  flag_tags.append("italic")
                    flag_str = f"  [{', '.join(flag_tags)}]" if flag_tags else ""
                    snippet = text if len(text) <= 140 else text[:137] + "..."
                    print(f"  {i:>3}. page {page_num:>3}, {sz:5.1f} pt, {font_name}{flag_str}")
                    print(f"       text: {snippet!r}")
                if cap < len(matches):
                    print(f"  ... {len(matches) - cap} more (use --full to show all)")
            print()

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

        # Font-size AND font-family / flag distribution.
        size_dist: Counter[float] = Counter()
        font_dist: Counter[str] = Counter()
        font_props: dict[str, dict[str, bool]] = {}
        font_size_dist: dict[str, Counter[float]] = {}
        # One sample per page per font, up to a cap. Each sample = (page, size, text).
        font_samples: dict[str, list[tuple[int, float, str]]] = {}
        font_pages: dict[str, set[int]] = {}
        # Bump this to change how many per-page samples we retain.
        MAX_SAMPLES_PER_FONT = 20
        total_chars = 0
        bold_chars = 0
        mono_chars = 0
        italic_chars = 0
        for page_num, page in enumerate(doc, start=1):
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
                        if not (sz and txt):
                            continue
                        n_chars = len(txt)
                        rounded_sz = round(sz * 2) / 2
                        size_dist[rounded_sz] += n_chars
                        font_name = span.get("font") or "(unknown)"
                        flags = int(span.get("flags") or 0)
                        font_dist[font_name] += n_chars
                        if font_name not in font_props:
                            font_props[font_name] = {
                                "bold": bool(flags & 16),
                                "italic": bool(flags & 2),
                                "serif": bool(flags & 4),
                                "monospaced": bool(flags & 8),
                            }
                            font_samples[font_name] = []
                            font_pages[font_name] = set()
                            font_size_dist[font_name] = Counter()
                        font_size_dist[font_name][rounded_sz] += n_chars
                        font_pages[font_name].add(page_num)
                        # Sample capture: first sighting per page, up to cap.
                        if (
                            len(font_samples[font_name]) < MAX_SAMPLES_PER_FONT
                            and page_num not in {s[0] for s in font_samples[font_name]}
                        ):
                            snippet = txt if len(txt) <= 140 else txt[:137] + "..."
                            font_samples[font_name].append((page_num, sz, snippet))
                        if flags & 16:
                            bold_chars += n_chars
                        if flags & 8:
                            mono_chars += n_chars
                        if flags & 2:
                            italic_chars += n_chars
                        total_chars += n_chars

        # Font-size distribution.
        dist_n = len(size_dist) if full else 8
        dist_title = (
            f"Font size distribution (all {len(size_dist)}, by character count):"
            if full else
            f"Font size distribution (top {min(8, len(size_dist))}, by character count):"
        )
        print(dist_title)
        for size, count in size_dist.most_common(dist_n):
            tags = []
            if size == body_size:
                tags.append("body")
            if size >= threshold:
                tags.append("heading-eligible")
            tag_str = f"  ({', '.join(tags)})" if tags else ""
            print(f"  {size:5.1f} pt : {count:>9,} chars{tag_str}")
        print()

        # Font-family analysis: per-family sizes, role classification, sample.
        def _classify(name: str, sizes: Counter, props: dict) -> str:
            if props.get("monospaced"):
                return "code / monospaced"
            total = sum(sizes.values())
            if total == 0:
                return "unknown"
            primary = sizes.most_common(1)[0][0]
            body_share = sum(c for s, c in sizes.items() if s == body_size) / total
            heading_share = sum(c for s, c in sizes.items() if s >= threshold) / total
            if body_share >= 0.7:
                return "body font"
            if heading_share >= 0.7:
                return "heading font"
            if props.get("bold") and primary >= body_size * 1.05:
                return "heading font (bold-emphasized)"
            if primary < body_size:
                return "small text (footnotes / captions?)"
            return "mixed / other"

        def _format_pages(pages: set[int], max_ranges: int = 6) -> str:
            if not pages:
                return "(none)"
            sorted_pages = sorted(pages)
            ranges: list[str] = []
            start = prev = sorted_pages[0]
            for p in sorted_pages[1:]:
                if p == prev + 1:
                    prev = p
                    continue
                ranges.append(str(start) if start == prev else f"{start}-{prev}")
                start = prev = p
            ranges.append(str(start) if start == prev else f"{start}-{prev}")
            if len(ranges) <= max_ranges:
                return ", ".join(ranges) + f"  ({len(pages)} page{'s' if len(pages) != 1 else ''})"
            shown = ", ".join(ranges[:max_ranges])
            return f"{shown}, ...  ({len(pages)} pages total)"

        family_n = len(font_dist) if full else 8
        # Default sample count: 3 per font; --full shows all we collected (up to 20).
        sample_n = MAX_SAMPLES_PER_FONT if full else 3
        family_title = (
            f"Font family analysis (all {len(font_dist)}):"
            if full else
            f"Font family analysis (top {min(8, len(font_dist))} by character count):"
        )
        print(family_title)
        for name, count in font_dist.most_common(family_n):
            props = font_props.get(name, {})
            sizes = font_size_dist.get(name, Counter())
            primary_size = sizes.most_common(1)[0][0] if sizes else 0.0
            all_sizes_sorted = sorted(sizes.keys())
            min_sz = all_sizes_sorted[0] if all_sizes_sorted else 0.0
            max_sz = all_sizes_sorted[-1] if all_sizes_sorted else 0.0
            size_range = (
                f"{min_sz:.1f}–{max_sz:.1f} pt"
                if min_sz != max_sz
                else f"{primary_size:.1f} pt"
            )
            flags = [k for k in ("bold", "italic", "monospaced", "serif") if props.get(k)]
            flag_str = ", ".join(flags) if flags else "none"
            role = _classify(name, sizes, props)
            pages_str = _format_pages(font_pages.get(name, set()))
            samples = font_samples.get(name, [])[:sample_n]
            total_samples = len(font_samples.get(name, []))

            print(f"  {name}")
            print(f"    chars:   {count:>9,}")
            print(f"    sizes:   {size_range}  (primary {primary_size:.1f} pt)")
            print(f"    flags:   {flag_str}")
            print(f"    role:    {role}")
            print(f"    pages:   {pages_str}")
            print(f"    samples ({len(samples)} of {total_samples} collected):")
            for page, sz, text in samples:
                print(f"      [page {page:>3}, {sz:5.1f} pt] {text!r}")
            print()
        if not full and len(font_dist) > family_n:
            print(f"  ... {len(font_dist) - family_n} more (use --full to print all)")
            print()

        # Aggregate style-flag usage.
        if total_chars > 0:
            print("Style-flag usage (as fraction of total text):")
            for label, chars in (("bold", bold_chars), ("monospaced", mono_chars), ("italic", italic_chars)):
                pct = chars / total_chars * 100
                print(f"  {label:<12}: {chars:>9,} chars  ({pct:5.1f}%)")
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
        per_page: dict[int, dict[str, int]] = {}
        repeating = top_repeats | bot_repeats

        def is_repeating(text: str) -> bool:
            if not repeating:
                return False
            import re
            return re.sub(r"\d+", "#", text.strip()) in repeating

        for page_num, page in enumerate(doc, start=1):
            per_page.setdefault(page_num, {"blocks": 0, "headings": 0, "paragraphs": 0})
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
                per_page[page_num]["blocks"] += 1
                max_size = max(sizes) if sizes else body_size
                if max_size >= threshold and len(text) <= 200:
                    level = _heading_level_from_size(max_size, body_size)
                    headings.append((page_num, level, max_size, text))
                    per_page[page_num]["headings"] += 1
                else:
                    n_paragraphs += 1
                    per_page[page_num]["paragraphs"] += 1

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
            heading_cap = len(headings) if full else min(len(headings), max_headings)
            label = "All headings:" if full else f"First {heading_cap} headings:"
            print(label)
            for page_num, level, size, text in headings[:heading_cap]:
                snippet = text if len(text) <= 70 else text[:67] + "..."
                print(f"  H{level} [page {page_num:>3}] {size:5.1f}pt  {snippet}")
            if not full and len(headings) > heading_cap:
                print(f"  ... {len(headings) - heading_cap} more (use --full to print everything)")
            print()

            if full:
                print(f"Per-page block counts ({n_pages} pages):")
                for page_num in sorted(per_page):
                    stats = per_page[page_num]
                    print(
                        f"  page {page_num:>3}: blocks={stats['blocks']:>4}  "
                        f"headings={stats['headings']:>3}  paragraphs={stats['paragraphs']:>4}"
                    )
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
        help="Maximum number of headings to print in default mode (default: 30). Ignored with --full.",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Print everything: all font sizes, all headings, per-page block counts.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Write the report to a UTF-8 text file instead of stdout.",
    )
    parser.add_argument(
        "-f", "--find", type=str, default=None,
        help="Search for a substring in the PDF (case-insensitive) and list every "
             "matching span's page, font, size, and flags. Combine with --full to show "
             "all matches instead of the first 30.",
    )
    args = parser.parse_args()

    pdf_path: Path = args.path.resolve()
    if not pdf_path.exists():
        print(f"Not found: {pdf_path}", file=sys.stderr)
        return 1
    if pdf_path.suffix.lower() != ".pdf":
        print(f"Not a PDF: {pdf_path}", file=sys.stderr)
        return 1

    if args.output:
        out_path = args.output.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        original_stdout = sys.stdout
        try:
            with out_path.open("w", encoding="utf-8", errors="replace") as fp:
                sys.stdout = fp
                rc = inspect(pdf_path, max_headings=args.max_headings, full=args.full, find=args.find)
        finally:
            sys.stdout = original_stdout
        print(f"Report written to {out_path}")
        return rc

    return inspect(pdf_path, max_headings=args.max_headings, full=args.full, find=args.find)


if __name__ == "__main__":
    sys.exit(main())
