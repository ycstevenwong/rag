"""Token-aware chunker that respects block boundaries and tracks provenance.

The chunker groups Blocks (from rag.parsers) into Chunks of ~CHUNK_TOKENS
tokens with CHUNK_OVERLAP overlap. It tries to:
  * keep a chunk's blocks within a single page (PDF) or slide (PPTX);
  * carry the current heading path (DOCX) into the chunk metadata so that
    retrieval matches against it;
  * never split inside a single block — if a block is too big, it becomes its
    own (possibly oversized) chunk, optionally sub-split on sentence boundaries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .parsers import Block


def count_tokens(text: str) -> int:
    # Rough heuristic: ~4 chars per token for English text. Good enough for
    # chunk-size control; we don't need byte-perfect counts here.
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    text: str
    token_count: int
    meta: dict[str, Any] = field(default_factory=dict)


def chunk_blocks(
    blocks: Iterable[Block],
    *,
    target_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    blocks = list(blocks)
    chunks: list[Chunk] = []
    buffer: list[Block] = []
    buffer_tokens = 0
    current_page: Any = None
    current_slide: Any = None
    current_heading_path: Any = None

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append(_blocks_to_chunk(buffer))
            buffer = []
            buffer_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block.text)
        block_page = block.meta.get("page")
        block_slide = block.meta.get("slide_num")
        block_heading_path = block.meta.get("heading_path")

        # Respect page / slide / section boundaries. A change in heading_path
        # means we've moved to a new section — flush before continuing so each
        # chunk stays within one section.
        crosses_page = (
            block_page is not None
            and current_page is not None
            and block_page != current_page
        )
        crosses_slide = (
            block_slide is not None
            and current_slide is not None
            and block_slide != current_slide
        )
        crosses_section = (
            block_heading_path is not None
            and current_heading_path is not None
            and block_heading_path != current_heading_path
        )
        if crosses_page or crosses_slide or crosses_section:
            flush()

        if buffer_tokens + block_tokens > target_tokens and buffer:
            flush()

        if block_tokens > target_tokens:
            if block.kind == "screen":
                # Screen mockups (e.g., CICS terminal captures) must stay
                # intact — splitting mid-screen destroys column alignment
                # and semantic meaning.
                chunks.append(_blocks_to_chunk([block]))
            else:
                for piece in _split_long_text(block.text, target_tokens):
                    chunks.append(_blocks_to_chunk([
                        Block(text=piece, kind=block.kind, meta=block.meta)
                    ]))
            current_page = block_page if block_page is not None else current_page
            current_slide = block_slide if block_slide is not None else current_slide
            if block_heading_path is not None:
                current_heading_path = block_heading_path
            continue

        buffer.append(block)
        buffer_tokens += block_tokens
        if block_page is not None:
            current_page = block_page
        if block_slide is not None:
            current_slide = block_slide
        if block_heading_path is not None:
            current_heading_path = block_heading_path

    flush()

    if overlap_tokens > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap_tokens)

    return chunks


def _blocks_to_chunk(buf: list[Block]) -> Chunk:
    text = "\n\n".join(b.text for b in buf).strip()
    meta: dict[str, Any] = {}
    pages = sorted({b.meta["page"] for b in buf if "page" in b.meta})
    if pages:
        meta["pages"] = pages
    slide_nums = sorted({b.meta["slide_num"] for b in buf if "slide_num" in b.meta})
    if slide_nums:
        meta["slide_nums"] = slide_nums
        titles = [b.meta.get("slide_title") for b in buf if b.meta.get("slide_title")]
        if titles:
            meta["slide_title"] = titles[0]
    headings = [b.meta.get("heading_path") for b in buf if b.meta.get("heading_path")]
    if headings:
        meta["heading_path"] = headings[-1]
    if any(b.kind == "screen" for b in buf):
        meta["contains_screen"] = True
    sheets = sorted({b.meta["sheet"] for b in buf if "sheet" in b.meta})
    if sheets:
        meta["sheets"] = sheets
        row_starts = [b.meta["row_start"] for b in buf if "row_start" in b.meta]
        row_ends = [b.meta["row_end"] for b in buf if "row_end" in b.meta]
        if row_starts and row_ends:
            meta["row_range"] = [min(row_starts), max(row_ends)]
    return Chunk(text=text, token_count=count_tokens(text), meta=meta)


def _split_long_text(text: str, target_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for sent in sentences:
        st = count_tokens(sent)
        if current and current_tokens + st > target_tokens:
            pieces.append(" ".join(current).strip())
            current = []
            current_tokens = 0
        if st > target_tokens:
            char_window = target_tokens * 4
            for i in range(0, len(sent), char_window):
                pieces.append(sent[i : i + char_window])
            continue
        current.append(sent)
        current_tokens += st
    if current:
        pieces.append(" ".join(current).strip())
    return [p for p in pieces if p.strip()]


def _apply_overlap(chunks: list[Chunk], overlap_tokens: int) -> list[Chunk]:
    out: list[Chunk] = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        cur = chunks[i]
        # Skip overlap when either side is a screen chunk. The tail of a
        # screen mockup is a column-aligned fragment that carries no useful
        # continuity, and prepending prose overlap onto a screen would break
        # the screen's alignment. Screens are atomic units, no overlap.
        if prev.meta.get("contains_screen") or cur.meta.get("contains_screen"):
            out.append(cur)
            continue
        tail = _tail_at_boundary(prev.text, overlap_tokens * 4)
        new_text = (tail + "\n\n" + cur.text).strip()
        out.append(Chunk(
            text=new_text,
            token_count=count_tokens(new_text),
            meta=cur.meta,
        ))
    return out


def _tail_at_boundary(text: str, max_chars: int) -> str:
    """Return a tail of ~max_chars, snapped to paragraph > sentence > word."""
    if len(text) <= max_chars:
        return text
    window = text[len(text) - max_chars:]
    for pattern in (r"\n\n+", r"[.!?]\s+", r"\s"):
        m = re.search(pattern, window)
        if m is not None:
            return window[m.end():].lstrip()
    return window
