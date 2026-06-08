"""Helpers for building cited context blocks and source records for the UI."""
from __future__ import annotations

from typing import Any

from .retriever import RetrievedChunk


def build_context_block(retrieved: list[RetrievedChunk], token_budget: int) -> tuple[str, list[dict]]:
    """Return (context_text, source_records).

    Each retrieved chunk is given a 1-indexed citation label that the model is
    instructed to use inline in its answer. Sources are returned to the UI so it
    can show a side panel.
    """
    from .chunker import count_tokens

    lines: list[str] = []
    sources: list[dict] = []
    used = 0
    for i, r in enumerate(retrieved, start=1):
        loc = _format_locator(r.chunk.filename, r.chunk.meta)
        header = f"[{i}] {loc}"
        body = r.chunk.text.strip()
        snippet = f"{header}\n{body}"
        st = count_tokens(snippet)
        if used + st > token_budget and lines:
            break
        used += st
        lines.append(snippet)
        sources.append({
            "n": i,
            "chunk_id": r.chunk.id,
            "doc_id": r.chunk.doc_id,
            "filename": r.chunk.filename,
            "locator": loc,
            "snippet": body[:600],
            "score": r.score,
        })
    return "\n\n---\n\n".join(lines), sources


def _format_locator(filename: str, meta: dict[str, Any]) -> str:
    parts = [filename]
    if meta.get("pages"):
        pages = meta["pages"]
        if len(pages) == 1:
            parts.append(f"p.{pages[0]}")
        else:
            parts.append(f"p.{pages[0]}–{pages[-1]}")
    if meta.get("slide_nums"):
        slides = meta["slide_nums"]
        title = meta.get("slide_title", "")
        slide_str = f"slide {slides[0]}" if len(slides) == 1 else f"slides {slides[0]}–{slides[-1]}"
        if title:
            slide_str += f" \"{title}\""
        parts.append(slide_str)
    if meta.get("heading_path"):
        parts.append(meta["heading_path"])
    if meta.get("sheets"):
        sheets = meta["sheets"]
        rng = meta.get("row_range")
        sheet_str = f"sheet {sheets[0]}"
        if rng:
            sheet_str += f" rows {rng[0]}–{rng[1]}"
        parts.append(sheet_str)
    return ", ".join(parts)
