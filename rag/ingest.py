"""Ingest pipeline: parse → chunk → embed → persist."""
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path

from . import config_facade as cfg
from .bm25_store import BM25Store
from .chunker import chunk_blocks
from .embeddings import Embedder
from .parsers import parse
from .vector_store import ChunkRecord, DocRecord, FaissStore


class IngestPipeline:
    def __init__(
        self,
        vector_store: FaissStore,
        bm25_store: BM25Store,
        embedder: Embedder,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder

    def ingest(self, file_path: Path, original_filename: str) -> dict:
        sha = _sha256(file_path)
        existing = self.vector_store.has_sha256(sha)
        if existing:
            return {"doc_id": existing, "n_chunks": 0, "duplicate": True}

        blocks = parse(file_path)
        if not blocks:
            raise ValueError("No extractable text in document.")

        chunks = chunk_blocks(
            blocks,
            target_tokens=cfg.CHUNK_TOKENS,
            overlap_tokens=cfg.CHUNK_OVERLAP,
        )
        if not chunks:
            raise ValueError("Chunking produced no chunks.")

        doc_id = uuid.uuid4().hex
        next_id = self.vector_store.next_chunk_id()
        records: list[ChunkRecord] = []
        embed_texts: list[str] = []
        for i, c in enumerate(chunks):
            rec_meta = dict(c.meta)
            rec = ChunkRecord(
                id=next_id + i,
                doc_id=doc_id,
                filename=original_filename,
                text=c.text,
                token_count=c.token_count,
                meta=rec_meta,
            )
            records.append(rec)
            embed_texts.append(_text_for_embedding(c.text, rec_meta))

        vectors = self.embedder.encode(embed_texts)
        self.vector_store.add(vectors, records)
        self.vector_store.add_doc(DocRecord(
            doc_id=doc_id,
            filename=original_filename,
            sha256=sha,
            uploaded_at=time.time(),
            n_chunks=len(records),
        ))

        self.bm25_store.rebuild(
            (c.id, c.text) for c in self.vector_store.chunks
        )
        self.vector_store.persist()
        self.bm25_store.persist()
        return {"doc_id": doc_id, "n_chunks": len(records), "duplicate": False}

    def delete(self, doc_id: str) -> int:
        removed = self.vector_store.delete_doc(doc_id)
        if removed:
            self.bm25_store.rebuild(
                (c.id, c.text) for c in self.vector_store.chunks
            )
            self.vector_store.persist()
            self.bm25_store.persist()
        return removed


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _text_for_embedding(text: str, meta: dict) -> str:
    """Prepend heading or slide title context so embeddings reflect structure."""
    prefix_parts = []
    if meta.get("heading_path"):
        prefix_parts.append(meta["heading_path"])
    if meta.get("slide_title"):
        prefix_parts.append(meta["slide_title"])
    if prefix_parts:
        return " | ".join(prefix_parts) + "\n\n" + text
    return text
