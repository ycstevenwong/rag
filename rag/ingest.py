"""Ingest pipeline: parse → chunk → embed → persist.

A file is embedded at most once. Re-ingesting the same content with
different metadata creates a new DocRecord pointing at the existing
FileRecord — no re-parse, no re-embed.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Iterator

import numpy as np

from . import config_facade as cfg
from .bm25_store import BM25Store
from .chunker import chunk_blocks
from .embeddings import Embedder
from .parsers import parse
from .vector_store import ChunkRecord, DocRecord, FaissStore, FileRecord


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

    def ingest_stream(
        self,
        file_path: Path,
        original_filename: str,
        source_type: str = "other",
        tags: list[str] | None = None,
        app_code: str = "",
        version: str = "",
        functionality: str = "",
        managed: bool = False,
    ) -> Iterator[dict]:
        """Yield progress events. Final event is {"type": "done", "result": {...}}."""
        # Manuals are scoped by APP_VERSION_MAP at query time, so the doc
        # itself must stay universal across apps. Strip any incoming app_code
        # for manuals so every ingest path (UI direct, UI pending approval,
        # bulk script, future API) is consistent.
        if source_type == "manual":
            app_code = ""
        sha = _sha256(file_path)
        existing_file = self.vector_store.find_file_by_sha(sha)

        if existing_file is not None:
            existing_doc_id = self.vector_store.find_doc_by_meta(
                existing_file.file_id, source_type, app_code, version, functionality
            )
            if existing_doc_id:
                yield {"type": "done", "result": {
                    "doc_id": existing_doc_id,
                    "n_chunks": 0,
                    "duplicate": True,
                }}
                return

            yield {"type": "stage", "stage": "Linking existing file"}
            doc_id = uuid.uuid4().hex
            self.vector_store.add_doc(DocRecord(
                doc_id=doc_id,
                file_id=existing_file.file_id,
                source_type=source_type,
                tags=list(tags or []),
                app_code=app_code,
                version=version,
                functionality=functionality,
                managed=managed,
            ))
            self.vector_store.persist()
            yield {"type": "done", "result": {
                "doc_id": doc_id,
                "n_chunks": existing_file.n_chunks,
                "duplicate": False,
                "linked": True,
            }}
            return

        yield {"type": "stage", "stage": "Parsing"}
        blocks = parse(file_path)
        if not blocks:
            raise ValueError("No extractable text in document.")

        yield {"type": "stage", "stage": "Chunking"}
        chunks = chunk_blocks(
            blocks,
            target_tokens=cfg.CHUNK_TOKENS,
            overlap_tokens=cfg.CHUNK_OVERLAP,
        )
        if not chunks:
            raise ValueError("Chunking produced no chunks.")

        file_id = uuid.uuid4().hex
        next_id = self.vector_store.next_chunk_id()
        records: list[ChunkRecord] = []
        embed_texts: list[str] = []
        for i, c in enumerate(chunks):
            rec_meta = dict(c.meta)
            rec = ChunkRecord(
                id=next_id + i,
                file_id=file_id,
                text=c.text,
                token_count=c.token_count,
                meta=rec_meta,
            )
            records.append(rec)
            embed_texts.append(_text_for_embedding(c.text, rec_meta))

        yield {"type": "stage", "stage": "Embedding"}
        batches: list[np.ndarray] = []
        for done, total, batch in self.embedder.encode_batched(embed_texts):
            batches.append(batch)
            yield {"type": "progress", "done": done, "total": total}
        vectors = np.vstack(batches)

        yield {"type": "stage", "stage": "Indexing"}
        self.vector_store.add(vectors, records)
        self.vector_store.add_file(FileRecord(
            file_id=file_id,
            sha256=sha,
            filename=original_filename,
            uploaded_at=time.time(),
            n_chunks=len(records),
        ))
        doc_id = uuid.uuid4().hex
        self.vector_store.add_doc(DocRecord(
            doc_id=doc_id,
            file_id=file_id,
            source_type=source_type,
            tags=list(tags or []),
            app_code=app_code,
            version=version,
            functionality=functionality,
            managed=managed,
        ))
        self.bm25_store.rebuild(
            (c.id, c.text) for c in self.vector_store.chunks
        )
        self.vector_store.persist()
        self.bm25_store.persist()
        yield {"type": "done", "result": {
            "doc_id": doc_id,
            "n_chunks": len(records),
            "duplicate": False,
        }}

    def delete(self, doc_id: str) -> int:
        removed = self.vector_store.delete_doc(doc_id)
        if removed:
            self.bm25_store.rebuild(
                (c.id, c.text) for c in self.vector_store.chunks
            )
            self.bm25_store.persist()
        self.vector_store.persist()
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
