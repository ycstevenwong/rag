"""Ingest pipeline: parse → chunk → embed → persist.

A file is embedded at most once. Re-ingesting the same content with
different metadata creates a new DocRecord pointing at the existing
FileRecord — no re-parse, no re-embed.
"""
from __future__ import annotations

import hashlib
import threading
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
        # Held across each multi-step change to the stores, never across a
        # yield or an embedding call.
        self._write_lock = threading.RLock()

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

        def new_doc(file_id: str) -> DocRecord:
            return DocRecord(
                doc_id=uuid.uuid4().hex,
                file_id=file_id,
                source_type=source_type,
                tags=list(tags or []),
                app_code=app_code,
                version=version,
                functionality=functionality,
                managed=managed,
            )

        def attach_to_existing_file(file: FileRecord) -> dict:
            existing_doc_id = self.vector_store.find_doc_by_meta(
                file.file_id, source_type, app_code, version, functionality
            )
            if existing_doc_id:
                return {"doc_id": existing_doc_id, "n_chunks": 0, "duplicate": True}
            doc = new_doc(file.file_id)
            self.vector_store.add_doc(doc)
            try:
                self.vector_store.persist()
            except BaseException:
                self._undo_doc(doc.doc_id)
                raise
            return {
                "doc_id": doc.doc_id,
                "n_chunks": file.n_chunks,
                "duplicate": False,
                "linked": True,
            }

        with self._write_lock:
            existing_file = self.vector_store.find_file_by_sha(sha)
            result = attach_to_existing_file(existing_file) if existing_file else None
        if result is not None:
            if result.get("linked"):
                yield {"type": "stage", "stage": "Linking existing file"}
            yield {"type": "done", "result": result}
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

        embed_texts = [_text_for_embedding(c.text, c.meta) for c in chunks]

        yield {"type": "stage", "stage": "Embedding"}
        batches: list[np.ndarray] = []
        for done, total, batch in self.embedder.encode_batched(embed_texts):
            batches.append(batch)
            yield {"type": "progress", "done": done, "total": total}
        vectors = np.vstack(batches)

        yield {"type": "stage", "stage": "Indexing"}
        # Embedding can take minutes, so IDs are allocated here rather than
        # before it, and the store is re-checked for a concurrent ingest of the
        # same bytes that finished in the meantime.
        with self._write_lock:
            existing_file = self.vector_store.find_file_by_sha(sha)
            if existing_file is not None:
                result = attach_to_existing_file(existing_file)
            else:
                file_id = uuid.uuid4().hex
                next_id = self.vector_store.next_chunk_id()
                records = [
                    ChunkRecord(
                        id=next_id + i,
                        file_id=file_id,
                        text=c.text,
                        token_count=c.token_count,
                        meta=dict(c.meta),
                    )
                    for i, c in enumerate(chunks)
                ]
                self.vector_store.add(vectors, records)
                self.vector_store.add_file(FileRecord(
                    file_id=file_id,
                    sha256=sha,
                    filename=original_filename,
                    uploaded_at=time.time(),
                    n_chunks=len(records),
                ))
                doc = new_doc(file_id)
                self.vector_store.add_doc(doc)
                try:
                    self.bm25_store.rebuild(
                        (c.id, c.text) for c in self.vector_store.chunks
                    )
                    self.vector_store.persist()
                except BaseException:
                    self._undo_doc(doc.doc_id)
                    raise
                result = {
                    "doc_id": doc.doc_id,
                    "n_chunks": len(records),
                    "duplicate": False,
                }
        yield {"type": "done", "result": result}

    def _undo_doc(self, doc_id: str) -> None:
        """Take a doc whose commit failed back out of memory, so a later persist
        can't write it. Also tries to bring the files back in line, since a failed
        persist may have replaced some of them. If that fails too, the next load
        reports the mismatch."""
        self.vector_store.delete_doc(doc_id)
        self.bm25_store.rebuild((c.id, c.text) for c in self.vector_store.chunks)
        try:
            self.vector_store.persist()
        except OSError as exc:
            print(f"[ingest] could not rewrite index after a failed commit: {exc}", flush=True)

    def delete(self, doc_id: str) -> int:
        with self._write_lock:
            removed = self.vector_store.delete_doc(doc_id)
            if removed:
                self.bm25_store.rebuild(
                    (c.id, c.text) for c in self.vector_store.chunks
                )
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
