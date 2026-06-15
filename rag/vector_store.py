"""FAISS-based vector store with on-disk persistence.

The store keeps three files in lockstep under config.INDEX_DIR:
  vectors.faiss  - the FAISS IndexFlatIP (cosine via L2-normalized vectors)
  chunks.json    - parallel list of chunk records (id, doc_id, text, meta)
  docs.json      - per-document summary records

All writes are atomic: written to a temp path and renamed into place.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ChunkRecord:
    id: int
    doc_id: str
    filename: str
    text: str
    token_count: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocRecord:
    doc_id: str
    filename: str
    sha256: str
    uploaded_at: float
    n_chunks: int


class FaissStore:
    def __init__(self, vectors_path: Path, chunks_path: Path, docs_path: Path):
        self.vectors_path = vectors_path
        self.chunks_path = chunks_path
        self.docs_path = docs_path
        self.dim: int | None = None
        self._lock = threading.RLock()
        self.chunks: list[ChunkRecord] = []
        self.docs: list[DocRecord] = []
        self._index = None
        self._load()

    def _load(self) -> None:
        import faiss

        if self.vectors_path.exists():
            self._index = faiss.read_index(str(self.vectors_path))
            self.dim = self._index.d

        if self.chunks_path.exists():
            data = json.loads(self.chunks_path.read_text())
            self.chunks = [ChunkRecord(**c) for c in data]
        if self.docs_path.exists():
            data = json.loads(self.docs_path.read_text())
            self.docs = [DocRecord(**d) for d in data]

    def add(self, vectors: np.ndarray, records: list[ChunkRecord]) -> None:
        import faiss

        assert vectors.shape[0] == len(records)
        with self._lock:
            if self._index is None:
                self.dim = int(vectors.shape[1])
                self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(vectors)
            self.chunks.extend(records)

    def add_doc(self, doc: DocRecord) -> None:
        with self._lock:
            self.docs.append(doc)

    def has_sha256(self, sha256: str) -> str | None:
        for d in self.docs:
            if d.sha256 == sha256:
                return d.doc_id
        return None

    def delete_doc(self, doc_id: str) -> int:
        """Remove a doc and rebuild the FAISS index. Returns removed chunk count."""
        import faiss

        with self._lock:
            keep_chunks = [c for c in self.chunks if c.doc_id != doc_id]
            removed = len(self.chunks) - len(keep_chunks)
            if removed == 0:
                return 0
            new_index = faiss.IndexFlatIP(self.dim)
            if keep_chunks:
                keep_ids = [c.id for c in keep_chunks]
                old_vecs = self._index.reconstruct_n(0, self._index.ntotal)
                id_to_pos = {c.id: i for i, c in enumerate(self.chunks)}
                positions = [id_to_pos[i] for i in keep_ids]
                kept = old_vecs[positions]
                new_index.add(kept)
            self._index = new_index
            self.chunks = keep_chunks
            self.docs = [d for d in self.docs if d.doc_id != doc_id]
            return removed

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[ChunkRecord, float]]:
        if self._index.ntotal == 0:
            return []
        q = query_vec.reshape(1, -1).astype(np.float32, copy=False)
        scores, ids = self._index.search(q, min(k, self._index.ntotal))
        out: list[tuple[ChunkRecord, float]] = []
        for idx, score in zip(ids[0], scores[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            out.append((self.chunks[idx], float(score)))
        return out

    def next_chunk_id(self) -> int:
        return (self.chunks[-1].id + 1) if self.chunks else 0

    def persist(self) -> None:
        import faiss

        with self._lock:
            _atomic_write_bytes(self.vectors_path, lambda p: faiss.write_index(self._index, str(p)))
            _atomic_write_text(
                self.chunks_path,
                json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False),
            )
            _atomic_write_text(
                self.docs_path,
                json.dumps([asdict(d) for d in self.docs], ensure_ascii=False, indent=2),
            )

    def get_chunk(self, chunk_id: int) -> ChunkRecord | None:
        for c in self.chunks:
            if c.id == chunk_id:
                return c
        return None


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _atomic_write_bytes(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    os.close(fd)
    try:
        writer(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
