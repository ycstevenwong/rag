"""FAISS-based vector store with on-disk persistence.

The store keeps four files in lockstep under config.INDEX_DIR:
  vectors.faiss  - the FAISS IndexFlatIP (cosine via L2-normalized vectors)
  chunks.json    - parallel list of chunk records (id, file_id, text, meta)
  files.json     - per-file record (file_id, sha256, filename, n_chunks)
  docs.json      - per-(file, metadata) record; many DocRecord -> one FileRecord

The decoupling lets a single physical file be shared by multiple metadata
"views" (e.g., the same PDF tagged for two different app_codes) without
re-embedding the content.

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
    file_id: str
    text: str
    token_count: int
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileRecord:
    file_id: str
    sha256: str
    filename: str
    uploaded_at: float
    n_chunks: int


@dataclass
class DocRecord:
    doc_id: str
    file_id: str
    source_type: str = "other"
    tags: list[str] = field(default_factory=list)
    app_code: str = ""
    version: str = ""
    functionality: str = ""
    managed: bool = False


class FaissStore:
    def __init__(
        self,
        vectors_path: Path,
        chunks_path: Path,
        docs_path: Path,
        files_path: Path,
    ):
        self.vectors_path = vectors_path
        self.chunks_path = chunks_path
        self.docs_path = docs_path
        self.files_path = files_path
        self.dim: int | None = None
        self._lock = threading.RLock()
        self.chunks: list[ChunkRecord] = []
        self.docs: list[DocRecord] = []
        self.files: list[FileRecord] = []
        self._index = None
        self._file_by_id: dict[str, FileRecord] = {}
        self._file_by_sha: dict[str, FileRecord] = {}
        self._docs_by_file: dict[str, list[DocRecord]] = {}
        self._load()

    def _load(self) -> None:
        import faiss

        if self.vectors_path.exists():
            self._index = faiss.read_index(str(self.vectors_path))
            self.dim = self._index.d

        if self.chunks_path.exists():
            data = json.loads(self.chunks_path.read_text(encoding="utf-8"))
            self.chunks = [ChunkRecord(**c) for c in data]
        if self.files_path.exists():
            data = json.loads(self.files_path.read_text(encoding="utf-8"))
            self.files = [FileRecord(**f) for f in data]
        if self.docs_path.exists():
            data = json.loads(self.docs_path.read_text(encoding="utf-8"))
            self.docs = [DocRecord(**d) for d in data]

        self._rebuild_indices()

    def _rebuild_indices(self) -> None:
        self._file_by_id = {f.file_id: f for f in self.files}
        self._file_by_sha = {f.sha256: f for f in self.files}
        self._docs_by_file = {}
        for d in self.docs:
            self._docs_by_file.setdefault(d.file_id, []).append(d)

    def add(self, vectors: np.ndarray, records: list[ChunkRecord]) -> None:
        import faiss

        assert vectors.shape[0] == len(records)
        with self._lock:
            if self._index is None:
                self.dim = int(vectors.shape[1])
                self._index = faiss.IndexFlatIP(self.dim)
            self._index.add(vectors)
            self.chunks.extend(records)

    def add_file(self, file: FileRecord) -> None:
        with self._lock:
            self.files.append(file)
            self._file_by_id[file.file_id] = file
            self._file_by_sha[file.sha256] = file

    def add_doc(self, doc: DocRecord) -> None:
        with self._lock:
            self.docs.append(doc)
            self._docs_by_file.setdefault(doc.file_id, []).append(doc)

    def find_file_by_sha(self, sha256: str) -> FileRecord | None:
        return self._file_by_sha.get(sha256)

    def get_file(self, file_id: str) -> FileRecord | None:
        return self._file_by_id.get(file_id)

    def docs_for_file(self, file_id: str) -> list[DocRecord]:
        return self._docs_by_file.get(file_id, [])

    def find_doc_by_meta(
        self,
        file_id: str,
        source_type: str,
        app_code: str,
        version: str,
        functionality: str,
    ) -> str | None:
        for d in self._docs_by_file.get(file_id, []):
            if (
                d.source_type == source_type
                and d.app_code == app_code
                and d.version == version
                and d.functionality == functionality
            ):
                return d.doc_id
        return None

    def delete_doc(self, doc_id: str) -> int:
        """Remove a DocRecord. If it was the last reference to its file,
        also remove the FileRecord and its chunks (rebuilding the FAISS
        index). Returns the number of chunks removed (0 when only the
        doc record was unlinked)."""
        import faiss

        with self._lock:
            doc = next((d for d in self.docs if d.doc_id == doc_id), None)
            if doc is None:
                return 0
            file_id = doc.file_id
            self.docs = [d for d in self.docs if d.doc_id != doc_id]
            siblings = [d for d in self._docs_by_file.get(file_id, []) if d.doc_id != doc_id]
            self._docs_by_file[file_id] = siblings
            if siblings:
                return 0

            keep_chunks = [c for c in self.chunks if c.file_id != file_id]
            removed = len(self.chunks) - len(keep_chunks)
            if self.dim is None:
                self.dim = self._index.d if self._index is not None else 0
            new_index = faiss.IndexFlatIP(self.dim) if self.dim else None
            if keep_chunks and self._index is not None:
                keep_ids = [c.id for c in keep_chunks]
                old_vecs = self._index.reconstruct_n(0, self._index.ntotal)
                id_to_pos = {c.id: i for i, c in enumerate(self.chunks)}
                positions = [id_to_pos[i] for i in keep_ids]
                kept = old_vecs[positions]
                new_index.add(kept)
            self._index = new_index
            self.chunks = keep_chunks
            file_to_remove = self._file_by_id.get(file_id)
            self.files = [f for f in self.files if f.file_id != file_id]
            self._file_by_id.pop(file_id, None)
            if file_to_remove is not None:
                self._file_by_sha.pop(file_to_remove.sha256, None)
            self._docs_by_file.pop(file_id, None)
            return removed

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[ChunkRecord, float]]:
        if self._index is None or self._index.ntotal == 0:
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
            if self._index is not None:
                _atomic_write_bytes(self.vectors_path, lambda p: faiss.write_index(self._index, str(p)))
            _atomic_write_text(
                self.chunks_path,
                json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=2),
            )
            _atomic_write_text(
                self.files_path,
                json.dumps([asdict(f) for f in self.files], ensure_ascii=False, indent=2),
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
