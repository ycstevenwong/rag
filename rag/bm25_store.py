"""BM25 keyword index over the chunk corpus, pickled to disk."""
from __future__ import annotations

import pickle
import re
import threading
from pathlib import Path
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Store:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self.tokenized: list[list[str]] = []
        self.chunk_ids: list[int] = []
        self._bm25 = None
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("rb") as f:
            data = pickle.load(f)
        self.tokenized = data["tokenized"]
        self.chunk_ids = data["chunk_ids"]
        if self.tokenized:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self.tokenized)

    def rebuild(self, corpus: Iterable[tuple[int, str]]) -> None:
        from rank_bm25 import BM25Okapi

        with self._lock:
            self.tokenized = []
            self.chunk_ids = []
            for cid, text in corpus:
                self.tokenized.append(tokenize(text))
                self.chunk_ids.append(cid)
            self._bm25 = BM25Okapi(self.tokenized) if self.tokenized else None

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        if len(scores) == 0:
            return []
        import numpy as np
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def persist(self) -> None:
        from .vector_store import atomic_write_bytes

        with self._lock:
            data = {"tokenized": self.tokenized, "chunk_ids": self.chunk_ids}

            def _write(p: Path) -> None:
                with p.open("wb") as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Routes through the same retry-aware os.replace as the FAISS /
            # JSON files, so a transient Windows file lock on bm25.pkl no
            # longer leaves the keyword index out of sync with the vector
            # index.
            atomic_write_bytes(self.path, _write)
