"""BM25 keyword index over the chunk corpus.

Held in memory only and rebuilt from the chunks at startup, so it can't drift
from them after a crash and there's no pickle file to trust.
"""
from __future__ import annotations

import re
import threading
from typing import Iterable


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Store:
    def __init__(self):
        self._lock = threading.RLock()
        self.tokenized: list[list[str]] = []
        self.chunk_ids: list[int] = []
        self._bm25 = None

    @classmethod
    def from_chunks(cls, chunks) -> "BM25Store":
        store = cls()
        store.rebuild((c.id, c.text) for c in chunks)
        return store

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
