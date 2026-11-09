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
        # (BM25Okapi, chunk_ids) replaced as one object, so a search running
        # during a rebuild keeps using a matching pair.
        self._state: tuple | None = None

    @classmethod
    def from_chunks(cls, chunks) -> "BM25Store":
        store = cls()
        store.rebuild((c.id, c.text) for c in chunks)
        return store

    def rebuild(self, corpus: Iterable[tuple[int, str]]) -> None:
        from rank_bm25 import BM25Okapi

        with self._lock:
            tokenized: list[list[str]] = []
            chunk_ids: list[int] = []
            for cid, text in corpus:
                tokenized.append(tokenize(text))
                chunk_ids.append(cid)
            self._state = (BM25Okapi(tokenized), chunk_ids) if tokenized else None

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        state = self._state
        if state is None:
            return []
        bm25, chunk_ids = state
        scores = bm25.get_scores(tokenize(query))
        if len(scores) == 0:
            return []
        import numpy as np
        top_idx = np.argsort(scores)[::-1][:k]
        return [(chunk_ids[i], float(scores[i])) for i in top_idx if scores[i] > 0]
