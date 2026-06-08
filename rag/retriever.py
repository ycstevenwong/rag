"""Hybrid retriever: vector + BM25 fused with Reciprocal Rank Fusion."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .bm25_store import BM25Store
from .embeddings import Embedder
from .vector_store import ChunkRecord, FaissStore


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    vector_rank: int | None
    bm25_rank: int | None


class HybridRetriever:
    def __init__(
        self,
        vector_store: FaissStore,
        bm25_store: BM25Store,
        embedder: Embedder,
    ):
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self.embedder = embedder

    def search(
        self,
        query: str,
        *,
        k_vector: int = 20,
        k_bm25: int = 20,
        top_n: int = 6,
        rrf_k: int = 60,
    ) -> list[RetrievedChunk]:
        if not self.vector_store.chunks:
            return []

        with ThreadPoolExecutor(max_workers=2) as ex:
            vec_future = ex.submit(self._vector_search, query, k_vector)
            bm25_future = ex.submit(self.bm25_store.search, query, k_bm25)
            vec_results = vec_future.result()
            bm25_results = bm25_future.result()

        vec_ranks: dict[int, int] = {}
        for rank, (chunk, _score) in enumerate(vec_results):
            vec_ranks[chunk.id] = rank

        bm25_ranks: dict[int, int] = {}
        for rank, (chunk_id, _score) in enumerate(bm25_results):
            bm25_ranks[chunk_id] = rank

        all_ids = set(vec_ranks) | set(bm25_ranks)
        fused: list[tuple[int, float]] = []
        for cid in all_ids:
            s = 0.0
            if cid in vec_ranks:
                s += 1.0 / (rrf_k + vec_ranks[cid] + 1)
            if cid in bm25_ranks:
                s += 1.0 / (rrf_k + bm25_ranks[cid] + 1)
            fused.append((cid, s))

        fused.sort(key=lambda x: x[1], reverse=True)
        top = fused[:top_n]

        out: list[RetrievedChunk] = []
        for cid, score in top:
            chunk = self.vector_store.get_chunk(cid)
            if chunk is None:
                continue
            out.append(RetrievedChunk(
                chunk=chunk,
                score=score,
                vector_rank=vec_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
            ))
        return out

    def _vector_search(self, query: str, k: int) -> list[tuple[ChunkRecord, float]]:
        q_vec = self.embedder.encode_query(query)
        return self.vector_store.search(q_vec, k)
