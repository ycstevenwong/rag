"""Hybrid retriever: vector + BM25 fused with Reciprocal Rank Fusion."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import config_facade as cfg
from .bm25_store import BM25Store
from .embeddings import Embedder
from .vector_store import ChunkRecord, DocRecord, FaissStore


MMR_LAMBDA = 0.7        # higher = favor query relevance, lower = favor diversity
MMR_POOL_MULTIPLIER = 3 # consider 3× top_n RRF candidates before MMR picks the final top_n


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    score: float
    vector_rank: int | None
    bm25_rank: int | None
    filename: str = ""
    doc: DocRecord | None = None


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
        top_n: int = 10,
        rrf_k: int = 60,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        if not self.vector_store.chunks:
            return []

        q_vec = self.embedder.encode_query(query)
        fetch_mult = 3 if filters else 1
        with ThreadPoolExecutor(max_workers=2) as ex:
            vec_future = ex.submit(self.vector_store.search, q_vec, k_vector * fetch_mult)
            bm25_future = ex.submit(self.bm25_store.search, query, k_bm25 * fetch_mult)
            vec_results = vec_future.result()
            bm25_results = bm25_future.result()

        # For each surviving chunk, remember which DocRecord we matched
        # against — used for citation metadata downstream.
        matching_doc: dict[int, DocRecord] = {}

        def chunk_passes(chunk_id: int) -> bool:
            chunk = self.vector_store.get_chunk(chunk_id)
            if chunk is None:
                return False
            docs = self.vector_store.docs_for_file(chunk.file_id)
            if not docs:
                return False
            if not filters:
                matching_doc[chunk_id] = docs[0]
                return True
            for d in docs:
                if _doc_matches(d, filters):
                    matching_doc[chunk_id] = d
                    return True
            return False

        vec_ranks: dict[int, int] = {}
        for chunk, _score in vec_results:
            if chunk_passes(chunk.id):
                vec_ranks[chunk.id] = len(vec_ranks)
                if len(vec_ranks) >= k_vector:
                    break

        bm25_ranks: dict[int, int] = {}
        for chunk_id, _score in bm25_results:
            if chunk_passes(chunk_id):
                bm25_ranks[chunk_id] = len(bm25_ranks)
                if len(bm25_ranks) >= k_bm25:
                    break

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
        fused_score_by_id = {cid: s for cid, s in fused}

        # Apply MMR over a larger RRF pool, then take top_n.
        pool_size = min(len(fused), max(top_n * MMR_POOL_MULTIPLIER, top_n))
        pool_ids = [cid for cid, _ in fused[:pool_size]]
        selected_ids = _mmr_select(
            q_vec, pool_ids, self.vector_store, top_n, lambda_=MMR_LAMBDA
        )

        out: list[RetrievedChunk] = []
        for cid in selected_ids:
            chunk = self.vector_store.get_chunk(cid)
            if chunk is None:
                continue
            file = self.vector_store.get_file(chunk.file_id)
            filename = file.filename if file else ""
            out.append(RetrievedChunk(
                chunk=chunk,
                score=fused_score_by_id.get(cid, 0.0),
                vector_rank=vec_ranks.get(cid),
                bm25_rank=bm25_ranks.get(cid),
                filename=filename,
                doc=matching_doc.get(cid),
            ))
        return out


def _mmr_select(
    query_vec: np.ndarray,
    candidate_ids: list[int],
    vector_store: FaissStore,
    top_n: int,
    *,
    lambda_: float = 0.7,
) -> list[int]:
    """Maximal Marginal Relevance selection over the candidate pool.

    Picks the candidate most similar to the query first, then iteratively picks
    the candidate that maximizes
        lambda_ * sim(c, q) - (1 - lambda_) * max_{s in selected} sim(c, s)
    so the final set is relevant but doesn't pile up near-duplicates.
    Vectors come from the FAISS store and are already L2-normalized, so dot
    products are cosine similarities.
    """
    if len(candidate_ids) <= top_n:
        return list(candidate_ids)

    kept_ids, cand_vecs = vector_store.get_vectors(candidate_ids)
    if not kept_ids:
        return list(candidate_ids)[:top_n]

    q = query_vec.astype(np.float32, copy=False)
    query_sims = cand_vecs @ q  # shape (N,)

    selected_local: list[int] = []
    available: list[int] = list(range(len(kept_ids)))

    first = int(np.argmax(query_sims))
    selected_local.append(first)
    available.remove(first)

    while len(selected_local) < top_n and available:
        sel_vecs = cand_vecs[selected_local]                    # (k, D)
        avail_vecs = cand_vecs[available]                       # (M, D)
        existing = avail_vecs @ sel_vecs.T                      # (M, k)
        max_existing = existing.max(axis=1)                     # (M,)
        avail_query = query_sims[available]                     # (M,)
        mmr = lambda_ * avail_query - (1.0 - lambda_) * max_existing
        best_local = int(np.argmax(mmr))
        chosen = available[best_local]
        selected_local.append(chosen)
        available.pop(best_local)

    return [kept_ids[i] for i in selected_local]


def _doc_matches(doc: DocRecord, filters: dict) -> bool:
    """Filter semantics: empty value on the doc = "universal" (passes any filter
    for that field). Non-empty must match the user's choice. Version is
    auto-resolved per (app_code, functionality) via APP_VERSION_MAP."""
    st = filters.get("source_type")
    if st and doc.source_type and doc.source_type != st:
        return False
    app_code = filters.get("app_code")
    if app_code and doc.app_code and doc.app_code != app_code:
        return False
    if app_code:
        version_map = cfg.APP_VERSION_MAP.get(app_code) or {}
        effective = version_map.get(doc.functionality, version_map.get("*"))
        if effective and doc.version:
            allowed = [effective] if isinstance(effective, str) else list(effective)
            if doc.version not in allowed:
                return False
    tags = filters.get("tags") or []
    if tags and doc.tags:
        doc_tags = set(doc.tags)
        for t in tags:
            if t not in doc_tags:
                return False
    return True
