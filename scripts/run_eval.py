"""Run an eval set against the retriever and report Recall@N + MRR.

Each eval item is a (question, expected) pair. The retriever runs the question
(applying any per-item filters), and the result is a hit if any retrieved chunk
in the top-N matches one of the expected_doc_ids / expected_filenames /
expected_keywords.

Usage:
    python scripts/run_eval.py                          # uses data/eval/eval.json
    python scripts/run_eval.py path/to/set.json
    python scripts/run_eval.py --top-n 10               # override top-N
    python scripts/run_eval.py --verbose                # print every question

WARNING: do NOT run this while `python app.py` is also running. The retriever
holds its own in-memory copy of the FAISS/BM25 stores; concurrent persists
from the running app would corrupt the index.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from rag.bm25_store import BM25Store
from rag.embeddings import Embedder
from rag.retriever import HybridRetriever
from rag.vector_store import FaissStore


DEFAULT_EVAL_PATH = ROOT / "data" / "eval" / "eval.json"


@dataclass
class RetrievedRow:
    rank: int
    filename: str
    score: float
    matched: bool
    snippet: str


@dataclass
class EvalResult:
    item_id: str
    question: str
    hit_rank: int | None
    top_filenames: list[str]
    rows: list[RetrievedRow]
    item: dict


def _chunk_satisfies(retrieved, file, item) -> bool:
    """Return True if this retrieved result hits one of the expected criteria."""
    expected_doc_ids = set(item.get("expected_doc_ids") or [])
    expected_filenames = [f.lower() for f in (item.get("expected_filenames") or [])]
    expected_keywords = [k.lower() for k in (item.get("expected_keywords") or [])]

    if expected_doc_ids and retrieved.doc is not None:
        if retrieved.doc.doc_id in expected_doc_ids:
            return True
    if expected_filenames and file is not None:
        fname = (file.filename or "").lower()
        if any(ef in fname for ef in expected_filenames):
            return True
    if expected_keywords:
        text = (retrieved.chunk.text or "").lower()
        if any(kw in text for kw in expected_keywords):
            return True
    return False


def run_one(item, retriever, vector_store, top_n) -> EvalResult:
    filters = item.get("filters") or None
    results = retriever.search(item["question"], top_n=top_n, filters=filters)

    hit_rank = None
    top_filenames: list[str] = []
    rows: list[RetrievedRow] = []
    for rank, r in enumerate(results, start=1):
        file = vector_store.get_file(r.chunk.file_id)
        fname = file.filename if file else "(unknown)"
        if rank <= 5:
            top_filenames.append(fname)
        matched = _chunk_satisfies(r, file, item)
        if hit_rank is None and matched:
            hit_rank = rank
        snippet = (r.chunk.text or "").strip().replace("\n", " ")
        if len(snippet) > 140:
            snippet = snippet[:140] + "…"
        rows.append(RetrievedRow(
            rank=rank, filename=fname, score=float(r.score),
            matched=matched, snippet=snippet,
        ))

    return EvalResult(
        item_id=item.get("id", "?"),
        question=item["question"],
        hit_rank=hit_rank,
        top_filenames=top_filenames,
        rows=rows,
        item=item,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the retriever against an eval set.")
    parser.add_argument(
        "path", nargs="?", default=str(DEFAULT_EVAL_PATH),
        help="Path to eval JSON (default: data/eval/eval.json)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Top-N for Recall@N (default: 10)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every question's result, not just failures.",
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="For every question, dump the top-N retrieved chunks with filename, score, and snippet.",
    )
    args = parser.parse_args()

    eval_path = Path(args.path)
    if not eval_path.exists():
        print(f"Eval set not found: {eval_path}", file=sys.stderr)
        return 1
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    items = data.get("items") if isinstance(data, dict) else data
    if not items:
        print(f"No items in eval set: {eval_path}", file=sys.stderr)
        return 1

    embedder = Embedder(cfg.EMBEDDING_MODEL, cfg.EMBEDDING_BASE_URL, cfg.EMBEDDING_API_KEY)
    vector_store = FaissStore(
        vectors_path=cfg.VECTORS_PATH,
        chunks_path=cfg.CHUNKS_PATH,
        docs_path=cfg.DOCS_PATH,
        files_path=cfg.FILES_PATH,
    )
    bm25_store = BM25Store(cfg.BM25_PATH)
    retriever = HybridRetriever(vector_store, bm25_store, embedder)

    print(f"Running {len(items)} questions through the retriever (top_n={args.top_n})...\n")

    results: list[EvalResult] = []
    for i, item in enumerate(items, start=1):
        try:
            r = run_one(item, retriever, vector_store, args.top_n)
        except Exception as exc:
            print(f"  [{i}/{len(items)}] {item.get('id', '?')}  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            r = EvalResult(
                item_id=item.get("id", "?"), question=item["question"],
                hit_rank=None, top_filenames=[], rows=[], item=item,
            )
        results.append(r)
        marker = f"✓ rank {r.hit_rank}" if r.hit_rank else "✗ MISS"
        line = f"  [{i:>3}/{len(items)}] {r.item_id:<30} {marker}"
        if args.verbose or args.detailed or r.hit_rank is None:
            print(line)

        if args.detailed:
            filters = r.item.get("filters") or {}
            expected_bits = []
            if r.item.get("expected_doc_ids"): expected_bits.append(f"doc_ids={r.item['expected_doc_ids']}")
            if r.item.get("expected_filenames"): expected_bits.append(f"filenames={r.item['expected_filenames']}")
            if r.item.get("expected_keywords"): expected_bits.append(f"keywords={r.item['expected_keywords']}")
            print(f"        Question: {r.question}")
            if filters:
                print(f"        Filters:  {json.dumps(filters)}")
            if expected_bits:
                print(f"        Expected: {'; '.join(expected_bits)}")
            print(f"        Retrieved (top {len(r.rows)}):")
            for row in r.rows:
                mark = "★" if row.matched else " "
                print(f"          {mark} {row.rank:>2}. [{row.score:6.4f}] {row.filename}")
                print(f"                 {row.snippet}")
            print()

    hits = [r for r in results if r.hit_rank is not None]
    n_total = len(results)
    n_hit = len(hits)
    recall = n_hit / n_total if n_total else 0.0
    mrr = sum(1.0 / r.hit_rank for r in hits) / n_total if n_total else 0.0

    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Questions:       {n_total}")
    print(f"  Hits @ {args.top_n:<4}     {n_hit}")
    print(f"  Recall @ {args.top_n:<4}   {recall * 100:5.1f}%")
    print(f"  MRR:             {mrr:.3f}")
    if hits:
        ranks = sorted(r.hit_rank for r in hits)
        median = ranks[len(ranks) // 2]
        print(f"  Median hit rank: {median}")

    failures = [r for r in results if r.hit_rank is None]
    if failures:
        print()
        print("=" * 60)
        print(f"Failures ({len(failures)})")
        print("=" * 60)
        for r in failures:
            print(f"  [{r.item_id}] {r.question}")
            for j, fname in enumerate(r.top_filenames, start=1):
                print(f"     {j}. {fname}")

    return 0 if recall >= 0.5 else 2  # non-zero exit on poor recall (handy for CI)


if __name__ == "__main__":
    sys.exit(main())
