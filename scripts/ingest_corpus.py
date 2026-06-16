"""Bulk-ingest a directory (or single file) of documents into the RAG index.

Usage:
    python scripts/ingest_corpus.py                  # uses data/corpus/
    python scripts/ingest_corpus.py path/to/docs/    # custom directory
    python scripts/ingest_corpus.py path/to/file.pdf # single file

Idempotent: documents already in the index (by SHA-256) are skipped.

WARNING: do NOT run this while `python app.py` is also running. Both
processes hold their own in-memory copies of the FAISS/BM25 stores and
the last writer wins on persist.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as cfg
from rag.bm25_store import BM25Store
from rag.embeddings import Embedder
from rag.ingest import IngestPipeline
from rag.vector_store import FaissStore


ALLOWED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".txt", ".md"}
DEFAULT_CORPUS_DIR = ROOT / "data" / "corpus"


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in ALLOWED_EXTENSIONS else []
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        )
    return []


def main() -> int:
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    target = Path(arg).resolve() if arg else DEFAULT_CORPUS_DIR

    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        if target == DEFAULT_CORPUS_DIR:
            print(
                f"Create {DEFAULT_CORPUS_DIR} and put documents there, "
                "or pass a path argument.",
                file=sys.stderr,
            )
        return 1

    files = collect_files(target)
    if not files:
        print(f"No supported documents found under {target}", file=sys.stderr)
        return 1

    embedder = Embedder(cfg.EMBEDDING_MODEL, cfg.EMBEDDING_BASE_URL, cfg.EMBEDDING_API_KEY)
    vector_store = FaissStore(
        vectors_path=cfg.VECTORS_PATH,
        chunks_path=cfg.CHUNKS_PATH,
        docs_path=cfg.DOCS_PATH,
    )
    bm25_store = BM25Store(cfg.BM25_PATH)
    ingest = IngestPipeline(vector_store, bm25_store, embedder)

    print(f"Ingesting {len(files)} document(s) from {target}\n")
    n_added = n_dup = n_err = 0

    for i, file_path in enumerate(files, 1):
        label = (
            file_path.relative_to(target) if target.is_dir() else file_path.name
        )
        print(f"[{i}/{len(files)}] {label}")
        try:
            for event in ingest.ingest_stream(file_path, original_filename=file_path.name):
                t = event["type"]
                if t == "stage":
                    print(f"  {event['stage']}...")
                elif t == "progress":
                    sys.stdout.write(f"\r  Embedding batch {event['done']}/{event['total']}")
                    sys.stdout.flush()
                    if event["done"] == event["total"]:
                        print()
                elif t == "done":
                    r = event["result"]
                    if r.get("duplicate"):
                        print("  Already indexed -> skipped")
                        n_dup += 1
                    else:
                        print(f"  Indexed {r['n_chunks']} chunks")
                        n_added += 1
        except Exception as exc:
            print(f"  ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
            n_err += 1

    print(f"\nDone. Added: {n_added}  Duplicate: {n_dup}  Errors: {n_err}")
    print(f"Total docs in index: {len(vector_store.docs)}")
    return 0 if n_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
