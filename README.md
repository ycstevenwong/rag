# card-rag

A local-first Retrieval-Augmented Generation app for technical documents.
Flask UI, no database (all state in files), hybrid retrieval (vector + BM25),
multi-turn chat with inline citations, and self-hosted embedding + LLM
endpoints called over HTTP.

## Stack at a glance

| Concern        | Choice                                                      |
|----------------|-------------------------------------------------------------|
| UI / web       | Flask, vanilla JS, SSE streaming                            |
| Embeddings     | Self-hosted HTTP endpoint via `requests.post` (default `bge-m3`) |
| Vector index   | FAISS (`IndexFlatIP`, cosine via normalized vectors)        |
| Keyword index  | `rank_bm25` BM25Okapi                                       |
| Fusion         | Reciprocal Rank Fusion (no extra deps)                      |
| Docs           | PDF (pypdf), DOCX (python-docx), PPTX (python-pptx), XLSX (openpyxl), TXT/MD |
| LLM            | Self-hosted HTTP endpoint via `requests.post` (Ollama-compatible `/api/chat` and `/api/generate`) |
| Persistence    | Plain files under `data/` — no database                     |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_BASE_URL, LLM_MODEL, EMBEDDING_MODEL

python app.py
# open http://127.0.0.1:5000
```

> **Python version note**: as of this project's setup, FAISS wheels
> may not yet support Python 3.14. If install fails, use Python 3.11 or 3.12
> (`brew install python@3.12`).

## How it works

1. **Upload** a document via the UI. The file is parsed format-by-format into
   `Block`s that carry provenance (page, slide, heading path, sheet/rows).
2. **Chunk** the block stream into ~600-token chunks with 80-token overlap.
   Page and slide boundaries are respected; heading paths are prepended to the
   embedded text so retrieval can match by section name.
3. **Embed** chunks by POSTing each one to the embedding endpoint
   (`{EMBEDDING_BASE_URL}/api/embeddings`) and append the returned vector to a
   FAISS `IndexFlatIP`. Chunk metadata + raw text live in `chunks.json`.
4. **BM25** index is rebuilt over the full corpus (cheap at this scale) and
   pickled to `bm25.pkl`.
5. **Query**: the user asks a question. If there is conversation history, the
   LLM rewrites the question as a standalone query. Vector and BM25 searches
   run in parallel; results are fused with RRF; the top N chunks are formatted
   with `[1]`, `[2]`, … markers and sent to the LLM as context.
6. **Answer** streams back via SSE — `LLMProvider` reads the line-delimited
   JSON from `{LLM_BASE_URL}/api/chat`. The UI parses `[n]` markers into
   clickable citations linked to a sources panel.

## Files of note

- `app.py` — Flask routes and SSE.
- `rag/parsers.py` — one parser per format → normalized `Block` records.
- `rag/chunker.py` — token-aware splitting that respects boundaries.
- `rag/embeddings.py` — HTTP embedder (`requests.post` → `/api/embeddings`).
- `rag/vector_store.py` — FAISS + JSON, atomic persist.
- `rag/bm25_store.py` — pickled BM25.
- `rag/retriever.py` — parallel hybrid search + RRF.
- `rag/ingest.py` — orchestrates parse → chunk → embed → persist.
- `rag/chat.py` — multi-turn chat with query rewrite.
- `rag/llm/__init__.py` — HTTP LLM provider (`requests.post` → `/api/chat`, `/api/generate`).

## Tests

```bash
pytest -q
```

Note: `test_retriever.py` requires FAISS and uses a stub embedder so it does
not hit the embedding endpoint. `test_chunker.py` only needs `tiktoken`.

## Configuration

All settings come from environment variables (see `.env`):

```ini
# LLM (self-hosted, Ollama-compatible HTTP API)
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.1

# Embedding (defaults to LLM_BASE_URL if not set)
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3
```

The LLM and embedding endpoints can point at the same host or different hosts —
both are called over plain HTTP with `requests.post`, so any server exposing
Ollama-compatible `/api/chat`, `/api/generate`, and `/api/embeddings` routes
will work.
