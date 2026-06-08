# card-rag

A local-first Retrieval-Augmented Generation app for technical documents.
Flask UI, no database (all state in files), hybrid retrieval (vector + BM25),
multi-turn chat with inline citations, and a pluggable LLM backend
(Anthropic / OpenAI / Ollama).

## Stack at a glance

| Concern        | Choice                                                      |
|----------------|-------------------------------------------------------------|
| UI / web       | Flask, vanilla JS, SSE streaming                            |
| Embeddings     | `sentence-transformers` (default `BAAI/bge-base-en-v1.5`)   |
| Vector index   | FAISS (`IndexFlatIP`, cosine via normalized vectors)        |
| Keyword index  | `rank_bm25` BM25Okapi                                       |
| Fusion         | Reciprocal Rank Fusion (no extra deps)                      |
| Docs           | PDF (pypdf), DOCX (python-docx), PPTX (python-pptx), XLSX (openpyxl), TXT/MD |
| LLM            | Anthropic / OpenAI / Ollama (lazy-imported per provider)    |
| Persistence    | Plain files under `data/` — no database                     |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_PROVIDER, model, and the relevant API key

python app.py
# open http://127.0.0.1:5000
```

> **Python version note**: as of this project's setup, FAISS and torch wheels
> may not yet support Python 3.14. If install fails, use Python 3.11 or 3.12
> (`brew install python@3.12`).

## How it works

1. **Upload** a document via the UI. The file is parsed format-by-format into
   `Block`s that carry provenance (page, slide, heading path, sheet/rows).
2. **Chunk** the block stream into ~600-token chunks with 80-token overlap.
   Page and slide boundaries are respected; heading paths are prepended to the
   embedded text so retrieval can match by section name.
3. **Embed** chunks locally with sentence-transformers (BGE) and append to a
   FAISS `IndexFlatIP`. Chunk metadata + raw text live in `chunks.json`.
4. **BM25** index is rebuilt over the full corpus (cheap at this scale) and
   pickled to `bm25.pkl`.
5. **Query**: the user asks a question. If there is conversation history, the
   LLM rewrites the question as a standalone query. Vector and BM25 searches
   run in parallel; results are fused with RRF; the top N chunks are formatted
   with `[1]`, `[2]`, … markers and sent to the LLM as context.
6. **Answer** streams back via SSE. The UI parses `[n]` markers into clickable
   citations linked to a sources panel.

## Files of note

- `app.py` — Flask routes and SSE.
- `rag/parsers.py` — one parser per format → normalized `Block` records.
- `rag/chunker.py` — token-aware splitting that respects boundaries.
- `rag/embeddings.py` — sentence-transformers wrapper, lazy model load.
- `rag/vector_store.py` — FAISS + JSON, atomic persist.
- `rag/bm25_store.py` — pickled BM25.
- `rag/retriever.py` — parallel hybrid search + RRF.
- `rag/ingest.py` — orchestrates parse → chunk → embed → persist.
- `rag/chat.py` — multi-turn chat with query rewrite.
- `rag/llm/` — pluggable provider implementations.

## Tests

```bash
pytest -q
```

Note: `test_retriever.py` requires FAISS and uses a stub embedder so it does
not download the BGE model. `test_chunker.py` only needs `tiktoken`.

## Switching providers

Edit `.env`:

```ini
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=sk-...
```

or for Ollama (requires `ollama serve` running):

```ini
LLM_PROVIDER=ollama
LLM_MODEL=llama3.1
OLLAMA_BASE_URL=http://localhost:11434
```

No code changes are needed.
