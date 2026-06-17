# card-rag

A local-first Retrieval-Augmented Generation app for technical documents.
Flask UI, no database (all state in files), hybrid retrieval (vector + BM25),
multi-turn chat with inline citations, and self-hosted embedding + LLM
endpoints called over HTTP.

## Stack at a glance

| Concern        | Choice                                                      |
|----------------|-------------------------------------------------------------|
| UI / web       | Flask, vanilla JS, SSE streaming, blue/white theme          |
| Embeddings     | OpenAI-compatible HTTP endpoint (`/v1/embeddings`), bearer-token auth |
| LLM            | OpenAI-compatible HTTP endpoint (`/v1/chat/completions`), bearer-token auth |
| Vector index   | FAISS (`IndexFlatIP`, cosine via normalized vectors)        |
| Keyword index  | `rank_bm25` BM25Okapi                                       |
| Fusion         | Reciprocal Rank Fusion (no extra deps)                      |
| Docs           | PDF (PyMuPDF), DOCX (python-docx), PPTX (python-pptx), XLSX (openpyxl), TXT/MD |
| Persistence    | Plain files under `data/` — no database                     |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env: set LLM_BASE_URL / LLM_MODEL / LLM_API_KEY
#          and EMBEDDING_BASE_URL / EMBEDDING_MODEL / EMBEDDING_API_KEY

python app.py
# open http://127.0.0.1:5000
```

> **Python version note**: FAISS wheels may not yet support Python 3.14.
> If install fails, use Python 3.11 or 3.12 (`brew install python@3.12`).

## How it works

1. **Upload** (UI) or **bulk-ingest** (script). The file is parsed format-by-format
   into `Block`s carrying provenance (page, slide, heading path, sheet/rows).
2. **Chunk** the block stream into ~600-token chunks with 80-token overlap.
   Page and slide boundaries are respected; heading paths are prepended to the
   embedded text. Overlap snaps to paragraph → sentence → word boundaries so
   chunks never start mid-word.
3. **Embed** chunks via `POST {EMBEDDING_BASE_URL}/v1/embeddings` (OpenAI shape:
   `model`, `input`, `encoding_format`, `input_type`). Vectors are L2-normalized
   and appended to a FAISS `IndexFlatIP`. Chunk text and metadata persist to
   `data/index/chunks.json`.
4. **BM25** index is rebuilt over the full corpus and pickled to `bm25.pkl`.
5. **Query**: if there's conversation history, the LLM rewrites the question as a
   standalone query. Vector and BM25 searches run in parallel; results are fused
   with RRF; metadata filters (source_type / app_code / tags / version policy)
   are applied to the candidate set; the top N chunks are formatted with
   `[1]`, `[2]`, … and sent to the LLM.
6. **Answer** streams back via SSE. The LLM endpoint speaks OpenAI-compatible
   server-sent events (`data: {...}\n\n`, terminated by `data: [DONE]`). The UI
   parses `[n]` markers into clickable citations linked to a sources panel.

## File / Doc / Chunk model

The store is **decoupled**, so the same physical file can carry multiple
metadata "views" without being re-embedded:

```
FileRecord   one per unique file content (SHA-keyed)
  ▲
  │ many
DocRecord    one per (file + metadata combo); references file_id
  ▲
  │ many
ChunkRecord  one per chunk; references file_id (not doc_id)
```

Ingesting the same PDF under two different `(source_type, app_code, version,
functionality)` combos creates two `DocRecord`s but reuses the existing chunks
— second ingest does **no** parsing or embedding. Deleting one view removes only
its `DocRecord`; the chunks and `FileRecord` live as long as any sibling view
references them.

## Metadata-driven filtering

Each `DocRecord` carries:

- `source_type` — `manual | spec | other`
- `app_code` — declared in `config.APP_CODES`
- `version` — e.g. `v1`, `v2`
- `functionality` — free-form (e.g. `login`, `mfa`, `token`)
- `tags` — free-form list
- `managed` — `True` for bulk-ingested (script-owned), `False` for UI uploads

Filter semantics (`rag/retriever.py`):

- **Empty value on the doc = universal** (passes any filter for that field).
- **Non-empty must match** the user's choice exactly.
- **Version policy**: when the filter has an `app_code`, `APP_VERSION_MAP[app_code]`
  decides the effective version per functionality. Manuals don't need to be
  stamped with `app_code` — the policy is enough.
- Policy values can be a string (`"v2"`) or a list (`["v1","v2"]`) when a
  functionality genuinely supports both.

### Config knobs

```python
# config.py
APP_CODES = ["auth-svc", "billing-svc"]

APP_VERSION_MAP = {
    "auth-svc":    {"login": "v2", "token": "v1", "mfa": ["v1","v2"], "*": "v2"},
    "billing-svc": {"*": "v1"},
}

DOCX_STOP_HEADINGS = [
    "Appendix",
    "References",
    "Revision History",
]
```

`DOCX_STOP_HEADINGS` is a case-insensitive substring list against
heading-styled paragraphs. The matching heading and everything after it is
dropped before chunking — useful for trimming appendices, indexes, and
boilerplate.

## Bulk ingest script

`scripts/ingest_corpus.py` is the canonical path for adding curated content
(manuals, specs). Three modes:

| Mode | Folder shape | Default `source_type` |
|---|---|---|
| `--by-filename` | flat: `<func>_<ver>_<anything>.ext` | `manual` |
| `--by-app-path` | `<app_code>/[<v?>/]/file.ext` | `spec` |
| `--by-path` | `<src_type>/<app_code>/<ver>/<func>/file.ext` | inferred from level 0 |

Typical layout:

```
data/corpus/
├── manual/                              ← --by-filename
│   ├── ca_v1_functional.pdf
│   ├── login_v2_functional.pdf
│   └── token_v1_functional.pdf
└── spec/                                ← --by-app-path
    ├── auth-svc/
    │   ├── api-spec.pdf
    │   └── data-model.pdf
    └── billing-svc/
        └── payment-spec.pdf
```

Run (with Flask stopped — the store isn't multi-writer-safe):

```bash
python scripts/ingest_corpus.py data/corpus/manual --by-filename
python scripts/ingest_corpus.py data/corpus/spec --by-app-path
```

Bulk-ingested docs are marked `managed=True`. They:

- Don't appear in the sidebar doc list (a summary `1,247 managed docs in index`
  is shown instead).
- Can't be deleted from the UI (the `/docs/<id>` endpoint returns 403 for
  managed docs, defense-in-depth beyond the UI affordance).

To replace a managed doc: edit the source file, wipe `data/index/*`, re-run the
script. SHA dedup makes re-runs cheap (already-indexed files are skipped).

## UI uploads vs bulk ingest

The UI's upload form is a deliberate scratch space:

- Always sets `source_type="other"` (the server hardcodes this — the dropdown
  is intentionally absent from the form).
- Always `managed=False`, so users can delete what they uploaded.
- Can carry `app_code` and `tags` so a personal note can be scoped.
- Empty `version` and `functionality` → universal at query time, so the note
  surfaces unless the user filters by Manual/Spec (which exclude `other`).

Curated manuals and specs only enter through `scripts/ingest_corpus.py`.

## Files of note

- `app.py` — Flask routes (upload SSE, chat SSE, docs, delete).
- `config.py` — single source of truth for app codes, version policy, stop headings.
- `rag/parsers.py` — one parser per format → normalized `Block`. DOCX walks body
  in document order with stop-heading support. PDF uses PyMuPDF with
  repeating-header/footer detection.
- `rag/chunker.py` — token-aware splitter; hierarchical-boundary overlap.
- `rag/embeddings.py` — OpenAI-compatible HTTP embedder, batched + asymmetric
  (`input_type=passage` at ingest, `input_type=query` at search).
- `rag/llm/__init__.py` — OpenAI-compatible HTTP LLM, SSE streaming.
- `rag/vector_store.py` — `FileRecord` / `DocRecord` / `ChunkRecord` + FAISS
  with atomic persistence and in-memory lookup indices.
- `rag/bm25_store.py` — pickled BM25.
- `rag/retriever.py` — parallel hybrid search, RRF, metadata-aware filtering
  with auto-resolved version policy.
- `rag/ingest.py` — orchestrates parse → chunk → embed → persist; emits per-batch
  progress events; reuses chunks when a file is re-ingested under new metadata.
- `rag/chat.py` — multi-turn chat with query rewrite.
- `rag/citations.py` — context block builder with filename / page / heading
  locators.
- `scripts/ingest_corpus.py` — bulk-ingest CLI; three layout modes; idempotent.

## Configuration reference

`.env`:

```ini
# LLM
LLM_BASE_URL=http://localhost:11434       # or your OpenAI-compatible server
LLM_MODEL=llama3.1
LLM_API_KEY=                              # bearer token; leave empty if endpoint has no auth

# Embeddings (defaults to LLM_BASE_URL / LLM_API_KEY if unset)
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=bge-m3
EMBEDDING_API_KEY=

# Chunking
CHUNK_TOKENS=600
CHUNK_OVERLAP=80

# Retrieval
TOP_K_VECTOR=20
TOP_K_BM25=20
TOP_N_FINAL=6
CONTEXT_TOKEN_BUDGET=3000

# Flask
FLASK_SECRET_KEY=change-me
MAX_UPLOAD_MB=50
```

`config.py` (not env-driven — edit code):

- `APP_CODES` — list of valid app codes for the upload and filter dropdowns.
- `APP_VERSION_MAP` — per-app version policy by functionality.
- `DOCX_STOP_HEADINGS` — list of substring keywords; matching heading and
  everything after is dropped during DOCX parsing.

## Tests

```bash
pytest -q
```

`test_retriever.py` requires FAISS and uses a stub embedder; no embedding
endpoint hit. `test_chunker.py` and `test_parsers.py` have no external
dependencies beyond the format-specific parser libraries.
