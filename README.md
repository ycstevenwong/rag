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
(manuals, specs). It walks a directory tree, parses each supported file
(`.pdf .docx .pptx .xlsx .txt .md`), and pushes the chunks through the same
pipeline the UI uploader uses — but with full metadata derived from the folder
or filename convention, and the `managed=True` flag so the docs are locked
from UI deletion.

### Three layout modes (mutually exclusive)

| Mode | Folder shape | Default `source_type` | Use for |
|---|---|---|---|
| `--by-filename` | flat: `<func>_<ver>_<anything>.ext` | `manual` | manuals (version-policy driven) |
| `--by-app-path` | `<app_code>/[<v>/]/file.ext` | `spec` | app-owned docs (specs, policies) |
| `--by-path` | `<src_type>/<app_code>/<ver>/<func>/file.ext` | inferred from level 0 | fully-explicit, general purpose |

### Typical layout

```
data/corpus/
├── manual/                              ← --by-filename
│   ├── ca_v1_functional.pdf
│   ├── ca_v2_functional.pdf
│   ├── login_v2_functional.pdf
│   ├── mfa_v2_functional.pdf
│   └── token_v1_functional.pdf
└── spec/                                ← --by-app-path
    ├── auth-svc/
    │   ├── api-spec.pdf
    │   └── data-model.pdf
    └── billing-svc/
        ├── payment-spec.pdf
        └── refund-spec.pdf
```

### CLI

```
python scripts/ingest_corpus.py [path] [mode-flag] [overrides...]
```

Positional:
- `path` — file or directory. Defaults to `data/corpus/`.

Mode (pick one):
- `--by-filename` — manuals convention; parse `<func>_<ver>_*.ext`. `<ver>` must
  match `v\d+`; trailing segments are ignored.
- `--by-app-path` — specs convention; level 0 under target = `app_code`,
  optional level 1 (matching `v\d+`) = `version`.
- `--by-path` — fully explicit; each level of `<src_type>/<app_code>/<ver>/<func>/`
  becomes the corresponding field.

Per-file overrides (apply when no path/filename info covers the field):
- `--source-type <s>` — `manual | spec | other` (mode-dependent default).
- `--app-code <code>` — must match a value in `config.APP_CODES` to be
  useful for filtering; warns otherwise.
- `--version <v>` — e.g. `v1`, `v2`.
- `--functionality <f>` — e.g. `login`, `mfa`.
- `--tags <a,b,c>` — comma-separated, applied to every file in the run.

### Run (Flask must be stopped — the store isn't multi-writer-safe)

```bash
# Wipe first if you want to re-index from scratch
rm -rf data/index/*

# Manuals
python scripts/ingest_corpus.py data/corpus/manual --by-filename

# Specs
python scripts/ingest_corpus.py data/corpus/spec --by-app-path
```

Single file with explicit metadata:

```bash
python scripts/ingest_corpus.py path/to/file.pdf \
    --source-type manual --app-code auth-svc \
    --version v2 --functionality login
```

### Expected output

```
Ingesting 5 document(s) from /Users/.../data/corpus/manual

[1/5] ca_v1_functional.pdf
  source_type=manual app_code=- version=v1 functionality=ca
  Parsing...
  Chunking...
  Embedding batch 4/4
  Indexed 27 chunks
[2/5] ca_v2_functional.pdf
  source_type=manual app_code=- version=v2 functionality=ca
  ...

Done. Added: 5  Duplicate: 0  Errors: 0
Total docs in index: 5
```

Warnings (don't abort the run) appear when inferred values fall outside the
known sets:

```
[3/5] manaul/auth-svc/v2/api.pdf
  WARNING: inferred source_type 'manaul' not in ['manual', 'other', 'spec']
[4/5] auth-svc/api.pdf
  WARNING: inferred app_code 'auth-svc' not in config.APP_CODES
```

Exit codes:
- `0` — every file succeeded (or was an idempotent skip).
- `1` — bad path, empty target.
- `2` — at least one file errored, others may have succeeded.

### Idempotency and re-ingest

Files are deduplicated by SHA-256. Re-running the same command:

- New files → ingested fresh.
- Same file + same metadata combo → reported as `Already indexed -> skipped`.
- Same file content + new metadata combo (e.g. shared with a second app_code
  via a different folder) → no re-parse / re-embed; just a new `DocRecord`
  pointing at the existing chunks. Output shows `Linking existing file...`.

### Managed flag

Every doc produced by the script has `managed=True`. They:

- **Don't appear in the sidebar doc list** — a summary
  `1,247 managed docs in index` shows above the user's own uploads.
- **Can't be deleted from the UI** — the `/docs/<id>` endpoint returns 403 for
  managed docs (defense in depth beyond just hiding the button).

To replace a managed doc: edit the source file on disk, wipe `data/index/*`,
re-run the script. Or to remove one without rebuilding, delete its source file
and re-run — the missing file simply isn't re-ingested.

### Bringing the changes online

```bash
python app.py
# open http://127.0.0.1:5000
```

In the sidebar **Filter retrieval**:
- `App code = auth-svc` + a manual question → `APP_VERSION_MAP` picks the
  correct version per functionality.
- `App code = auth-svc` + `Source type = Spec` → only auth-svc's specs.
- Leave filters blank → all corpus content is fair game.

## Eval set and retrieval metrics

`scripts/run_eval.py` runs a JSON-defined set of test questions through the
retriever and reports `Recall@N`, `MRR`, and `median hit rank` — the
measurement layer for every retrieval-quality experiment (chunking,
embeddings, fusion weights, MMR tuning, etc.). Without numbers from a
fixed eval set, every change is a guess.

### Eval set format

`data/eval/eval.json` ships as a template with an embedded `_schema` block
and four example entries. Replace the placeholders with ~20–30 real
questions from your corpus. Each entry:

```json
{
  "id": "q07",
  "question": "How do I refund a payment?",
  "filters": {"app_code": "billing-svc", "source_type": "manual"},
  "expected_doc_ids": [],
  "expected_filenames": ["refund"],
  "expected_keywords": ["refund", "REVERSE-TRANS"],
  "notes": "Free-form context for the human author."
}
```

The three `expected_*` fields are **ORed** — the first retrieved chunk in
the top-N that satisfies any one of them sets the hit rank for that
question. Pick whichever level is convenient to author:

| Field | When to use | Match semantics |
|---|---|---|
| `expected_doc_ids` | You know the exact `DocRecord.doc_id` | Exact match |
| `expected_filenames` | You know which file should be cited | Case-insensitive substring |
| `expected_keywords` | Any chunk mentioning the phrase counts | Case-insensitive substring in chunk text |

Loose `expected_keywords` can produce false positives (any chunk with the
word "refund" counts as a hit); use distinctive phrases or identifiers
where possible.

### Run

```bash
# Stop Flask first (the eval script loads the FAISS / BM25 store
# in-process; concurrent persists from a running app risk corruption).

# Baseline summary — uses data/eval/eval.json, top-N=10
python scripts/run_eval.py

# Per-question rank, not just failures
python scripts/run_eval.py --verbose

# Full per-question dump: top-N retrieved with filename, score, and snippet,
# with a star next to the rows that satisfied any expected_* field.
python scripts/run_eval.py --detailed

# Tighter recall threshold
python scripts/run_eval.py --top-n 5

# Different eval file
python scripts/run_eval.py path/to/other-eval.json
```

### Sample output

```
============================================================
Results
============================================================
  Questions:       30
  Hits @ 10        24
  Recall @ 10      80.0%
  MRR:             0.61
  Median hit rank: 2

============================================================
Failures (6)
============================================================
  [q03-procedural-refund] How do I refund a payment?
     1. token-overview.pdf
     2. login-manual.pdf
     3. password-reset.pdf
```

Exit codes: `0` on `Recall@N ≥ 50%`, `2` otherwise (useful for CI gating).

### What the metrics mean

| Metric | Question it answers |
|---|---|
| **Hits @ N** | "Did the right chunk show up at all in the top-N?" Binary per question. |
| **Recall @ N** | Same as Hits, as a percentage of total questions. |
| **MRR** | "When the right chunk *is* found, how high is it?" Penalizes burying the answer at rank 10. |
| **Median hit rank** | "Typical position of the right chunk when it's found." Complementary to MRR. |

Track **Recall @ N** as the headline number; **MRR** as the secondary check
that ranking didn't regress. A healthy change moves both in the same
direction.

### Workflow

1. Author 20–30 real questions in `data/eval/eval.json`.
2. Stop Flask, run `python scripts/run_eval.py`, record the baseline numbers.
3. Make one change at a time (chunk size, MMR lambda, model swap, etc.).
4. Re-run the script — accept the change if metrics go up, revert if they
   go down.
5. Use `--detailed` to diagnose specific weak hits and failures.

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
- `scripts/run_eval.py` — runs a JSON eval set through the retriever; reports
  Recall@N / MRR / median hit rank; `--detailed` dumps per-question chunks.
- `data/eval/eval.json` — eval set template (4 placeholder entries); replace
  with real questions to establish a measurable baseline.

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
