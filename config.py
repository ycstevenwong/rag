"""Centralized config loaded from environment variables (.env)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"
SESSION_DIR = DATA_DIR / "sessions"

for d in (UPLOAD_DIR, INDEX_DIR, SESSION_DIR):
    d.mkdir(parents=True, exist_ok=True)

VECTORS_PATH = INDEX_DIR / "vectors.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
DOCS_PATH = INDEX_DIR / "docs.json"
BM25_PATH = INDEX_DIR / "bm25.pkl"

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", LLM_BASE_URL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "")

CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

TOP_K_VECTOR = int(os.getenv("TOP_K_VECTOR", "20"))
TOP_K_BM25 = int(os.getenv("TOP_K_BM25", "20"))
TOP_N_FINAL = int(os.getenv("TOP_N_FINAL", "6"))
CONTEXT_TOKEN_BUDGET = int(os.getenv("CONTEXT_TOKEN_BUDGET", "3000"))

FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

# Edit this list to add or remove the app codes shown in the upload form
# and filter sidebar. Empty string is always accepted as "unspecified".
APP_CODES = [
    # "app-a",
    # "app-b",
]

# Per-app version map for manuals.
# Keys: app_code (must also appear in APP_CODES).
# Inner keys: functionality name (free-form, set at ingest time).
# Inner values: the "current truth" version for that functionality.
# Use "*" as the inner key to set a default for any functionality not
# explicitly listed. When the user picks an app_code at query time,
# the retriever auto-filters manual chunks to the right version per
# functionality using this map.
APP_VERSION_MAP: dict[str, dict[str, str]] = {
    # "auth-svc": {
    #     "login": "v2",
    #     "token": "v1",
    #     "*":     "v2",
    # },
    # "billing-svc": {
    #     "*": "v1",
    # },
}
