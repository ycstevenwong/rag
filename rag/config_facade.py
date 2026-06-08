"""Re-export of root config to avoid sys.path gymnastics inside the package."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_cfg = importlib.import_module("config")

CHUNK_TOKENS = _cfg.CHUNK_TOKENS
CHUNK_OVERLAP = _cfg.CHUNK_OVERLAP
TOP_K_VECTOR = _cfg.TOP_K_VECTOR
TOP_K_BM25 = _cfg.TOP_K_BM25
TOP_N_FINAL = _cfg.TOP_N_FINAL
CONTEXT_TOKEN_BUDGET = _cfg.CONTEXT_TOKEN_BUDGET
EMBEDDING_MODEL = _cfg.EMBEDDING_MODEL
LLM_PROVIDER = _cfg.LLM_PROVIDER
LLM_MODEL = _cfg.LLM_MODEL
VECTORS_PATH = _cfg.VECTORS_PATH
CHUNKS_PATH = _cfg.CHUNKS_PATH
DOCS_PATH = _cfg.DOCS_PATH
BM25_PATH = _cfg.BM25_PATH
UPLOAD_DIR = _cfg.UPLOAD_DIR
SESSION_DIR = _cfg.SESSION_DIR
MAX_UPLOAD_MB = _cfg.MAX_UPLOAD_MB
FLASK_SECRET_KEY = _cfg.FLASK_SECRET_KEY
