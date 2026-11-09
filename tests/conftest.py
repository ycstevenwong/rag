"""Shared test setup: isolated data directories and stub model endpoints.

`app.py` builds an app at import time and its startup cleanup deletes old files
in the uploads directory, so config must point at a temp directory before
anything imports it. Nothing here talks to a real embedding or LLM endpoint.
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

import config
from rag import config_facade

_PATH_NAMES = ("VECTORS_PATH", "CHUNKS_PATH", "DOCS_PATH", "FILES_PATH")


def _data_paths(root: Path) -> dict[str, Path]:
    index = root / "index"
    paths = {
        "DATA_DIR": root,
        "UPLOAD_DIR": root / "uploads",
        "INDEX_DIR": index,
        "SESSION_DIR": root / "sessions",
        "PENDING_DIR": root / "pending",
        "VECTORS_PATH": index / "vectors.faiss",
        "CHUNKS_PATH": index / "chunks.json",
        "DOCS_PATH": index / "docs.json",
        "FILES_PATH": index / "files.json",
    }
    for name in ("UPLOAD_DIR", "INDEX_DIR", "SESSION_DIR", "PENDING_DIR"):
        paths[name].mkdir(parents=True, exist_ok=True)
    return paths


# Redirect before any test module can import app.py.
_session_root = Path(tempfile.mkdtemp(prefix="rag-tests-"))
for _name, _value in _data_paths(_session_root).items():
    setattr(config, _name, _value)
config_facade.SESSION_DIR = config.SESSION_DIR
for _name in _PATH_NAMES:
    setattr(config_facade, _name, getattr(config, _name))
# A developer's local .env must not change what importing app.py does.
config.ADMIN_USERNAME = ""
config.ADMIN_PASSWORD_HASH = ""
# Tests opt in to app rules, so the real ones don't constrain unrelated tests.
REAL_APP_CODES = config.APP_CODES
REAL_APP_VERSION_MAP = config.APP_VERSION_MAP
REAL_FUNCTIONALITIES = getattr(config, "FUNCTIONALITIES", None)
REAL_VERSIONS = getattr(config, "VERSIONS", None)
config.FUNCTIONALITIES = []
config.VERSIONS = []
config.APP_CODES = []
config.APP_VERSION_MAP = {}
config_facade.APP_VERSION_MAP = {}


class StubEmbedder:
    """Deterministic hashed bag-of-words embedder with the real Embedder's API."""

    dim = 64

    def __init__(self, *_args, **_kwargs):
        pass

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for word in text.lower().split():
            h = 0
            for ch in word:
                h = (h * 131 + ord(ch)) & 0xFFFFFFFF
            v[h % self.dim] += 1
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def encode_query(self, text: str) -> np.ndarray:
        return self._vec(text)

    def encode_batched(self, texts, *, batch_size: int = 32):
        total = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch = np.stack([self._vec(t) for t in texts[i : i + batch_size]])
            yield (i // batch_size + 1, total, batch)


class StubLLM:
    """Records what it was asked and replies with a fixed answer."""

    reply = "Stub answer [1]."

    def __init__(self, *_args, **_kwargs):
        self.stream_calls: list[list[dict]] = []
        self.complete_calls: list[str] = []

    def stream_chat(self, _system: str, messages: list[dict]):
        self.stream_calls.append(messages)
        yield self.reply

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        self.complete_calls.append(prompt)
        return "standalone question"


@pytest.fixture
def data_dirs(tmp_path, monkeypatch):
    """Point config at a fresh directory for this test."""
    paths = _data_paths(tmp_path / "data")
    for name, value in paths.items():
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(config_facade, "SESSION_DIR", paths["SESSION_DIR"])
    return paths


@pytest.fixture
def app_module(data_dirs, monkeypatch):
    """The app module with stub embedding and LLM endpoints."""
    import app as module
    import rag.chat

    monkeypatch.setattr(module, "Embedder", StubEmbedder)
    monkeypatch.setattr(rag.chat, "LLMProvider", StubLLM)
    return module


@pytest.fixture
def flask_app(app_module):
    return app_module.create_app()


@pytest.fixture
def client(flask_app):
    return flask_app.test_client()


def docx_bytes(text: str) -> bytes:
    import docx

    document = docx.Document()
    for paragraph in text.split("\n\n"):
        document.add_paragraph(paragraph)
    buffer = io.BytesIO()
    document.save(buffer)
    return buffer.getvalue()


def pdf_bytes(text: str) -> bytes:
    import pymupdf

    document = pymupdf.open()
    document.new_page().insert_textbox(pymupdf.Rect(72, 72, 520, 760), text, fontsize=11)
    return document.tobytes()
