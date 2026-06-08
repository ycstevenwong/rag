"""Thin wrapper around sentence-transformers with lazy model loading."""
from __future__ import annotations

from threading import Lock
from typing import Sequence

import numpy as np


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = Lock()

    def _load(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        return self._load().get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return vectors.astype(np.float32, copy=False)

    def encode_query(self, text: str) -> np.ndarray:
        # BGE models recommend a query instruction for retrieval.
        if "bge" in self.model_name.lower():
            text = "Represent this sentence for searching relevant passages: " + text
        return self.encode([text])[0]
