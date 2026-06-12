"""Embedder that calls an Ollama HTTP endpoint."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import requests


class Embedder:
    def __init__(self, model_name: str, base_url: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.encode(["probe"]).shape[1]
        return self._dim

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        vectors = []
        for text in texts:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)

    def encode_query(self, text: str) -> np.ndarray:
        if "bge" in self.model_name.lower():
            text = "Represent this sentence for searching relevant passages: " + text
        return self.encode([text])[0]
