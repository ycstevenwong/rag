"""Embedder that calls an OpenAI-compatible HTTP embeddings endpoint."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import requests


class Embedder:
    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = self.encode(["probe"]).shape[1]
        return self._dim

    def _embed(self, texts: Sequence[str], input_type: str) -> np.ndarray:
        resp = requests.post(
            f"{self.base_url}/v1/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model_name,
                "input": list(texts),
                "encoding_format": "float",
                "input_type": input_type,
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = sorted(resp.json()["data"], key=lambda x: x["index"])
        arr = np.array([d["embedding"] for d in data], dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-10)

    def encode(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        out = []
        for i in range(0, len(texts), batch_size):
            out.append(self._embed(texts[i : i + batch_size], input_type="passage"))
        return np.vstack(out) if out else np.zeros((0, self.dim), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return self._embed([text], input_type="query")[0]
