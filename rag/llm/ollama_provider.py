from __future__ import annotations

import os
from typing import Iterator

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model: str):
        import ollama
        self.model = model
        host = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._client = ollama.Client(host=host)

    def stream_chat(self, system: str, messages: list[dict]) -> Iterator[str]:
        full_messages = [{"role": "system", "content": system}, *messages]
        stream = self._client.chat(model=self.model, messages=full_messages, stream=True)
        for chunk in stream:
            content = chunk.get("message", {}).get("content")
            if content:
                yield content

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        resp = self._client.generate(
            model=self.model,
            prompt=prompt,
            options={"num_predict": max_tokens},
            stream=False,
        )
        return resp.get("response", "")
