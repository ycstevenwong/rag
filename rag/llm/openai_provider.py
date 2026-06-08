from __future__ import annotations

import os
from typing import Iterator

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str):
        import openai
        self.model = model
        self._client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def stream_chat(self, system: str, messages: list[dict]) -> Iterator[str]:
        full_messages = [{"role": "system", "content": system}, *messages]
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=full_messages,
            stream=True,
            max_tokens=2048,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""
