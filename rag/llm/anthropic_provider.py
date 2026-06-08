from __future__ import annotations

import os
from typing import Iterator

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def stream_chat(self, system: str, messages: list[dict]) -> Iterator[str]:
        with self._client.messages.stream(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=2048,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in resp.content if hasattr(block, "text"))
