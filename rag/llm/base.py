"""Abstract LLM provider interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator


class LLMProvider(ABC):
    @abstractmethod
    def stream_chat(self, system: str, messages: list[dict]) -> Iterator[str]:
        """Stream assistant text tokens.

        messages is a list of {"role": "user" | "assistant", "content": str}.
        """

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        """One-shot completion used for query rewriting."""
