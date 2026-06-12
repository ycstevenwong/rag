"""Self-hosted LLM provider that calls an HTTP endpoint via requests.post."""
from __future__ import annotations

import json
from typing import Iterator

import requests


class LLMProvider:
    def __init__(self, model: str, base_url: str):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def stream_chat(self, system: str, messages: list[dict]) -> Iterator[str]:
        full_messages = [{"role": "system", "content": system}, *messages]
        with requests.post(
            f"{self.base_url}/api/chat",
            json={"model": self.model, "messages": full_messages, "stream": True},
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content")
                if content:
                    yield content
                if chunk.get("done"):
                    break

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_tokens},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")
