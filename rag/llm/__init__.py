"""LLM provider that calls an OpenAI-compatible /v1/chat/completions endpoint."""
from __future__ import annotations

import json
from typing import Iterator

import requests


class LLMProvider:
    def __init__(self, model: str, base_url: str, api_key: str):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def stream_chat(self, system: str, messages: list[dict]) -> Iterator[str]:
        full_messages = [{"role": "system", "content": system}, *messages]
        with requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json={"model": self.model, "messages": full_messages, "stream": True},
            stream=True,
            timeout=300,
        ) as resp:
            resp.raise_for_status()
            # Split on raw bytes and decode each line as UTF-8. With no charset on
            # text/event-stream, requests would decode as Latin-1, which garbles
            # the text and turns byte 0x85 into a line break.
            for raw_line in resp.iter_lines():
                line = raw_line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                # Some servers send chunks with no choices, such as usage totals.
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
