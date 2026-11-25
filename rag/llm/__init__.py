"""LLM provider that calls an OpenAI-compatible /v1/chat/completions endpoint."""
from __future__ import annotations

import json
from typing import Callable, Iterator

import requests

# An answer that kept repeating itself ran for hours with no limit. Normal answers
# are under 500 tokens.
ANSWER_MAX_TOKENS = 1500


class LLMProvider:
    def __init__(self, model: str, base_url: str, api_key: str, temperature: float = 0.0):
        self.model = model
        # At a model's default temperature the same question can get an answer one
        # time and "couldn't find it" the next.
        self.temperature = temperature
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.api_key}"}

    def stream_chat(
        self,
        system: str,
        messages: list[dict],
        on_finish: Callable[[str], None] | None = None,
    ) -> Iterator[str]:
        """Yield the answer's text as it arrives. on_finish gets the reason the
        model stopped, such as "stop" or "length", when the server sends one."""
        full_messages = [{"role": "system", "content": system}, *messages]
        with requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": full_messages,
                "stream": True,
                "temperature": self.temperature,
                "max_tokens": ANSWER_MAX_TOKENS,
            },
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
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield content
                finish_reason = choices[0].get("finish_reason")
                if finish_reason and on_finish:
                    on_finish(finish_reason)

    def complete(self, prompt: str, max_tokens: int = 256) -> str:
        resp = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers=self._headers(),
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "stream": False,
                "temperature": self.temperature,
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
