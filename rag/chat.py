"""Multi-turn chat orchestrator: rewrite → retrieve → prompt → stream."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

from . import config_facade as cfg
from .citations import build_context_block
from .llm import LLMProvider
from .retriever import HybridRetriever

SYSTEM_PROMPT = """You answer technical questions using ONLY the provided context.

Rules:
- Cite each factual claim with the bracket reference of the source it came from, e.g. [1] or [2][3].
- If the context does not contain enough information to answer, say so plainly. Do not invent details.
- Prefer precise technical language; quote exact identifiers (function names, error codes, parameters) when relevant.
- When relevant, structure answers with short bullet lists or fenced code blocks.
"""

REWRITE_PROMPT = """Given the conversation history and a follow-up message, rewrite the follow-up as a standalone question that can be understood without the history. Output only the rewritten question, no preamble.

History:
{history}

Follow-up: {question}

Standalone question:"""


class ChatService:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self._provider: LLMProvider | None = None

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = LLMProvider(cfg.LLM_MODEL, cfg.LLM_BASE_URL, cfg.LLM_API_KEY)
        return self._provider

    def chat_stream(
        self,
        session_id: str,
        question: str,
    ) -> Iterator[dict]:
        """Yield events: {"type": "sources", "sources": [...]} then
        {"type": "delta", "text": "..."} chunks, then {"type": "done"}.
        """
        history = _load_session(session_id)
        provider = self._get_provider()

        standalone_query = question
        if history:
            standalone_query = self._rewrite_query(provider, history, question)

        retrieved = self.retriever.search(
            standalone_query,
            k_vector=cfg.TOP_K_VECTOR,
            k_bm25=cfg.TOP_K_BM25,
            top_n=cfg.TOP_N_FINAL,
        )

        context_text, sources = build_context_block(
            retrieved, token_budget=cfg.CONTEXT_TOKEN_BUDGET
        )

        yield {"type": "sources", "sources": sources, "rewritten_query": standalone_query}

        user_message = (
            f"Context:\n{context_text}\n\nQuestion: {question}"
            if context_text
            else f"No retrieved context is available. Answer based only on what the user has said.\n\nQuestion: {question}"
        )

        messages = [*_trim_history(history, max_turns=8), {"role": "user", "content": user_message}]

        full_reply: list[str] = []
        try:
            for delta in provider.stream_chat(SYSTEM_PROMPT, messages):
                full_reply.append(delta)
                yield {"type": "delta", "text": delta}
        except Exception as exc:
            yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
            return

        assistant_text = "".join(full_reply)
        history.append({"role": "user", "content": question, "ts": time.time()})
        history.append({
            "role": "assistant",
            "content": assistant_text,
            "ts": time.time(),
            "sources": sources,
            "rewritten_query": standalone_query if standalone_query != question else None,
        })
        _save_session(session_id, history)
        yield {"type": "done"}

    def _rewrite_query(self, provider: LLMProvider, history: list[dict], question: str) -> str:
        recent = history[-4:]
        history_text = "\n".join(f"{m['role'].title()}: {m['content']}" for m in recent)
        prompt = REWRITE_PROMPT.format(history=history_text, question=question)
        try:
            return provider.complete(prompt, max_tokens=128).strip() or question
        except Exception as exc:
            print(f"[rewrite_query] fallback to original question: {type(exc).__name__}: {exc}", flush=True)
            return question


def _session_path(session_id: str) -> Path:
    safe = "".join(c for c in session_id if c.isalnum() or c in "-_")[:64] or "default"
    return cfg.SESSION_DIR / f"{safe}.json"


def _load_session(session_id: str) -> list[dict]:
    p = _session_path(session_id)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_session(session_id: str, history: list[dict]) -> None:
    p = _session_path(session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def _trim_history(history: list[dict], max_turns: int) -> list[dict]:
    trimmed = history[-(max_turns * 2):]
    return [{"role": m["role"], "content": m["content"]} for m in trimmed]
