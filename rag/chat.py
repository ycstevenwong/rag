"""Multi-turn chat orchestrator: rewrite → retrieve → prompt → stream."""
from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Iterator

from . import config_facade as cfg
from .chunker import count_tokens
from .citations import build_context_block
from .llm import LLMProvider
from .retriever import HybridRetriever
from .vector_store import atomic_write_text

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

NO_SOURCES_REPLY = "I couldn't find anything about that in the documents for this app."

_CITATION_RE = re.compile(r"[ \t]*(?:\[\d+(?:,\s*\d+)*\])+")
_CODE_SPAN_RE = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")

_session_lock = threading.RLock()


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
        filters: dict | None = None,
    ) -> Iterator[dict]:
        """Yield events: {"type": "sources", "sources": [...]} then
        {"type": "delta", "text": "..."} chunks, then {"type": "done"}.
        """
        try:
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
                filters=filters,
            )

            context_text, sources = build_context_block(
                retrieved, token_budget=cfg.CONTEXT_TOKEN_BUDGET
            )
        except Exception as exc:
            yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
            return

        yield {"type": "sources", "sources": sources, "rewritten_query": standalone_query}

        full_reply: list[str] = []
        if not sources:
            # Nothing to answer from. The model would otherwise answer from its own
            # knowledge, which could contradict the documents for this app.
            full_reply.append(NO_SOURCES_REPLY)
            yield {"type": "delta", "text": NO_SOURCES_REPLY}
        else:
            user_message = f"Context:\n{context_text}\n\nQuestion: {question}"
            history_budget = (
                cfg.CONTEXT_TOKEN_BUDGET - count_tokens(SYSTEM_PROMPT) - count_tokens(user_message)
            )
            messages = [
                *_trim_history(history, max_turns=8, token_budget=history_budget),
                {"role": "user", "content": user_message},
            ]
            try:
                for delta in provider.stream_chat(SYSTEM_PROMPT, messages):
                    full_reply.append(delta)
                    yield {"type": "delta", "text": delta}
            except Exception as exc:
                yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
                return

        assistant_text = "".join(full_reply)
        try:
            _append_to_session(session_id, [
                {"role": "user", "content": question, "ts": time.time()},
                {
                    "role": "assistant",
                    "content": assistant_text,
                    "ts": time.time(),
                    "sources": sources,
                    "rewritten_query": standalone_query if standalone_query != question else None,
                },
            ])
        except OSError as exc:
            yield {"type": "error", "error": f"Answer not saved to history: {exc}"}
            return
        yield {"type": "done"}

    def _rewrite_query(self, provider: LLMProvider, history: list[dict], question: str) -> str:
        recent = history[-4:]
        history_text = "\n".join(
            f"{m['role'].title()}: {_strip_citations(m['content'])}" for m in recent
        )
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
    with _session_lock:
        if not p.exists():
            return []
        try:
            history = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            history = None
        if isinstance(history, list):
            return history
        # Keep an unreadable file, or the next save would destroy what's left of it.
        kept = p.with_name(f"{p.name}.corrupt-{time.time_ns()}")
        p.replace(kept)
        print(f"[session] unreadable history moved to {kept.name}", flush=True)
        return []


def _append_to_session(session_id: str, messages: list[dict]) -> None:
    """Add a finished turn to the file's current contents, not to the copy loaded
    when the turn began, so concurrent turns in one session don't drop each other."""
    p = _session_path(session_id)
    with _session_lock:
        history = _load_session(session_id)
        history.extend(messages)
        atomic_write_text(p, json.dumps(history, ensure_ascii=False, indent=2))


def _strip_citations(text: str) -> str:
    """Remove [n] markers, leaving code spans alone. An earlier answer's numbers
    refer to that turn's sources and would collide with the current turn's."""
    parts = _CODE_SPAN_RE.split(text)
    # re.split with a capture group puts the code spans at the odd indexes.
    return "".join(
        part if i % 2 else _CITATION_RE.sub("", part) for i, part in enumerate(parts)
    )


def _trim_history(history: list[dict], max_turns: int, token_budget: int) -> list[dict]:
    """The most recent whole turns that fit in the budget, oldest first. The current
    question and its context have already been charged, so history gets what's left."""
    recent = [
        {"role": m["role"], "content": _strip_citations(m["content"])}
        for m in history[-(max_turns * 2):]
    ]
    kept: list[dict] = []
    used = 0
    for end in range(len(recent), 0, -2):
        turn = recent[max(0, end - 2) : end]
        cost = sum(count_tokens(m["content"]) for m in turn)
        if used + cost > token_budget:
            break
        kept = turn + kept
        used += cost
    return kept
