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
from .retriever import Retriever
from .vector_store import atomic_write_text

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources in the context. You have no other knowledge.

Rules:
- State only what a source directly says. Do not use outside knowledge, even for well-known facts.
- If no source directly answers the question, reply only that you couldn't find it in the documents. Never start that reply with "Yes" or "No", and do not cite anything.
- A source not mentioning something is not evidence for or against it.
- After each factual claim, cite the number of a source that directly states it, e.g. [1] or [2][3]. Do not cite a source just because it covers a related topic.
- If a source gives a condition, exception, waiver or limit for the fact asked about, include it. Do not add other facts the question did not ask about.
- Use quotation marks only for wording copied exactly from a source.
- Prefer precise technical language; give exact identifiers (function names, error codes, parameters) when relevant.
- When relevant, structure answers with short bullet lists or fenced code blocks.
"""

REWRITE_PROMPT = """Given the conversation history and a follow-up message, rewrite the follow-up as a standalone question that can be understood without the history. Keep the follow-up's own words and meaning. Only replace the words that point back (such as it, this, they) with what they refer to in the history. If the follow-up already makes sense on its own, return it unchanged. Write it in the same language as the follow-up. Output only the rewritten question, no preamble.

History:
{history}

Follow-up: {question}

Standalone question:"""

NO_SOURCES_REPLY = "I couldn't find anything about that in the documents for this app."

# Sent after each question. Earlier answers reach the model without their [n]
# markers (those numbers meant that turn's sources), so a few turns in it would
# copy them and stop citing. Being the last thing it reads outweighs that.
ANSWER_REMINDER = "Answer from the sources above and cite each fact as [n]."

# Shown, not saved, when an answer is stopped for repeating itself or for reaching
# the length limit. Saved, the model would copy it into later answers.
CUT_SHORT_NOTE = "\n\n(This answer was cut short because it was getting too long.)"

_CITATION_RE = re.compile(r"[ \t]*(?:\[\d+(?:,\s*\d+)*\])+")
_CODE_SPAN_RE = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")

_session_lock = threading.RLock()


class ChatService:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever
        self._provider: LLMProvider | None = None

    def _get_provider(self) -> LLMProvider:
        if self._provider is None:
            self._provider = LLMProvider(
                cfg.LLM_MODEL, cfg.LLM_BASE_URL, cfg.LLM_API_KEY, cfg.LLM_TEMPERATURE
            )
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
            if history and _refers_back(question):
                standalone_query = self._rewrite_query(provider, history, question)

            retrieved = self.retriever.search(
                standalone_query,
                k_vector=cfg.TOP_K_VECTOR,
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
            if ANSWER_REMINDER:
                user_message += f"\n\n{ANSWER_REMINDER}"
            history_budget = (
                cfg.CONTEXT_TOKEN_BUDGET - count_tokens(SYSTEM_PROMPT) - count_tokens(user_message)
            )
            messages = [
                *_trim_history(history, max_turns=8, token_budget=history_budget),
                {"role": "user", "content": user_message},
            ]
            yes_no = _YesNoBeforeDecline()
            repeats = _RepeatGuard()
            finish_reasons: list[str] = []
            stream = provider.stream_chat(SYSTEM_PROMPT, messages, on_finish=finish_reasons.append)
            try:
                for delta in stream:
                    text = yes_no.feed(delta)
                    if text:
                        full_reply.append(text)
                        yield {"type": "delta", "text": text}
                        if repeats.feed(text):
                            break
            except Exception as exc:
                text = yes_no.finish()
                if text:
                    yield {"type": "delta", "text": text}
                yield {"type": "error", "error": f"{type(exc).__name__}: {exc}"}
                return
            finally:
                # Closes the request, so the model stops writing a stopped answer.
                close = getattr(stream, "close", None)
                if close:
                    close()
            text = yes_no.finish()
            if text:
                full_reply.append(text)
                yield {"type": "delta", "text": text}
            if repeats.stopped or "length" in finish_reasons:
                yield {"type": "delta", "text": CUT_SHORT_NOTE}

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


# Words that point back at the conversation, in English, Malay and Indonesian.
_REFERENCE_WORDS = frozenset("""
    it its it's this that these those they them their same above previous earlier former latter
    ia itu ini tersebut mereka dia sama
""".split())
_CONTINUATIONS = (
    "what about", "how about", "and ", "also", "then ",
    "bagaimana dengan", "macam mana dengan", "bagaimana pula", "dan ", "juga",
)
# Malay/Indonesian "-nya" often means "its" ("syaratnya": its conditions).
_NOT_REFERENCE_NYA = frozenset({"hanya", "punya", "tanya", "bertanya"})
_WORD_RE = re.compile(r"[\w'’]+")
# "it" that points forward to a "to ..." clause, as in "How much does it cost to
# replace a card?" or "Is it possible to waive the fee?". Without the clause
# ("How much does it cost?") the "it" still points back.
_FORWARD_IT_RE = re.compile(
    r"(?:\bit\s+(?:costs?|takes?|took)\b"
    r"|\b(?:is|was|will|would)\s+it\s+(?:be\s+)?"
    r"(?:possible|necessary|required|mandatory|allowed|permitted|ok|okay|safe|advisable|worth|important|enough)\b)"
    r"(?=.*\bto\s+\w)"
)


def _refers_back(question: str) -> bool:
    """Whether the question needs the conversation to be understood. One that
    stands on its own is searched as asked: rewriting it pulls in the earlier
    topic or language."""
    text = question.strip().lower()
    words = [w.replace("’", "'") for w in _WORD_RE.findall(text)]
    if len(words) <= 2 or text.startswith(_CONTINUATIONS):
        return True
    words = [w.replace("’", "'") for w in _WORD_RE.findall(_FORWARD_IT_RE.sub(" ", text))]
    return any(
        w in _REFERENCE_WORDS or (w.endswith("nya") and len(w) > 5 and w not in _NOT_REFERENCE_NYA)
        for w in words
    )


_YES_NO_PREFIX_RE = re.compile(r"^\s*(?:yes|no)\s*[,.!:;—–-]\s*", re.IGNORECASE)
_BARE_YES_NO_RE = re.compile(r"^\s*(?:yes|no)\s*$", re.IGNORECASE)
_FIRST_SENTENCE_END_RE = re.compile(r"[.!?\n]")
_COULD_NOT_FIND_RE = re.compile(r"\b(?:could(?:n['’]t| not)|can(?:['’]t|not)|unable to) find\b", re.IGNORECASE)
_SOURCES_RE = re.compile(r"\b(?:sources?|documents?|context)\b", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:not|no)\b|n['’]t\b", re.IGNORECASE)
_CITED_RE = re.compile(r"\[\d")
# Without a sentence end by then, the reply isn't a short decline.
_MAX_HELD = 300


def _is_decline(sentence: str) -> bool:
    """Says nothing was found, or that the sources don't cover it, citing nothing."""
    if _CITED_RE.search(sentence):
        return False
    if _COULD_NOT_FIND_RE.search(sentence):
        return True
    return bool(_SOURCES_RE.search(sentence) and _NEGATION_RE.search(sentence))


class _YesNoBeforeDecline:
    """Removes a leading "Yes,"/"No," from a reply that says the sources don't
    cover the question: "No, the sources do not mention X" reads as an answer.

    Only a reply starting with yes or no is held back, until its first sentence
    after that word is complete. Every other reply streams as it arrives."""

    def __init__(self) -> None:
        self._held = ""
        self._released = False

    def feed(self, text: str) -> str:
        if self._released:
            return text
        self._held += text
        return self._decide(final=False)

    def finish(self) -> str:
        return "" if self._released else self._decide(final=True)

    def _decide(self, final: bool) -> str:
        held = self._held
        start = held.lstrip()
        could_become_yes_no = start.isalpha() or not start
        if not final and could_become_yes_no and len(start) < 4 and (
            "yes".startswith(start.lower()) or "no".startswith(start.lower())
        ):
            return ""
        prefix = _YES_NO_PREFIX_RE.match(held)
        if not prefix:
            # "No" may still be followed by a comma in the next piece.
            if not final and _BARE_YES_NO_RE.match(held):
                return ""
            return self._release(held)
        rest = held[prefix.end():]
        end = _FIRST_SENTENCE_END_RE.search(rest)
        if not end and not final and len(rest) < _MAX_HELD:
            return ""
        sentence = rest[: end.end()] if end else rest
        if rest and _is_decline(sentence):
            return self._release(rest[0].upper() + rest[1:])
        return self._release(held)

    def _release(self, text: str) -> str:
        self._released = True
        self._held = ""
        return text


# Shorter lines, such as "}" or a bare "- Yes [1]", repeat in normal answers.
_MIN_REPEAT_LINE = 40


class _RepeatGuard:
    """Says when the answer writes a long line it already wrote. At temperature 0
    the model sometimes loops, writing the same sentences until stopped."""

    def __init__(self) -> None:
        self._partial = ""
        self._seen: set[str] = set()
        self._in_code = False
        self.stopped = False

    def feed(self, text: str) -> bool:
        *lines, self._partial = (self._partial + text).split("\n")
        for line in lines:
            line = " ".join(line.split())
            if line.startswith("```"):
                self._in_code = not self._in_code
            elif not self._in_code and len(line) >= _MIN_REPEAT_LINE:
                if line in self._seen:
                    self.stopped = True
                    return True
                self._seen.add(line)
        return False


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
