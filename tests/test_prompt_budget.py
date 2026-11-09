"""The total prompt must stay within the token budget, history included."""
from __future__ import annotations

import pytest

from rag import config_facade
from rag.chat import SYSTEM_PROMPT, ChatService, _append_to_session
from rag.chunker import count_tokens
from rag.retriever import RetrievedChunk
from rag.vector_store import ChunkRecord

from .conftest import StubLLM


class _OneDocument:
    def search(self, _query, **_kwargs):
        chunk = ChunkRecord(1, "f", "short fact", 2)
        return [RetrievedChunk(chunk, 0.1, 0, 0, "a.pdf")]


# The system prompt plus about 25 tokens for the question and its one-source context.
_OVERHEAD = count_tokens(SYSTEM_PROMPT) + 25


@pytest.fixture
def service(data_dirs):
    chat = ChatService(_OneDocument())
    chat._provider = StubLLM()
    return chat


def _seed_turns(session_id: str, n: int, chars_per_message: int = 400) -> None:
    for i in range(n):
        _append_to_session(session_id, [
            {"role": "user", "content": f"q{i} " + "x" * chars_per_message},
            {"role": "assistant", "content": f"a{i} " + "y" * chars_per_message},
        ])


def _history_sent(service, question: str = "now") -> list[str]:
    list(service.chat_stream("s", question))
    return [m["content"][:2] for m in service._provider.stream_calls[-1][:-1]]


def test_oldest_turns_are_dropped_to_fit_the_budget(service, monkeypatch):
    # Room for two 200-token turns but not three.
    monkeypatch.setattr(config_facade, "CONTEXT_TOKEN_BUDGET", _OVERHEAD + 450)
    _seed_turns("s", 5)

    assert _history_sent(service) == ["q3", "a3", "q4", "a4"]


def test_all_turns_are_sent_when_they_fit(service, monkeypatch):
    monkeypatch.setattr(config_facade, "CONTEXT_TOKEN_BUDGET", 16000)
    _seed_turns("s", 2)

    assert _history_sent(service) == ["q0", "a0", "q1", "a1"]


def test_turn_limit_still_applies(service, monkeypatch):
    monkeypatch.setattr(config_facade, "CONTEXT_TOKEN_BUDGET", 100000)
    _seed_turns("s", 10, chars_per_message=10)

    assert len(_history_sent(service)) == 16


def test_no_history_is_sent_when_the_current_message_uses_the_budget(service, monkeypatch):
    monkeypatch.setattr(config_facade, "CONTEXT_TOKEN_BUDGET", 50)
    _seed_turns("s", 2)

    assert _history_sent(service, question="z" * 400) == []


def test_turns_are_never_split(service, monkeypatch):
    """Room for one 100-token message, but not its pair."""
    monkeypatch.setattr(config_facade, "CONTEXT_TOKEN_BUDGET", _OVERHEAD + 150)
    _seed_turns("s", 1)

    assert _history_sent(service) == []


@pytest.mark.parametrize(
    "text, expected",
    [
        ("a" * 400, 100),      # about 4 characters per token for Latin text
        ("退" * 100, 100),     # about 1 token per ideograph
        ("カ" * 50, 50),
        ("환" * 50, 50),
        ("a" * 40 + "退" * 10, 20),
        ("", 1),
    ],
)
def test_count_tokens(text, expected):
    assert count_tokens(text) == expected
