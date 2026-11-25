"""With nothing retrieved there's nothing to answer from, so the LLM isn't asked."""
from __future__ import annotations

import pytest

from rag.chat import NO_SOURCES_REPLY, ChatService, _load_session

from .conftest import StubLLM


class _NoDocuments:
    def search(self, _query, **_kwargs):
        return []


@pytest.fixture
def service(data_dirs):
    chat = ChatService(_NoDocuments())
    chat._provider = StubLLM()
    return chat


def test_llm_is_not_called_when_nothing_is_retrieved(service):
    list(service.chat_stream("s", "Are crypto assets insured?"))

    assert service._provider.stream_calls == []


def test_reply_says_nothing_was_found(service):
    events = list(service.chat_stream("s", "Are crypto assets insured?"))

    assert [e["type"] for e in events] == ["sources", "delta", "done"]
    assert events[0]["sources"] == []
    assert events[1]["text"] == NO_SOURCES_REPLY
    assert "couldn't find" in NO_SOURCES_REPLY


def test_refusal_is_saved_like_any_other_turn(service):
    list(service.chat_stream("s", "Are crypto assets insured?"))

    assert [m["content"] for m in _load_session("s")] == ["Are crypto assets insured?", NO_SOURCES_REPLY]
