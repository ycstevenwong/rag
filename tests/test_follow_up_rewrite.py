"""Only a question that refers back to the conversation is rewritten. Rewriting a
question that already stands on its own pulled the earlier topic or language into
the search: after a Basel question, "What is the maximum loan tenure?" was
searched as "...loan tenure under Basel III" and found nothing."""
from __future__ import annotations

import pytest

from rag.chat import ChatService, _append_to_session
from rag.retriever import RetrievedChunk
from rag.vector_store import ChunkRecord

from .conftest import StubLLM


class _RecordingRetriever:
    def __init__(self):
        self.queries: list[str] = []

    def search(self, query, **_kwargs):
        self.queries.append(query)
        return [RetrievedChunk(ChunkRecord(1, "f", "a fact", 2), 0.1, 0, 0, "a.pdf")]


def _searched_for(question: str) -> tuple[str, StubLLM]:
    _append_to_session("s", [
        {"role": "user", "content": "What is the Basel III output floor?"},
        {"role": "assistant", "content": "The output floor is 72.5% [1]."},
    ])
    retriever = _RecordingRetriever()
    service = ChatService(retriever)
    llm = StubLLM()
    service._provider = llm
    events = list(service.chat_stream("s", question))
    assert events[-1]["type"] == "done"
    return retriever.queries[0], llm


@pytest.mark.parametrize(
    "question",
    [
        "What is the maximum loan tenure?",
        "Berapakah tempoh pinjaman maksimum?",
        "Is the processing fee waived for personal loans?",
        # "it" here points forward to the "to ..." part, not back at the conversation.
        # After a Reg E question, the first was searched as the Reg E question again.
        "How much does it cost to replace a lost debit card?",
        "How long does it take to deliver a replacement card?",
        "Is it possible to waive the replacement fee?",
        "Is it necessary to block the old card first?",
    ],
)
def test_a_question_that_stands_on_its_own_is_searched_as_asked(data_dirs, question):
    query, llm = _searched_for(question)

    assert query == question
    assert llm.complete_calls == []


@pytest.mark.parametrize(
    "question",
    [
        "When does it take effect?",
        "How much does it cost?",
        "Is it possible?",
        "How long does it take for smaller banks?",
        "What about this for smaller banks?",
        "And for MY?",
        "Why?",
        "Bilakah ia berkuat kuasa?",
        "Kapan itu berlaku?",
        "Bagaimana dengan bank kecil?",
    ],
)
def test_a_question_that_refers_back_is_rewritten(data_dirs, question):
    query, _ = _searched_for(question)

    assert query == "standalone question"


def test_the_rewrite_keeps_the_follow_ups_language(data_dirs):
    _, llm = _searched_for("When does it take effect?")

    assert "same language as the follow-up" in llm.complete_calls[0]


def test_the_rewrite_keeps_the_follow_ups_own_words(data_dirs):
    """A rewrite replaced "How much does it cost to replace a lost debit card?" with
    the earlier Reg E question, so the search never looked for the fee."""
    _, llm = _searched_for("When does it take effect?")

    assert "Keep the follow-up's own words" in llm.complete_calls[0]
    assert "return it unchanged" in llm.complete_calls[0]
