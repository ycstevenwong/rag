"""Each question ends with a reminder to cite. Earlier answers are sent without
their [n] markers, and a few turns into a chat the model copied that and stopped
citing (16 of 21 answers cited, against 21 of 21 in fresh chats)."""
from __future__ import annotations

from rag.chat import ChatService, _append_to_session
from rag.retriever import RetrievedChunk
from rag.vector_store import ChunkRecord

from .conftest import StubLLM


class _OneDocument:
    def search(self, _query, **_kwargs):
        return [RetrievedChunk(ChunkRecord(1, "f", "The fee is SGD 15.", 5), 0.9, 0, "a.docx")]


def test_the_question_ends_with_a_reminder_to_cite(data_dirs):
    _append_to_session("s", [
        {"role": "user", "content": "What is the output floor?"},
        {"role": "assistant", "content": "It is 72.5% [1]."},
    ])
    service = ChatService(_OneDocument())
    service._provider = StubLLM()

    list(service.chat_stream("s", "How much is the replacement fee?"))

    messages = service._provider.stream_calls[-1]
    assert messages[1]["content"] == "It is 72.5%."  # history still has no old numbers
    assert messages[-1]["content"].endswith(
        "Question: How much is the replacement fee?\n\n"
        "Answer from the sources above and cite each fact as [n]."
    )
