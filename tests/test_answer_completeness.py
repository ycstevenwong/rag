"""Answers include the conditions the source attaches. Asked the replacement fee,
the model answered "SGD 15 [6]" and left out that the first replacement each year
is free (6 of 9 such answers complete, against 9 of 9 with this rule).

The rule is limited to the fact asked about. Asked to include every condition the
sources give, the model answered "What is the standard FDIC amount?" with 13
paragraphs, and once looped on them for hours."""
from __future__ import annotations

from rag.chat import ChatService
from rag.retriever import RetrievedChunk
from rag.vector_store import ChunkRecord

from .conftest import StubLLM


class _OneDocument:
    def search(self, _query, **_kwargs):
        return [RetrievedChunk(ChunkRecord(1, "f", "The fee is SGD 15.", 5), 0.9, 0, "a.docx")]


class _SystemRecordingLLM(StubLLM):
    def stream_chat(self, system, messages, on_finish=None):
        self.system = system
        yield from super().stream_chat(system, messages)


def test_the_model_is_told_to_include_conditions_and_exceptions(data_dirs):
    service = ChatService(_OneDocument())
    service._provider = _SystemRecordingLLM()

    list(service.chat_stream("s", "How much is the replacement fee?"))

    system = service._provider.system
    assert "condition, exception, waiver or limit for the fact asked about, include it" in system
    assert "Do not add other facts the question did not ask about." in system
