"""Citation numbers only mean something within the turn that produced them."""
from __future__ import annotations

import pytest

from rag.chat import ChatService, _load_session
from rag.ingest import IngestPipeline
from rag.retriever import Retriever
from rag.vector_store import FaissStore

from .conftest import StubEmbedder, StubLLM


@pytest.fixture
def service(tmp_path, data_dirs):
    root = tmp_path / "index"
    root.mkdir()
    store = FaissStore(root / "v", root / "c", root / "d", root / "f")
    embedder = StubEmbedder()
    pipeline = IngestPipeline(store, embedder)
    doc = tmp_path / "limits.txt"
    doc.write_text("The limit is 500 requests.")
    for _ in pipeline.ingest_stream(doc, "limits.txt"):
        pass
    chat = ChatService(Retriever(store, embedder))
    chat._provider = StubLLM()
    return chat


def _ask(service, question: str, session_id: str = "s1") -> list[dict]:
    return list(service.chat_stream(session_id, question))


def test_earlier_citation_markers_are_not_sent_to_the_llm(service):
    service._provider.reply = "The limit is 500 [1][2], see also [1, 2]."
    _ask(service, "first")
    _ask(service, "second")

    history = service._provider.stream_calls[-1]
    assert history[1] == {"role": "assistant", "content": "The limit is 500, see also."}


def test_earlier_citation_markers_are_not_in_the_rewrite_prompt(service):
    service._provider.reply = "The limit is 500 [1]."
    _ask(service, "first")
    _ask(service, "second")

    assert "[1]" not in service._provider.complete_calls[-1]
    assert "The limit is 500." in service._provider.complete_calls[-1]


def test_code_in_earlier_answers_is_preserved(service):
    service._provider.reply = "Use `rows[1]` here [1].\n```\nx = arr[2]\n```"
    _ask(service, "first")
    _ask(service, "second")

    assert (
        service._provider.stream_calls[-1][1]["content"]
        == "Use `rows[1]` here.\n```\nx = arr[2]\n```"
    )


def test_citation_attached_to_a_word_is_removed(service):
    service._provider.reply = "It allows 500 requests[1]."
    _ask(service, "first")
    _ask(service, "second")

    assert service._provider.stream_calls[-1][1]["content"] == "It allows 500 requests."


def test_saved_session_keeps_citations_for_display(service):
    service._provider.reply = "The limit is 500 [1]."
    _ask(service, "first")

    assert _load_session("s1")[1]["content"] == "The limit is 500 [1]."


def test_current_turn_context_keeps_its_numbering(service):
    _ask(service, "limit")

    assert service._provider.stream_calls[-1][-1]["content"].startswith("Context:\n[1] limits.txt")
