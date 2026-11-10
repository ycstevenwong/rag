"""A reply saying the sources don't cover something must not start with "Yes" or
"No": that reads as an answer. Seen on real Ollama: "No, the sources do not
mention remittance transfers." """
from __future__ import annotations

import pytest

from rag.chat import ChatService, _load_session
from rag.retriever import RetrievedChunk
from rag.vector_store import ChunkRecord


class _OneDocument:
    def search(self, _query, **_kwargs):
        return [RetrievedChunk(ChunkRecord(1, "f", "a fact", 2), 0.1, 0, 0, "a.pdf")]


class _PiecesLLM:
    """Streams a reply in the given pieces, as a real model does."""

    def __init__(self, pieces: list[str]):
        self.pieces = pieces

    def stream_chat(self, _system, _messages):
        yield from self.pieces

    def complete(self, _prompt, max_tokens=256):
        return "standalone question"


def _chat(data_dirs, pieces: list[str]) -> list[str]:
    service = ChatService(_OneDocument())
    service._provider = _PiecesLLM(pieces)
    events = list(service.chat_stream("s", "Does Regulation E cover remittance transfers?"))
    assert events[-1]["type"] == "done"
    return [e["text"] for e in events if e["type"] == "delta"]


@pytest.mark.parametrize(
    "pieces, expected",
    [
        (["No", ", the sources", " do not mention remittance transfers."],
         "The sources do not mention remittance transfers."),
        (["Yes, I couldn't find that in the documents."],
         "I couldn't find that in the documents."),
        (["No.", " The documents don't say whether remittance transfers are covered."],
         "The documents don't say whether remittance transfers are covered."),
        (["NO", " — ", "no information about that is in the provided sources."],
         "No information about that is in the provided sources."),
    ],
)
def test_yes_or_no_before_a_decline_is_removed(data_dirs, pieces, expected):
    assert "".join(_chat(data_dirs, pieces)) == expected


@pytest.mark.parametrize(
    "pieces",
    [
        ["No, the FDIC does not insure crypto assets [1]."],
        ["Yes", ", Regulation E covers remittance transfers [2]."],
        ["No, the documents list it as excluded [1]."],
        ["No direct answer is found in the provided sources."],
        ["Nothing in the documents covers that."],
    ],
)
def test_answers_and_other_replies_are_unchanged(data_dirs, pieces):
    assert "".join(_chat(data_dirs, pieces)) == "".join(pieces)


def test_other_replies_still_stream_piece_by_piece(data_dirs):
    pieces = ["The output", " floor is", " 72.5% [1]."]

    assert _chat(data_dirs, pieces) == pieces


def test_saved_history_has_the_corrected_reply(data_dirs):
    _chat(data_dirs, ["No, the sources do not mention it."])

    assert _load_session("s")[-1]["content"] == "The sources do not mention it."
