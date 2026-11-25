"""An answer that starts repeating itself is stopped. At temperature 0 the model
answered an FDIC question correctly, then wrote the same cited sentences again and
again for hours, and the chat never ended."""
from __future__ import annotations

from rag.chat import CUT_SHORT_NOTE, ChatService
from rag.retriever import RetrievedChunk
from rag.vector_store import ChunkRecord

from .conftest import StubLLM

LOOPING = "The FDIC insures deposits up to $250,000 per depositor. [1]\n\n"


class _OneDocument:
    def search(self, _query, **_kwargs):
        return [RetrievedChunk(ChunkRecord(1, "f", "FDIC insurance is $250,000.", 5), 0.9, 0, "a.pdf")]


class _LoopingLLM(StubLLM):
    def __init__(self):
        super().__init__()
        self.closed = False
        self.sent = 0

    def stream_chat(self, _system, messages, on_finish=None):
        self.stream_calls.append(messages)
        try:
            yield "Intro line that is not repeated anywhere else. [1]\n\n"
            for _ in range(500):
                # Sent in pieces, the way tokens arrive.
                for piece in (LOOPING[:20], LOOPING[20:45], LOOPING[45:]):
                    self.sent += 1
                    yield piece
        finally:
            self.closed = True


class _CappedLLM(StubLLM):
    def stream_chat(self, _system, messages, on_finish=None):
        self.stream_calls.append(messages)
        yield "The first part of a very long answer [1]"
        if on_finish:
            on_finish("length")


def _answer(llm) -> tuple[str, list[dict]]:
    service = ChatService(_OneDocument())
    service._provider = llm
    events = list(service.chat_stream("s", "What is the FDIC insurance amount?"))
    return "".join(e["text"] for e in events if e["type"] == "delta"), events


def test_an_answer_that_repeats_itself_is_stopped(data_dirs):
    llm = _LoopingLLM()

    text, events = _answer(llm)

    assert events[-1]["type"] == "done"
    assert text.count(LOOPING.strip()) <= 2
    assert text.endswith(CUT_SHORT_NOTE)
    assert llm.closed  # the request to the model is closed, so it stops generating
    assert llm.sent < 20


def test_an_answer_that_reaches_the_length_limit_says_so(data_dirs):
    text, events = _answer(_CappedLLM())

    assert events[-1]["type"] == "done"
    assert text == "The first part of a very long answer [1]" + CUT_SHORT_NOTE


def test_a_normal_answer_has_no_note(data_dirs):
    text, _ = _answer(StubLLM())

    assert text == "Stub answer [1]."


def test_a_repeated_short_line_is_not_a_loop(data_dirs):
    """Short lines such as a bare citation or "}" repeat in normal answers."""
    llm = StubLLM()
    llm.reply = "- Fee: SGD 15 [1]\n- Waived: yes\n```\n}\n}\n}\n```\n- Fee: SGD 15 [1]\n"

    text, _ = _answer(llm)

    assert text == llm.reply


def test_the_note_is_not_saved_into_the_history_sent_to_the_model(data_dirs):
    service = ChatService(_OneDocument())
    service._provider = _CappedLLM()
    list(service.chat_stream("s", "What is the FDIC insurance amount?"))
    list(service.chat_stream("s", "And for joint accounts?"))

    history = service._provider.stream_calls[-1]
    assert CUT_SHORT_NOTE.strip() not in history[1]["content"]
