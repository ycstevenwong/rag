"""LLMProvider against a real local HTTP server speaking OpenAI-style SSE."""
from __future__ import annotations

import http.server
import json
import threading

import pytest

from rag.llm import LLMProvider


@pytest.fixture
def sse_provider():
    """Returns a function serving a raw SSE body and giving a provider for it."""
    state = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.send_response(200)
            self.send_header("Content-Type", state["content_type"])
            self.end_headers()
            self.wfile.write(state["body"])

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def serve(*events, content_type="text/event-stream"):
        state["content_type"] = content_type
        state["body"] = "".join(
            f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events
        ).encode("utf-8") + b"data: [DONE]\n\n"
        return LLMProvider("model", f"http://127.0.0.1:{server.server_address[1]}", "key")

    yield serve
    server.shutdown()


def _delta(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}}]}


def _stream(provider) -> str:
    return "".join(provider.stream_chat("system", [{"role": "user", "content": "q"}]))


@pytest.mark.parametrize(
    "content_type", ["text/event-stream", "text/event-stream; charset=utf-8"]
)
def test_non_ascii_text_arrives_intact(sse_provider, content_type):
    provider = sse_provider(_delta("naïve — “quoted” 退款"), content_type=content_type)

    assert _stream(provider) == "naïve — “quoted” 退款"


def test_utf8_byte_0x85_does_not_split_a_line(sse_provider):
    """Å encodes as C3 85, and 0x85 is a line break once decoded as Latin-1."""
    provider = sse_provider(_delta("Ångström"))

    assert _stream(provider) == "Ångström"


def test_chunk_with_empty_choices_is_skipped(sse_provider):
    provider = sse_provider({"choices": []}, _delta("hello"), {"choices": [], "usage": {"total": 1}})

    assert _stream(provider) == "hello"


def test_unicode_line_separator_in_content_is_not_a_line_break(sse_provider):
    provider = sse_provider(_delta("one two"))

    assert _stream(provider) == "one two"
