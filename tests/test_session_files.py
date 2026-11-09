"""Chat history must survive crashes, corrupt files and concurrent turns."""
from __future__ import annotations

import os

import pytest

from rag.chat import ChatService, _load_session, _session_path

from .conftest import StubLLM


class _NoDocuments:
    def search(self, _query, **_kwargs):
        return []


@pytest.fixture
def service(data_dirs):
    chat = ChatService(_NoDocuments())
    chat._provider = StubLLM()
    return chat


def _ask(service, session_id: str, question: str) -> None:
    list(service.chat_stream(session_id, question))


def test_failed_save_keeps_the_previous_history(service, monkeypatch):
    _ask(service, "s", "first")
    before = _session_path("s").read_text(encoding="utf-8")

    def crash_mid_save(*_args, **_kwargs):
        raise OSError("simulated crash")

    monkeypatch.setattr(os, "replace", crash_mid_save)
    list(service.chat_stream("s", "second"))

    assert _session_path("s").read_text(encoding="utf-8") == before


def test_failed_save_is_reported_to_the_client(service, monkeypatch):
    def crash_mid_save(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", crash_mid_save)

    events = list(service.chat_stream("s", "first"))

    assert events[-1]["type"] == "error"
    assert "disk full" in events[-1]["error"]


def test_save_leaves_no_temp_files_behind(service, data_dirs):
    _ask(service, "s", "first")

    assert [p.name for p in data_dirs["SESSION_DIR"].iterdir()] == ["s.json"]


def test_interleaved_turns_in_one_session_are_both_kept(service):
    first = service.chat_stream("s", "from tab one")
    second = service.chat_stream("s", "from tab two")
    next(first)   # both have loaded the history and are waiting on the LLM
    next(second)

    list(first)
    list(second)

    assert [m["content"] for m in _load_session("s") if m["role"] == "user"] == [
        "from tab one",
        "from tab two",
    ]


def test_corrupt_session_file_is_kept_aside_not_overwritten(service, data_dirs):
    path = _session_path("s")
    path.write_text('[{"role": "user", "cont', encoding="utf-8")

    _ask(service, "s", "next question")

    assert [m["content"] for m in _load_session("s") if m["role"] == "user"] == ["next question"]
    kept = [p for p in data_dirs["SESSION_DIR"].iterdir() if p != path]
    assert [p.read_text(encoding="utf-8") for p in kept] == ['[{"role": "user", "cont']


def test_session_file_with_the_wrong_shape_counts_as_corrupt(service, data_dirs):
    _session_path("s").write_text('{"not": "a list"}', encoding="utf-8")

    assert _load_session("s") == []
    assert len(list(data_dirs["SESSION_DIR"].iterdir())) == 1
    assert not _session_path("s").exists()
