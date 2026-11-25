"""Two versions can share a filename, so citations have to name the version."""
from __future__ import annotations

from rag.citations import build_context_block
from rag.retriever import RetrievedChunk
from rag.vector_store import ChunkRecord, DocRecord


def _retrieved(version: str, meta=None, doc=True) -> RetrievedChunk:
    chunk = ChunkRecord(1, "f", "The loan tenure is 7 years.", 5, meta or {"pages": [3]})
    record = DocRecord("d", "f", version=version) if doc else None
    return RetrievedChunk(chunk, 0.1, 0, "loan_manual.pdf", record)


def test_locator_names_the_version():
    _, sources = build_context_block([_retrieved("v2")], token_budget=1000)

    assert sources[0]["locator"] == "loan_manual.pdf (v2), p.3"
    assert sources[0]["version"] == "v2"


def test_the_model_sees_the_version_too():
    context, _ = build_context_block([_retrieved("v2")], token_budget=1000)

    assert context.startswith("[1] loan_manual.pdf (v2), p.3\n")


def test_doc_without_a_version_has_a_plain_locator():
    _, sources = build_context_block([_retrieved("")], token_budget=1000)

    assert sources[0]["locator"] == "loan_manual.pdf, p.3"
    assert sources[0]["version"] == ""


def test_chunk_without_a_matched_doc_has_a_plain_locator():
    _, sources = build_context_block([_retrieved("", doc=False)], token_budget=1000)

    assert sources[0]["locator"] == "loan_manual.pdf, p.3"
