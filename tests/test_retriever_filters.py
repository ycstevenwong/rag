"""Filtering and ranking, using real stores and the stub embedder."""
from __future__ import annotations

import numpy as np
import pytest

from rag.retriever import Retriever
from rag.vector_store import ChunkRecord, DocRecord, FaissStore, FileRecord

from .conftest import StubEmbedder


def _build(root, files: dict[str, dict]) -> Retriever:
    """files: {file_id: {"texts": [...], "doc": DocRecord kwargs}}"""
    store = FaissStore(root / "v", root / "c", root / "d", root / "f")
    embedder = StubEmbedder()
    next_id = 0
    for file_id, spec in files.items():
        records = []
        for text in spec["texts"]:
            records.append(ChunkRecord(next_id, file_id, text, 5))
            next_id += 1
        store.add(np.stack([embedder.encode_query(r.text) for r in records]), records)
        store.add_file(FileRecord(file_id, file_id, f"{file_id}.pdf", 0.0, len(records)))
        store.add_doc(DocRecord(doc_id=f"doc-{file_id}", file_id=file_id, **spec["doc"]))
    return Retriever(store, embedder)


def test_filter_finds_matches_ranked_below_the_unfiltered_top(tmp_path):
    retriever = _build(tmp_path, {
        "other": {
            "texts": [f"refund policy overview section {i} refund refund" for i in range(500)],
            "doc": {"app_code": "app-a"},
        },
        "target": {
            "texts": [f"refund handled by the billing team note {i}" for i in range(5)],
            "doc": {"app_code": "app-b"},
        },
    })

    results = retriever.search(
        "refund policy overview", k_vector=20, top_n=10, filters={"app_code": "app-b"}
    )

    assert sorted(r.chunk.text for r in results) == [
        f"refund handled by the billing team note {i}" for i in range(5)
    ]
    assert {r.doc.doc_id for r in results} == {"doc-target"}


def test_filter_excludes_non_matching_documents(tmp_path):
    retriever = _build(tmp_path, {
        "a": {"texts": ["alpha token"], "doc": {"app_code": "app-a"}},
        "b": {"texts": ["alpha token again"], "doc": {"app_code": "app-b"}},
        "c": {"texts": ["unrelated words"], "doc": {"app_code": "app-c"}},
    })

    results = retriever.search("alpha token", filters={"app_code": "app-a"})

    assert [r.chunk.text for r in results] == ["alpha token"]


def test_filtered_ranks_stop_at_the_k_limits(tmp_path):
    retriever = _build(tmp_path, {
        "a": {
            "texts": [f"shared words number {i}" for i in range(50)],
            "doc": {"app_code": "app-a"},
        },
    })

    results = retriever.search(
        "shared words", k_vector=3, top_n=50, filters={"app_code": "app-a"}
    )

    assert sorted(r.vector_rank for r in results) == [0, 1, 2]


def test_results_are_ranked_by_cosine_similarity_to_the_query(tmp_path):
    retriever = _build(tmp_path, {
        "a": {
            "texts": ["kilo lima mike", "november oscar papa", "quebec romeo sierra"],
            "doc": {},
        },
    })
    embedder = StubEmbedder()
    query = embedder.encode_query("kilo lima mike")

    results = retriever.search("kilo lima mike", k_vector=3, top_n=3)

    assert results[0].chunk.text == "kilo lima mike"
    assert results[0].vector_rank == 0
    for r in results:
        assert r.score == pytest.approx(float(query @ embedder.encode_query(r.chunk.text)), abs=1e-5)
    by_rank = sorted(results, key=lambda r: r.vector_rank)
    assert [r.score for r in by_rank] == sorted((r.score for r in results), reverse=True)


RULES = {
    "SG": {"loan": "v2", "*": "v1"},
    "MY": {"*": "v2"},
    "ID": {"*": "v1"},
}


@pytest.mark.parametrize(
    "app, functionality, version, visible",
    [
        ("SG", "loan", "v2", True),
        ("SG", "loan", "v1", False),
        ("SG", "card", "v1", True),
        ("SG", "card", "v2", False),
        ("MY", "loan", "v2", True),
        ("MY", "loan", "v1", False),
        ("MY", "card", "v2", True),
        ("ID", "loan", "v1", True),
        ("ID", "loan", "v2", False),
        # Labels typed differently must follow the same rule, not fall through to "*".
        ("SG", "Loan", "v2", True),
        ("SG", "LOAN ", "v1", False),
        ("MY", "card", "V2", True),
        ("ID", "card", "V2", False),
        ("SG", "Loan", "V2", True),
    ],
)
def test_version_rules_ignore_case_and_spacing(monkeypatch, app, functionality, version, visible):
    from rag import config_facade
    from rag.retriever import _doc_matches

    monkeypatch.setattr(config_facade, "APP_VERSION_MAP", RULES)
    doc = DocRecord("d", "f", source_type="manual", version=version, functionality=functionality)

    assert _doc_matches(doc, {"app_code": app}) is visible


def test_version_rule_given_as_a_list_ignores_case(monkeypatch):
    from rag import config_facade
    from rag.retriever import _doc_matches

    monkeypatch.setattr(config_facade, "APP_VERSION_MAP", {"SG": {"*": ["V1", "v2"]}})

    assert _doc_matches(DocRecord("d", "f", version="v1"), {"app_code": "SG"}) is True
    assert _doc_matches(DocRecord("d", "f", version="V2"), {"app_code": "SG"}) is True
    assert _doc_matches(DocRecord("d", "f", version="v3"), {"app_code": "SG"}) is False
