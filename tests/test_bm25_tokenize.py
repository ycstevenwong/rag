"""Keyword search has to work for languages the embedding model already handles."""
from __future__ import annotations

import pytest

from rag.bm25_store import BM25Store, tokenize


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Error XK-204 in snake_case", ["error", "xk", "204", "in", "snake_case"]),
        ("café Überprüfung", ["café", "überprüfung"]),
        ("café", ["café"]),  # decomposed accent
        ("Привет мир", ["привет", "мир"]),
        ("退款流程", ["退款", "款流", "流程"]),
        ("退", ["退"]),
        ("API退款v2", ["api", "退款", "v2"]),
        ("カタカナ", ["カタ", "タカ", "カナ"]),
        ("환불 절차", ["환불", "절차"]),
        ("!!! ???", []),
    ],
)
def test_tokenize(text, expected):
    assert tokenize(text) == expected


def test_cjk_chunk_is_found_by_keyword():
    store = BM25Store()
    store.rebuild([(0, "english words here"), (1, "退款流程说明"), (2, "other english text")])

    assert [cid for cid, _ in store.search("退款", 5)] == [1]


@pytest.mark.parametrize("text", ["!!! ???", "---"])
def test_corpus_with_no_tokens_does_not_crash(text):
    store = BM25Store()

    store.rebuild([(0, text)])

    assert store.search("anything", 5) == []
