"""The configured rules: SG loan is v2 and the rest v1, MY all v2, ID all v1."""
from __future__ import annotations

import pytest

from rag import config_facade
from rag.retriever import _doc_matches
from rag.vector_store import DocRecord

from .conftest import REAL_APP_CODES, REAL_APP_VERSION_MAP


def test_the_three_apps_are_available():
    assert REAL_APP_CODES == ["SG", "MY", "ID"]


@pytest.mark.parametrize(
    "app, functionality, right, wrong",
    [
        ("SG", "loan", "v2", "v1"),
        ("SG", "other", "v1", "v2"),
        ("SG", "card", "v1", "v2"),
        ("MY", "loan", "v2", "v1"),
        ("MY", "other", "v2", "v1"),
        ("ID", "loan", "v1", "v2"),
        ("ID", "other", "v1", "v2"),
    ],
)
def test_each_app_sees_only_its_version(monkeypatch, app, functionality, right, wrong):
    monkeypatch.setattr(config_facade, "APP_VERSION_MAP", REAL_APP_VERSION_MAP)
    doc = lambda version: DocRecord("d", "f", source_type="manual", version=version, functionality=functionality)

    assert _doc_matches(doc(right), {"app_code": app}) is True
    assert _doc_matches(doc(wrong), {"app_code": app}) is False
