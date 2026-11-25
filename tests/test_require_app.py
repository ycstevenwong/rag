"""Without an app there's no version rule, so versions would be mixed silently."""
from __future__ import annotations

import pytest

import config


@pytest.fixture
def apps(monkeypatch):
    monkeypatch.setattr(config, "APP_CODES", ["SG", "MY", "ID"])


@pytest.mark.parametrize(
    "filters",
    [None, {}, {"app_code": ""}, {"app_code": "Thailand"}, {"app_code": "sg"}, {"source_type": "manual"}, "SG"],
)
def test_chat_needs_a_known_app(apps, client, data_dirs, filters):
    response = client.post("/chat", json={"session_id": "s", "message": "hello", "filters": filters})

    assert response.status_code == 400
    assert "app" in response.json["error"].lower()
    assert list(data_dirs["SESSION_DIR"].iterdir()) == []


@pytest.mark.parametrize("app", ["SG", "MY", "ID"])
def test_chat_accepts_a_known_app(apps, client, app):
    response = client.post("/chat", json={"session_id": "s", "message": "hello", "filters": {"app_code": app}})

    assert response.status_code == 200


def test_chat_needs_no_app_when_none_are_configured(client):
    assert client.post("/chat", json={"session_id": "s", "message": "hello"}).status_code == 200


def test_chat_filter_offers_no_all_apps_option(apps, client):
    page = client.get("/").get_data(as_text=True)

    assert "Any app" not in page
