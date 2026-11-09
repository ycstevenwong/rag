"""The real UI in Chrome against a live server with stub model endpoints.

Skipped unless the `playwright` package and a local Chrome are available. It needs
no `playwright install`, because it uses the installed Chrome.
"""
from __future__ import annotations

import io
import threading

import pytest
from werkzeug.security import generate_password_hash
from werkzeug.serving import make_server

import config

sync_api = pytest.importorskip("playwright.sync_api")

_PASSWORD = "test-password"


@pytest.fixture
def live_app(app_module, monkeypatch):
    """A running server with admin login on, and one way to add documents to it."""
    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(
        config, "ADMIN_PASSWORD_HASH", generate_password_hash(_PASSWORD, method="pbkdf2:sha256:1000")
    )
    monkeypatch.setattr(config, "FLASK_SECRET_KEY", "a-private-test-key")
    app = app_module.create_app()
    server = make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield app, f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    thread.join()


@pytest.fixture
def page():
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(channel="chrome", headless=True)
        except sync_api.Error as exc:
            pytest.skip(f"Chrome isn't available: {exc}")
        yield browser.new_page()
        browser.close()


def _log_in(page, base_url: str) -> None:
    response = page.request.post(
        f"{base_url}/admin/login", data={"username": "admin", "password": _PASSWORD}
    )
    assert response.ok


def _open(page, base_url: str) -> None:
    page.goto(base_url)
    page.wait_for_function("document.querySelector('#upload-hint-user') !== null")
    page.wait_for_load_state("networkidle")


def test_anonymous_visitor_sees_only_the_anonymous_upload_controls(live_app, page):
    _, base_url = live_app

    _open(page, base_url)

    assert page.is_visible("#upload-hint-user")
    assert page.is_visible("#anonymous-only-fields")
    assert not page.is_visible("#upload-hint-admin")


def test_admin_sees_only_the_admin_upload_controls(live_app, page):
    _, base_url = live_app
    _log_in(page, base_url)

    _open(page, base_url)

    assert page.is_visible("#upload-hint-admin")
    assert not page.is_visible("#upload-hint-user")
    assert not page.is_visible("#anonymous-only-fields")
