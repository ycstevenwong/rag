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

from .conftest import docx_bytes

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
    monkeypatch.setattr(config, "APP_CODES", ["SG", "MY", "ID"])
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


def _upload(app, name: str, body: bytes) -> None:
    client = app.test_client()
    with client.session_transaction() as session:
        session["is_admin"] = True
    client.post(
        "/upload",
        data={"file": (io.BytesIO(body), name), "source_type": "manual"},
        content_type="multipart/form-data",
    ).get_data()


def _ask(page, question: str, nth_answer: int) -> None:
    page.fill("#message-input", question)
    page.keyboard.press("Enter")
    page.wait_for_function(
        """n => {
            const answers = document.querySelectorAll('.message.assistant');
            return answers.length === n && !answers[n - 1].classList.contains('cursor')
                && answers[n - 1].textContent.length > 0;
        }""",
        arg=nth_answer,
    )


def test_old_citation_restores_that_answers_search_query(live_app, page):
    app, base_url = live_app
    _upload(app, "zebra.docx", docx_bytes("zebra quota limit is 500"))
    _open(page, base_url)
    page.select_option("#filter-app-code", "SG")
    _ask(page, "zebra quota", 1)
    _ask(page, "a follow-up", 2)  # StubLLM rewrites this to "standalone question"
    query = page.locator("#rewritten-query")
    assert "standalone question" in query.inner_text()

    page.locator(".message.assistant").nth(0).locator(".citation").first.click()
    assert "zebra quota" in query.inner_text()

    page.locator(".message.assistant").nth(1).locator(".citation").first.click()
    assert "standalone question" in query.inner_text()


def test_send_is_disabled_until_an_app_is_selected(live_app, page):
    _, base_url = live_app
    _open(page, base_url)
    assert page.locator("#filter-app-code option").all_inner_texts() == ["Select app *", "SG", "MY", "ID"]
    assert page.is_disabled("#send-btn")

    page.fill("#message-input", "hello")
    page.keyboard.press("Enter")
    page.wait_for_timeout(200)
    assert page.locator(".message").count() == 0
    assert page.input_value("#message-input") == "hello"

    page.select_option("#filter-app-code", "SG")
    assert page.is_enabled("#send-btn")


def test_changing_app_starts_a_new_conversation(live_app, page):
    """Earlier answers come from the other app's document versions."""
    app, base_url = live_app
    _upload(app, "zebra.docx", docx_bytes("zebra quota limit is 500"))
    _open(page, base_url)
    page.select_option("#filter-app-code", "SG")
    _ask(page, "zebra quota", 1)
    first_session = page.inner_text("#session-id")

    page.select_option("#filter-app-code", "MY")

    assert page.locator(".message").count() == 0
    assert page.inner_text("#session-id") != first_session
    assert page.is_enabled("#send-btn")


def test_reloading_starts_a_fresh_chat(live_app, page):
    """The earlier turns are no longer on screen, so they mustn't shape the next answer."""
    app, base_url = live_app
    _upload(app, "zebra.docx", docx_bytes("zebra quota limit is 500"))
    _open(page, base_url)
    page.select_option("#filter-app-code", "SG")
    _ask(page, "zebra quota", 1)
    first_session = page.inner_text("#session-id")

    page.reload()
    page.wait_for_load_state("networkidle")
    page.select_option("#filter-app-code", "SG")
    _ask(page, "zebra quota", 1)

    assert page.inner_text("#session-id") != first_session
    # A turn with history is rewritten (the stub says "standalone question"); a
    # fresh chat's first question is searched as asked.
    assert "zebra quota" in page.inner_text("#rewritten-query")


def test_labels_are_chosen_from_dropdowns_including_when_reviewing(live_app, page, monkeypatch):
    app, base_url = live_app
    monkeypatch.setattr(config, "FUNCTIONALITIES", ["loan", "other"])
    monkeypatch.setattr(config, "VERSIONS", ["v1", "v2"])
    _log_in(page, base_url)
    _open(page, base_url)
    assert page.locator("#upload-version option").all_inner_texts() == ["Version *", "v1", "v2"]
    assert page.locator("#upload-functionality option").all_inner_texts() == ["Functionality *", "loan", "other"]

    # Upload as an admin through the real form, then edit through the docs table.
    page.select_option("#upload-source-type", "manual")
    page.select_option("#upload-version", "v1")
    page.select_option("#upload-functionality", "loan")
    page.set_input_files(
        "#file-input",
        {"name": "guide.docx", "mimeType": "application/octet-stream", "buffer": docx_bytes("loan guide")},
    )
    page.wait_for_function("/Indexed/.test(document.querySelector('#upload-status').textContent)")
    page.click("#open-docs-modal")
    page.click('#docs-tbody [data-action="edit"]')
    edit = page.locator("tr.edit-row")
    assert edit.locator('[data-field="version"]').input_value() == "v1"
    assert edit.locator('[data-field="version"] option').all_inner_texts()[-2:] == ["v1", "v2"]
    edit.locator('[data-field="version"]').select_option("v2")
    edit.locator('[data-action="save-edit"]').click()
    page.wait_for_function("document.querySelector('tr.edit-row') === null")

    response = page.request.get(f"{base_url}/docs")
    assert response.json()["items"][0]["version"] == "v2"


def _fill_admin_upload_form(page, base_url: str, monkeypatch) -> None:
    monkeypatch.setattr(config, "FUNCTIONALITIES", ["loan"])
    monkeypatch.setattr(config, "VERSIONS", ["v1"])
    _log_in(page, base_url)
    _open(page, base_url)
    page.select_option("#upload-source-type", "manual")
    page.select_option("#upload-version", "v1")
    page.select_option("#upload-functionality", "loan")


def test_a_rejected_upload_shows_the_servers_reason(live_app, page, monkeypatch):
    _, base_url = live_app
    _fill_admin_upload_form(page, base_url, monkeypatch)

    page.set_input_files(
        "#file-input",
        {"name": "fees.xlsx", "mimeType": "application/octet-stream", "buffer": b"not a docx"},
    )

    status = page.locator("#upload-status")
    status.filter(has_text="Unsupported").wait_for()
    assert "Convert to .docx or .pdf first" in status.inner_text()
