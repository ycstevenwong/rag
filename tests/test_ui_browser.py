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
    _ask(page, "what about it?", 2)  # StubLLM rewrites this to "standalone question"
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


def _two_documents(app) -> None:
    _upload(app, "zebra.docx", docx_bytes("zebra quota limit is 500"))
    _upload(app, "yak.docx", docx_bytes("yak grazing season is spring"))


def test_sources_the_answer_cites_come_first_and_the_rest_are_folded(live_app, page):
    app, base_url = live_app
    _two_documents(app)
    _open(page, base_url)
    page.select_option("#filter-app-code", "SG")

    _ask(page, "zebra quota", 1)  # the stub answers "Stub answer [1]."

    assert page.locator("#sources-list > .source-card").count() == 1
    assert page.locator("#sources-list > .source-card .n").inner_text() == "1"
    others = page.locator("#sources-list details.other-sources")
    assert "Other passages searched (1)" in others.locator("summary").inner_text()
    assert not others.locator(".source-card").is_visible()


def test_a_reply_that_cites_nothing_says_no_source_was_used(live_app, page, monkeypatch):
    from .conftest import StubLLM

    monkeypatch.setattr(StubLLM, "reply", "I couldn't find it in the documents.")
    app, base_url = live_app
    _two_documents(app)
    _open(page, base_url)
    page.select_option("#filter-app-code", "SG")

    _ask(page, "zebra quota", 1)

    assert page.locator("#sources-list > .source-card").count() == 0
    assert "This answer didn't use any of the documents." in page.inner_text("#sources-list")
    assert "Other passages searched (2)" in page.inner_text("#sources-list details.other-sources summary")


def _admin_upload_through_the_form(page, name: str, text: str) -> None:
    page.set_input_files(
        "#file-input",
        {"name": name, "mimeType": "application/octet-stream", "buffer": docx_bytes(text)},
    )
    page.wait_for_function("/Indexed/.test(document.querySelector('#upload-status').textContent)")


def test_the_app_field_comes_back_after_a_manual_upload(live_app, page, monkeypatch):
    _, base_url = live_app
    _fill_admin_upload_form(page, base_url, monkeypatch)
    _admin_upload_through_the_form(page, "guide.docx", "loan guide")

    # The form is cleared for the next upload, so its type is no longer Manual.
    assert page.input_value("#upload-source-type") == ""
    assert page.is_visible("#upload-app-code")


def test_saving_an_edit_says_it_was_saved(live_app, page, monkeypatch):
    _, base_url = live_app
    _fill_admin_upload_form(page, base_url, monkeypatch)
    _admin_upload_through_the_form(page, "guide.docx", "loan guide")
    page.click("#open-docs-modal")
    page.click('#docs-tbody [data-action="edit"]')

    page.locator('tr.edit-row [data-action="save-edit"]').click()

    page.locator("#docs-notice").filter(has_text="Saved guide.docx.").wait_for()


def test_the_delete_prompt_names_the_entrys_labels(live_app, page, monkeypatch):
    _, base_url = live_app
    _fill_admin_upload_form(page, base_url, monkeypatch)
    _admin_upload_through_the_form(page, "guide.docx", "loan guide")
    page.click("#open-docs-modal")
    prompts = []
    page.on("dialog", lambda d: (prompts.append(d.message), d.dismiss()))

    page.click('#docs-tbody [data-action="delete"]')

    assert prompts == ['Delete "guide.docx" (manual · v1 · loan)?']


def test_approving_an_upload_says_it_was_indexed(live_app, page, monkeypatch):
    app, base_url = live_app
    monkeypatch.setattr(config, "FUNCTIONALITIES", ["loan"])
    monkeypatch.setattr(config, "VERSIONS", ["v1"])
    app.test_client().post(
        "/upload",
        data={
            "file": (io.BytesIO(docx_bytes("yak facts")), "yak.docx"),
            "source_type": "manual", "version": "v1", "functionality": "loan",
            "requester": "staff",
        },
        content_type="multipart/form-data",
    ).get_data()
    _log_in(page, base_url)
    _open(page, base_url)

    page.click('#pending-list [data-action="approve"]')
    page.click('#pending-list [data-action="confirm-approve"]')

    page.locator("#pending-notice").filter(has_text="Approved and indexed yak.docx.").wait_for()
    assert not page.is_visible("#pending-section")


def test_an_admin_sees_a_log_out_button_and_no_empty_upload_list(live_app, page):
    _, base_url = live_app
    _log_in(page, base_url)
    _open(page, base_url)

    assert "Log out" in page.inner_text("#admin-toggle")
    assert "No user uploads" not in page.inner_text("body")
