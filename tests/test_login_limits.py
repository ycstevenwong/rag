"""Login must resist guessing without locking out everyone behind one proxy."""
from __future__ import annotations

import time

import pytest
from werkzeug.security import generate_password_hash

import config

_HASH = generate_password_hash("right-password", method="pbkdf2:sha256:1000")


@pytest.fixture
def admin_on(app_module, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", _HASH)
    monkeypatch.setattr(config, "FLASK_SECRET_KEY", "a-private-test-key")
    app_module._login_failures.clear()
    yield app_module
    app_module._login_failures.clear()


def _login(client, *, username="admin", password="wrong", ip="10.0.0.1", forwarded=None):
    headers = {"X-Forwarded-For": forwarded} if forwarded else {}
    return client.post(
        "/admin/login",
        json={"username": username, "password": password},
        environ_base={"REMOTE_ADDR": ip},
        headers=headers,
    )


def test_password_hash_is_checked_even_for_a_wrong_username(admin_on, monkeypatch):
    """Skipping the slow hash would reveal which usernames are valid by timing."""
    real_check = admin_on.check_password_hash
    calls = []

    def spy(stored, password):
        calls.append(password)
        return real_check(stored, password)

    monkeypatch.setattr(admin_on, "check_password_hash", spy)
    client = admin_on.create_app().test_client()

    assert _login(client, username="not-admin").status_code == 401
    assert calls == ["wrong"]


def test_correct_login_succeeds(admin_on):
    client = admin_on.create_app().test_client()

    assert _login(client, password="right-password").json == {"is_admin": True}


def test_right_password_with_wrong_username_fails(admin_on):
    client = admin_on.create_app().test_client()

    assert _login(client, username="other", password="right-password").status_code == 401


def test_non_ascii_username_fails_cleanly(admin_on):
    client = admin_on.create_app().test_client()

    assert _login(client, username="ädmin").status_code == 401


@pytest.mark.parametrize(
    "body",
    [
        {"username": 5, "password": "x"},
        {"username": "admin", "password": 5},
        {"username": ["admin"], "password": {"a": 1}},
        [1, 2],
    ],
)
def test_malformed_login_is_a_failed_attempt_not_a_crash(admin_on, body):
    client = admin_on.create_app().test_client()

    assert client.post("/admin/login", json=body).status_code == 401
    assert len(admin_on._login_failures) == 1


def test_five_failures_block_the_address(admin_on):
    client = admin_on.create_app().test_client()
    for _ in range(5):
        assert _login(client).status_code == 401

    assert _login(client, password="right-password").status_code == 429


def test_behind_a_trusted_proxy_each_client_has_its_own_limit(admin_on, monkeypatch):
    monkeypatch.setattr(config, "TRUSTED_PROXY_COUNT", 1, raising=False)
    client = admin_on.create_app().test_client()
    for _ in range(5):
        _login(client, ip="10.0.0.9", forwarded="203.0.113.1")

    assert _login(client, ip="10.0.0.9", forwarded="203.0.113.1").status_code == 429
    assert (
        _login(client, ip="10.0.0.9", forwarded="203.0.113.2", password="right-password").status_code
        == 200
    )


def test_forwarded_header_is_ignored_without_a_trusted_proxy(admin_on, monkeypatch):
    """Otherwise anyone could dodge the limit by sending a different header each time."""
    monkeypatch.setattr(config, "TRUSTED_PROXY_COUNT", 0, raising=False)
    client = admin_on.create_app().test_client()
    for i in range(5):
        _login(client, forwarded=f"203.0.113.{i}")

    assert _login(client, forwarded="203.0.113.99").status_code == 429


def test_expired_failures_are_forgotten(admin_on):
    admin_on._login_failures["198.51.100.7"] = [time.time() - 10_000]
    client = admin_on.create_app().test_client()

    _login(client)

    assert "198.51.100.7" not in admin_on._login_failures
