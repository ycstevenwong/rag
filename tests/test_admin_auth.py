"""Admin access must not depend on a publicly known signing key."""
from __future__ import annotations

import pytest
from flask import Flask
from flask.sessions import SecureCookieSessionInterface
from werkzeug.security import generate_password_hash

import config


def _forged_admin_cookie(secret: str) -> str:
    signer = Flask("forger")
    signer.secret_key = secret
    return SecureCookieSessionInterface().get_signing_serializer(signer).dumps({"is_admin": True})


def _enable_admin(monkeypatch) -> None:
    monkeypatch.setattr(config, "ADMIN_USERNAME", "admin")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", generate_password_hash("pw"))


def test_admin_cookie_is_ignored_when_login_is_not_configured(client, monkeypatch):
    monkeypatch.setattr(config, "ADMIN_USERNAME", "")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", "")
    client.set_cookie("session", _forged_admin_cookie(config.FLASK_SECRET_KEY))

    assert client.get("/admin/me").json["is_admin"] is False
    assert client.get("/admin/pending").status_code == 403


@pytest.mark.parametrize("secret", ["dev-secret-change-me", "change-me"])
def test_startup_refused_when_admin_login_uses_a_known_secret(app_module, monkeypatch, secret):
    _enable_admin(monkeypatch)
    monkeypatch.setattr(config, "FLASK_SECRET_KEY", secret)

    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY"):
        app_module.create_app()


@pytest.mark.parametrize("secret", ["dev-secret-change-me", "change-me"])
def test_startup_allowed_with_known_secret_when_admin_is_off(app_module, monkeypatch, secret):
    monkeypatch.setattr(config, "ADMIN_USERNAME", "")
    monkeypatch.setattr(config, "ADMIN_PASSWORD_HASH", "")
    monkeypatch.setattr(config, "FLASK_SECRET_KEY", secret)

    assert app_module.create_app().test_client().get("/admin/me").json["enabled"] is False


def test_startup_allowed_when_admin_login_has_a_private_secret(app_module, monkeypatch):
    _enable_admin(monkeypatch)
    monkeypatch.setattr(config, "FLASK_SECRET_KEY", "a-long-private-value")

    assert app_module.create_app().test_client().get("/admin/me").json["enabled"] is True
