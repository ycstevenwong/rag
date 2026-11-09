"""Running `python app.py` must not expose the Werkzeug debugger by default."""
from __future__ import annotations

import runpy
from pathlib import Path

import flask
import pytest

import config

APP_PY = str(Path(__file__).resolve().parent.parent / "app.py")


def _debug_value_passed_to_run(monkeypatch, app_module) -> bool:
    captured = {}
    monkeypatch.setattr(flask.Flask, "run", lambda self, **kwargs: captured.update(kwargs))
    runpy.run_path(APP_PY, run_name="__main__")
    return captured["debug"]


def test_debug_is_off_by_default(app_module, monkeypatch):
    monkeypatch.setattr(config, "FLASK_DEBUG", False, raising=False)

    assert _debug_value_passed_to_run(monkeypatch, app_module) is False


def test_debug_can_be_enabled_by_config(app_module, monkeypatch):
    monkeypatch.setattr(config, "FLASK_DEBUG", True, raising=False)

    assert _debug_value_passed_to_run(monkeypatch, app_module) is True


@pytest.mark.parametrize(
    "raw, expected",
    [("1", True), ("true", True), ("0", False), ("", False), (None, False)],
)
def test_env_flag_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("SOME_FLAG", raising=False)
    else:
        monkeypatch.setenv("SOME_FLAG", raw)

    assert config.env_flag("SOME_FLAG") is expected


@pytest.mark.parametrize("secure", [True, False])
def test_session_cookie_secure_follows_config(app_module, monkeypatch, secure):
    monkeypatch.setattr(config, "SESSION_COOKIE_SECURE", secure, raising=False)

    assert app_module.create_app().config["SESSION_COOKIE_SECURE"] is secure
