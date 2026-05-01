import importlib
import sys

import pytest

from backend.app import app as entrypoint_app
from backend.application.main import create_app


def test_application_main_builds_the_same_app_shape():
    app = create_app()

    assert app.title == entrypoint_app.title
    assert any(route.path == "/chat" for route in app.routes)


def test_production_rejects_default_jwt_secret(monkeypatch):
    original_auth = sys.modules.get("backend.security.auth")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    sys.modules.pop("backend.security.auth", None)
    with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
        importlib.import_module("backend.security.auth")

    try:
        monkeypatch.setenv("JWT_SECRET_KEY", "test-secret")
        sys.modules.pop("backend.security.auth", None)
        importlib.import_module("backend.security.auth")
    finally:
        if original_auth is not None:
            sys.modules["backend.security.auth"] = original_auth
        else:
            sys.modules.pop("backend.security.auth", None)
