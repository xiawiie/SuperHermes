from backend.app import app as entrypoint_app
from backend.application.main import create_app


def test_application_main_builds_the_same_app_shape():
    app = create_app()

    assert app.title == entrypoint_app.title
    assert any(route.path == "/chat" for route in app.routes)
