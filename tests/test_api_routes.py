from backend.app import create_app


EXPECTED_ROUTE_MODULES = {
    "/auth/register": "backend.routers.auth",
    "/auth/login": "backend.routers.auth",
    "/auth/me": "backend.routers.auth",
    "/chat": "backend.routers.chat",
    "/chat/stream": "backend.routers.chat",
    "/sessions": "backend.routers.sessions",
    "/sessions/{session_id}": "backend.routers.sessions",
    "/documents": "backend.routers.documents",
    "/documents/upload": "backend.routers.documents",
    "/documents/{filename}": "backend.routers.documents",
}


def test_routes_are_owned_by_capability_router_modules():
    app = create_app()
    route_modules = {}
    for route in app.routes:
        path = getattr(route, "path", None)
        if path in EXPECTED_ROUTE_MODULES:
            route_modules[path] = getattr(getattr(route, "endpoint", None), "__module__", "")

    assert route_modules == EXPECTED_ROUTE_MODULES
