from backend.routers.auth import router as auth_router
from backend.routers.chat import router as chat_router
from backend.routers.documents import router as documents_router
from backend.routers.sessions import router as sessions_router

__all__ = [
    "auth_router",
    "chat_router",
    "documents_router",
    "sessions_router",
]
