from fastapi import APIRouter

from backend.routers.auth import router as auth_router
from backend.routers.chat import router as chat_router
from backend.routers.documents import router as documents_router
from backend.routers.sessions import router as sessions_router

router = APIRouter()
router.include_router(auth_router)
router.include_router(chat_router)
router.include_router(sessions_router)
router.include_router(documents_router)
