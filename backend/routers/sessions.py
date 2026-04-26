import asyncio

from fastapi import APIRouter, Depends, HTTPException

from backend.chat.agent import storage
from backend.security.auth import get_current_user
from backend.infra.db.models import User
from backend.contracts.schemas import (
    MessageInfo,
    SessionDeleteResponse,
    SessionInfo,
    SessionListResponse,
    SessionMessagesResponse,
    SessionRenameRequest,
    SessionRenameResponse,
)

router = APIRouter()


@router.get("/sessions/{session_id}", response_model=SessionMessagesResponse)
async def get_session_messages(session_id: str, current_user: User = Depends(get_current_user)):
    """Return messages for a chat session."""
    try:
        records = await asyncio.to_thread(storage.get_session_messages, current_user.username, session_id)
        messages = [
            MessageInfo(
                type=msg["type"],
                content=msg["content"],
                timestamp=msg["timestamp"],
                rag_trace=msg.get("rag_trace"),
            )
            for msg in records
        ]
        return SessionMessagesResponse(messages=messages)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(current_user: User = Depends(get_current_user)):
    """List chat sessions for the current user."""
    try:
        session_records = await asyncio.to_thread(storage.list_session_infos, current_user.username)
        sessions = [SessionInfo(**item) for item in session_records]
        sessions.sort(key=lambda x: x.updated_at, reverse=True)
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/sessions/{session_id}", response_model=SessionRenameResponse)
async def rename_session(
    session_id: str,
    request: SessionRenameRequest,
    current_user: User = Depends(get_current_user),
):
    """Rename a chat session title."""
    try:
        result = await asyncio.to_thread(
            storage.update_session_title,
            current_user.username,
            session_id,
            request.title,
        )
        if not result:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionRenameResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}", response_model=SessionDeleteResponse)
async def delete_session(session_id: str, current_user: User = Depends(get_current_user)):
    """Delete a chat session for the current user."""
    try:
        deleted = await asyncio.to_thread(storage.delete_session, current_user.username, session_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionDeleteResponse(session_id=session_id, message="Session deleted")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
