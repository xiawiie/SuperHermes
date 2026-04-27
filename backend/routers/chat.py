import asyncio
import json
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.security.auth import get_current_user
from backend.infra.db.models import User
from backend.contracts.schemas import ChatRequest, ChatResponse
from backend.chat.agent import chat_with_agent as run_chat, chat_with_agent_stream as run_chat_stream

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    try:
        session_id = request.session_id or "default_session"
        resp = await asyncio.to_thread(
            run_chat,
            request.message,
            current_user.username,
            session_id,
            request.context_files or [],
        )
        if isinstance(resp, dict):
            return ChatResponse(**resp)
        return ChatResponse(response=resp)
    except Exception as e:
        message = str(e)
        match = re.search(r"Error code:\s*(\d{3})", message)
        if match:
            code = int(match.group(1))
            if code == 429:
                raise HTTPException(
                    status_code=429,
                    detail=(
                        "ä¸æ¸¸æ¨¡åæå¡è§¦åéæµ/é¢åº¦éå¶ï¼?29ï¼ãè¯·æ£æ¥è´¦å·é¢åº?æ¨¡åç¶æã\n"
                        f"åå§éè¯¯ï¼{message}"
                    ),
                )
            if code in (401, 403):
                raise HTTPException(status_code=code, detail=message)
            raise HTTPException(status_code=code, detail=message)
        raise HTTPException(status_code=500, detail=message)


@router.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """è·?Agent å¯¹è¯ (æµå¼)"""

    async def event_generator():
        try:
            session_id = request.session_id or "default_session"
            async for chunk in run_chat_stream(
                request.message,
                current_user.username,
                session_id,
                bool(request.regenerate),
                request.context_files or [],
            ):
                yield chunk
        except ValueError as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
        except Exception as e:
            error_data = {"type": "error", "content": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
