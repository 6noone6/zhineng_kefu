from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_agent_service, get_optional_user, get_session_service
from src.core.auth import try_decode_token
from src.core.input_guard import validate_user_message
from src.core.metrics import CHAT_REQUESTS, RESPONSE_LATENCY
from src.core.rate_limit import limiter
from src.core.security import validate_session_id, verify_api_key, verify_ws_api_key
from src.db import get_db
from src.utils.text import sanitize_assistant_reply
from src.models.user import User
from src.services.agent_service import AgentService
from src.services.session_service import SessionService

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["chat"])

_ws_message_times: dict[str, list[float]] = defaultdict(list)
_WS_RATE_LIMIT = 30
_WS_RATE_WINDOW = 60.0


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tool_name: str | None = None
    citations: list[str] = []
    tools_used: list[str] = []


async def _resolve_user_context(
    db: AsyncSession,
    sessions: SessionService,
    session_id: str,
    user: User | None,
) -> tuple[str | None, str | None]:
    user_id = user.id if user else None
    user_email = user.email if user else None
    if user_id:
        await sessions.bind_user(db, session_id, user_id)
    return user_id, user_email


class SessionResponse(BaseModel):
    id: str
    status: str
    messages: list[dict]


async def _resolve_websocket_session(
    db: AsyncSession,
    sessions: SessionService,
    *,
    client_session_id: str | None,
    connection_session_id: str | None,
    user_id: str | None,
) -> tuple[str, bool]:
    """
    Pick a valid session for this WS message. Creates a new session when the
    client or connection holds a stale/deleted session id.
    """
    candidates: list[str] = []
    if client_session_id and validate_session_id(client_session_id):
        candidates.append(client_session_id)
    if connection_session_id and validate_session_id(connection_session_id):
        if connection_session_id not in candidates:
            candidates.append(connection_session_id)

    for sid in candidates:
        if await sessions.get_session(db, sid):
            return sid, False

    session = await sessions.create_session(db, user_id=user_id)
    await db.flush()
    return session.id, True


def _check_ws_rate(ip: str) -> bool:
    now = time.monotonic()
    times = _ws_message_times[ip]
    _ws_message_times[ip] = [t for t in times if now - t < _WS_RATE_WINDOW]
    if len(_ws_message_times[ip]) >= _WS_RATE_LIMIT:
        return False
    _ws_message_times[ip].append(now)
    return True


async def _ws_send_json(websocket: WebSocket, payload: dict) -> bool:
    """Send to WebSocket client; return False if the connection is already closed."""
    if websocket.client_state != WebSocketState.CONNECTED:
        return False
    try:
        await websocket.send_json(payload)
        return True
    except WebSocketDisconnect:
        return False
    except RuntimeError as exc:
        if "close message" in str(exc).lower():
            return False
        raise


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    agent: Annotated[AgentService, Depends(get_agent_service)],
    sessions: Annotated[SessionService, Depends(get_session_service)],
    user: Annotated[User | None, Depends(get_optional_user)],
    _: Annotated[str, Depends(verify_api_key)],
):
    CHAT_REQUESTS.labels(method="rest").inc()

    if body.session_id and not validate_session_id(body.session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    try:
        message = validate_user_message(body.message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if body.session_id:
        session = await sessions.get_session(db, body.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        session_id = body.session_id
    else:
        session = await sessions.create_session(db, user_id=user.id if user else None)
        session_id = session.id

    user_id, user_email = await _resolve_user_context(db, sessions, session_id, user)

    await sessions.add_message(db, session_id, "user", message)
    try:
        start = time.perf_counter()
        response = await agent.process(
            message,
            session_id=session_id,
            db=db,
            sessions=sessions,
            user_id=user_id,
            user_email=user_email,
        )
        RESPONSE_LATENCY.labels(method="rest").observe(time.perf_counter() - start)
    except Exception:
        logger.exception("chat_process_failed", session_id=session_id)
        raise HTTPException(status_code=500, detail="Failed to process message")

    clean_answer = sanitize_assistant_reply(response.answer)
    await sessions.add_message(
        db,
        session_id,
        "assistant",
        clean_answer,
        tool_calls={"tools": response.tools_used} if response.tools_used else (
            {"tool": response.tool_name} if response.tool_name else None
        ),
        citations=response.citations or None,
    )
    await sessions.update_session_summary(session_id, message, clean_answer)

    return ChatResponse(
        session_id=session_id,
        answer=clean_answer,
        tool_name=response.tool_name,
        citations=response.citations,
        tools_used=response.tools_used,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
@limiter.limit("60/minute")
async def get_session_history(
    request: Request,
    session_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    sessions: Annotated[SessionService, Depends(get_session_service)],
    _: Annotated[str, Depends(verify_api_key)],
):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = await sessions.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = await sessions.get_messages(db, session_id)
    return SessionResponse(
        id=session.id,
        status=session.status,
        messages=[
            {
                "role": m.role,
                "content": m.content,
                "citations": m.citations,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
    )


@router.delete("/sessions/{session_id}")
@limiter.limit("30/minute")
async def delete_session_history(
    request: Request,
    session_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    sessions: Annotated[SessionService, Depends(get_session_service)],
    _: Annotated[str, Depends(verify_api_key)],
):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    deleted = await sessions.delete_session(db, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True, "session_id": session_id}


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    api_key = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key")
    try:
        verify_ws_api_key(api_key)
    except HTTPException:
        await websocket.close(code=1008, reason="Unauthorized")
        return

    await websocket.accept()
    app = websocket.app
    agent: AgentService = app.state.agent_service
    sessions: SessionService = app.state.session_service

    from src.db import async_session_factory

    session_id: str | None = None
    client_ip = websocket.client.host if websocket.client else "unknown"
    ws_token = websocket.query_params.get("token")
    ws_payload = try_decode_token(ws_token)
    ws_user_id = ws_payload.get("sub") if ws_payload else None
    ws_user_email = ws_payload.get("email") if ws_payload else None

    try:
        while True:
            raw = await websocket.receive_text()
            if not _check_ws_rate(client_ip):
                if not await _ws_send_json(
                    websocket,
                    {"type": "error", "error": "Rate limit exceeded. Please slow down."},
                ):
                    break
                continue

            CHAT_REQUESTS.labels(method="websocket").inc()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                if not await _ws_send_json(
                    websocket, {"type": "error", "error": "Invalid JSON payload"}
                ):
                    break
                continue

            message = data.get("message", "").strip()
            if not message:
                continue

            try:
                message = validate_user_message(message)
            except ValueError as exc:
                if not await _ws_send_json(websocket, {"type": "error", "error": str(exc)}):
                    break
                continue

            async with async_session_factory() as db:
                client_sid = data.get("session_id")
                if client_sid and not validate_session_id(client_sid):
                    if not await _ws_send_json(
                        websocket, {"type": "error", "error": "Invalid session_id"}
                    ):
                        break
                    continue

                session_id, is_new_session = await _resolve_websocket_session(
                    db,
                    sessions,
                    client_session_id=client_sid,
                    connection_session_id=session_id,
                    user_id=ws_user_id,
                )
                if is_new_session:
                    await db.commit()
                    if not await _ws_send_json(
                        websocket, {"type": "session", "session_id": session_id}
                    ):
                        break

                if ws_user_id:
                    await sessions.bind_user(db, session_id, ws_user_id)

                await sessions.add_message(db, session_id, "user", message)

                answer = ""
                tool_name = None
                citations: list[str] = []
                tools_used: list[str] = []
                client_disconnected = False
                try:
                    start = time.perf_counter()
                    async for event in agent.process_stream(
                        message,
                        session_id=session_id,
                        db=db,
                        sessions=sessions,
                        user_id=ws_user_id,
                        user_email=ws_user_email,
                    ):
                        if event["type"] == "chunk":
                            if not await _ws_send_json(
                                websocket,
                                {"type": "chunk", "content": event["content"]},
                            ):
                                client_disconnected = True
                                break
                        elif event["type"] == "done":
                            answer = event.get("answer", "")
                            tool_name = event.get("tool_name")
                            citations = event.get("citations") or []
                            tools_used = event.get("tools_used") or []
                    RESPONSE_LATENCY.labels(method="websocket").observe(
                        time.perf_counter() - start
                    )
                except WebSocketDisconnect:
                    client_disconnected = True
                except Exception:
                    logger.exception("ws_chat_process_failed", session_id=session_id)
                    if not await _ws_send_json(
                        websocket,
                        {"type": "error", "error": "Failed to process message"},
                    ):
                        break
                    continue

                if client_disconnected:
                    break

                answer = sanitize_assistant_reply(answer)
                await sessions.add_message(
                    db,
                    session_id,
                    "assistant",
                    answer,
                    tool_calls={"tools": tools_used} if tools_used else (
                        {"tool": tool_name} if tool_name else None
                    ),
                    citations=citations or None,
                )
                await sessions.update_session_summary(session_id, message, answer)
                await db.commit()

                if not await _ws_send_json(
                    websocket,
                    {
                        "type": "done",
                        "session_id": session_id,
                        "answer": answer,
                        "tool_name": tool_name,
                        "tools_used": tools_used,
                        "citations": citations,
                    },
                ):
                    break
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("ws_unhandled_error")
        await _ws_send_json(
            websocket, {"type": "error", "error": "Internal server error"}
        )
