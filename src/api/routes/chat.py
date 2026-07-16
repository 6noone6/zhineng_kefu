from __future__ import annotations

import asyncio
import json
import time
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.websockets import WebSocketState

from src.api.deps import get_agent_service, get_optional_user, get_session_service
from src.core.auth import try_decode_token
from src.core.config import get_settings
from src.core.input_guard import validate_user_message
from src.core.metrics import CHAT_REQUESTS, RESPONSE_LATENCY
from src.core.rate_limit import limiter
from src.core.security import validate_session_id
from src.db import get_db
from src.models.session import ChatSession
from src.models.user import User
from src.redis_client import get_redis
from src.services.agent_service import AgentService
from src.services.session_service import SessionService
from src.utils.text import sanitize_assistant_reply

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["chat"])

_WS_RATE_LIMIT = 30
_WS_RATE_WINDOW = 60
_ANON_SIDS_COOKIE = "kefu_anon_sids"


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tool_name: str | None = None
    citations: list[str] = []
    tools_used: list[str] = []


class SessionResponse(BaseModel):
    id: str
    status: str
    messages: list[dict]


def _assert_session_owner(session: ChatSession, user: User | None) -> None:
    """Deny cross-user access when the session is bound to an owner."""
    if session.user_id and (user is None or user.id != session.user_id):
        raise HTTPException(status_code=403, detail="Not allowed to access this session")


def _anon_sids_from_cookie(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [s.strip() for s in raw.split(",") if s.strip() and validate_session_id(s.strip())]


def _set_anon_session_cookie(response: Response, session_id: str, request: Request) -> None:
    existing = _anon_sids_from_cookie(request.cookies.get(_ANON_SIDS_COOKIE))
    merged = [session_id] + [s for s in existing if s != session_id]
    merged = merged[:30]
    settings = get_settings()
    response.set_cookie(
        key=_ANON_SIDS_COOKIE,
        value=",".join(merged),
        httponly=True,
        samesite="lax",
        max_age=settings.session_redis_ttl,
        secure=settings.env == "production",
    )


def _assert_anon_history_access(request: Request, session: ChatSession) -> None:
    """In production, anonymous history requires matching session cookie."""
    if session.user_id:
        return
    settings = get_settings()
    if settings.env != "production":
        return
    allowed = set(_anon_sids_from_cookie(request.cookies.get(_ANON_SIDS_COOKIE)))
    if session.id not in allowed:
        raise HTTPException(status_code=403, detail="Not allowed to access this session")


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


async def _resolve_websocket_session(
    db: AsyncSession,
    sessions: SessionService,
    *,
    client_session_id: str | None,
    connection_session_id: str | None,
    user_id: str | None,
) -> tuple[str, bool]:
    """Resolve an existing session, or create-if-missing for a valid client UUID."""
    candidates: list[str] = []
    if client_session_id and validate_session_id(client_session_id):
        candidates.append(client_session_id)
    if connection_session_id and validate_session_id(connection_session_id):
        if connection_session_id not in candidates:
            candidates.append(connection_session_id)

    for sid in candidates:
        session = await sessions.get_session(db, sid)
        if not session:
            continue
        if session.user_id and user_id and session.user_id != user_id:
            continue
        if session.user_id and not user_id:
            continue
        return sid, False

    # Align with REST: reuse the client's UUID when it is valid but unknown.
    preferred_id = (
        client_session_id
        if client_session_id and validate_session_id(client_session_id)
        else None
    )
    session = await sessions.create_session(
        db, user_id=user_id, session_id=preferred_id
    )
    await db.flush()
    return session.id, True


def _client_ip(websocket: WebSocket) -> str:
    settings = get_settings()
    if settings.trusted_proxy:
        forwarded = websocket.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return websocket.client.host if websocket.client else "unknown"


async def _check_ws_rate(ip: str) -> bool:
    """Redis sliding-window rate limit; falls back to allow if Redis is down in dev."""
    redis = await get_redis()
    key = f"ws:rate:{ip}"
    if redis is None:
        return True
    try:
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, _WS_RATE_WINDOW)
        return count <= _WS_RATE_LIMIT
    except Exception as exc:
        logger.warning("ws_rate_redis_failed", error=str(exc))
        return True


async def _ws_send_json(websocket: WebSocket, payload: dict) -> bool:
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


def _ws_auth_payload(websocket: WebSocket) -> dict | None:
    """Prefer HttpOnly cookie; query ?token= kept only as deprecated fallback."""
    settings = get_settings()
    cookie = websocket.cookies.get(settings.jwt_cookie_name)
    payload = try_decode_token(cookie)
    if payload:
        return payload
    return try_decode_token(websocket.query_params.get("token"))


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("30/minute")
async def chat(
    request: Request,
    body: ChatRequest,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
    agent: Annotated[AgentService, Depends(get_agent_service)],
    sessions: Annotated[SessionService, Depends(get_session_service)],
    user: Annotated[User | None, Depends(get_optional_user)],
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
            # Create-if-missing for valid unknown UUIDs (aligns with WS).
            session = await sessions.create_session(
                db,
                user_id=user.id if user else None,
                session_id=body.session_id,
            )
        else:
            _assert_session_owner(session, user)
        session_id = session.id
    else:
        session = await sessions.create_session(db, user_id=user.id if user else None)
        session_id = session.id

    user_id, user_email = await _resolve_user_context(db, sessions, session_id, user)
    await sessions.add_message(db, session_id, "user", message)
    await db.commit()

    try:
        start = time.perf_counter()
        agent_response = await agent.process(
            message,
            session_id=session_id,
            db=None,
            sessions=sessions,
            user_id=user_id,
            user_email=user_email,
        )
        RESPONSE_LATENCY.labels(method="rest").observe(time.perf_counter() - start)
    except Exception:
        logger.exception("chat_process_failed", session_id=session_id)
        raise HTTPException(status_code=500, detail="Failed to process message")

    clean_answer = sanitize_assistant_reply(agent_response.answer)
    await sessions.add_message(
        db,
        session_id,
        "assistant",
        clean_answer,
        tool_calls={"tools": agent_response.tools_used} if agent_response.tools_used else (
            {"tool": agent_response.tool_name} if agent_response.tool_name else None
        ),
        citations=agent_response.citations or None,
    )
    await sessions.update_session_summary(session_id, message, clean_answer)

    if not user:
        _set_anon_session_cookie(response, session_id, request)

    return ChatResponse(
        session_id=session_id,
        answer=clean_answer,
        tool_name=agent_response.tool_name,
        citations=agent_response.citations,
        tools_used=agent_response.tools_used,
    )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
@limiter.limit("60/minute")
async def get_session_history(
    request: Request,
    session_id: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    sessions: Annotated[SessionService, Depends(get_session_service)],
    user: Annotated[User | None, Depends(get_optional_user)],
):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = await sessions.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, user)
    _assert_anon_history_access(request, session)

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
    user: Annotated[User | None, Depends(get_optional_user)],
):
    if not validate_session_id(session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = await sessions.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    _assert_session_owner(session, user)
    _assert_anon_history_access(request, session)

    deleted = await sessions.delete_session(db, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True, "session_id": session_id}


@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    # Public chat WS: JWT via HttpOnly cookie (preferred) or deprecated ?token=.
    await websocket.accept()
    app = websocket.app
    agent: AgentService = app.state.agent_service
    sessions: SessionService = app.state.session_service

    from src.db import async_session_factory

    session_id: str | None = None
    client_ip = _client_ip(websocket)
    ws_payload = _ws_auth_payload(websocket)
    ws_user_id = ws_payload.get("sub") if ws_payload else None
    ws_user_email = ws_payload.get("email") if ws_payload else None

    inbound: asyncio.Queue[str | None] = asyncio.Queue()
    cancel_event = asyncio.Event()

    async def _inbound_reader() -> None:
        try:
            while True:
                raw = await websocket.receive_text()
                await inbound.put(raw)
        except WebSocketDisconnect:
            await inbound.put(None)
        except Exception:
            logger.exception("ws_reader_failed")
            await inbound.put(None)

    reader_task = asyncio.create_task(_inbound_reader())

    try:
        while True:
            raw = await inbound.get()
            if raw is None:
                break

            if not await _check_ws_rate(client_ip):
                if not await _ws_send_json(
                    websocket,
                    {"type": "error", "error": "Rate limit exceeded. Please slow down."},
                ):
                    break
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                if not await _ws_send_json(
                    websocket, {"type": "error", "error": "Invalid JSON payload"}
                ):
                    break
                continue

            # Handle cancel before requiring a non-empty message / counting a chat request.
            if data.get("type") == "cancel":
                cancel_event.set()
                continue

            CHAT_REQUESTS.labels(method="websocket").inc()

            message = data.get("message", "").strip()
            if not message:
                continue

            try:
                message = validate_user_message(message)
            except ValueError as exc:
                if not await _ws_send_json(websocket, {"type": "error", "error": str(exc)}):
                    break
                continue

            cancel_event.clear()

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
                await db.commit()

            answer = ""
            tool_name = None
            citations: list[str] = []
            tools_used: list[str] = []
            client_disconnected = False
            cancelled = False
            deferred_inbound: list[str] = []
            stream = agent.process_stream(
                message,
                session_id=session_id,
                db=None,
                sessions=sessions,
                user_id=ws_user_id,
                user_email=ws_user_email,
                cancel_event=cancel_event,
            )
            next_event_task: asyncio.Task | None = None
            next_inbound_task: asyncio.Task | None = None
            try:
                start = time.perf_counter()
                agen = stream.__aiter__()
                next_event_task = asyncio.create_task(agen.__anext__())
                next_inbound_task = asyncio.create_task(inbound.get())

                while True:
                    done, _pending = await asyncio.wait(
                        {next_event_task, next_inbound_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    if next_inbound_task in done:
                        inbound_raw = next_inbound_task.result()
                        if inbound_raw is None:
                            client_disconnected = True
                            cancel_event.set()
                            next_event_task.cancel()
                            break
                        try:
                            inbound_data = json.loads(inbound_raw)
                        except json.JSONDecodeError:
                            next_inbound_task = asyncio.create_task(inbound.get())
                            continue

                        if inbound_data.get("type") == "cancel":
                            cancelled = True
                            cancel_event.set()
                            next_event_task.cancel()
                            break

                        # Defer non-cancel frames until after this stream finishes.
                        deferred_inbound.append(inbound_raw)
                        next_inbound_task = asyncio.create_task(inbound.get())
                        continue

                    try:
                        event = next_event_task.result()
                    except StopAsyncIteration:
                        break
                    except asyncio.CancelledError:
                        cancelled = True
                        break
                    except Exception:
                        raise

                    if cancel_event.is_set():
                        cancelled = True
                        break

                    if event["type"] == "chunk":
                        if not await _ws_send_json(
                            websocket,
                            {"type": "chunk", "content": event["content"]},
                        ):
                            client_disconnected = True
                            cancel_event.set()
                            break
                    elif event["type"] == "done":
                        answer = event.get("answer", "")
                        tool_name = event.get("tool_name")
                        citations = event.get("citations") or []
                        tools_used = event.get("tools_used") or []

                    next_event_task = asyncio.create_task(agen.__anext__())

                RESPONSE_LATENCY.labels(method="websocket").observe(
                    time.perf_counter() - start
                )
            except WebSocketDisconnect:
                client_disconnected = True
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                logger.exception("ws_chat_process_failed", session_id=session_id)
                if not await _ws_send_json(
                    websocket,
                    {"type": "error", "error": "Failed to process message"},
                ):
                    break
                continue
            finally:
                for task in (next_event_task, next_inbound_task):
                    if task is not None and not task.done():
                        task.cancel()
                        try:
                            await task
                        except (asyncio.CancelledError, Exception):
                            pass
                await stream.aclose()
                for item in deferred_inbound:
                    await inbound.put(item)

            if client_disconnected:
                break

            if cancelled or cancel_event.is_set():
                await _ws_send_json(
                    websocket,
                    {"type": "cancelled", "session_id": session_id},
                )
                continue

            answer = sanitize_assistant_reply(answer)
            async with async_session_factory() as db:
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
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except (asyncio.CancelledError, Exception):
            pass
