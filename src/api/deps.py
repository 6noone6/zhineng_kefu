from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.auth import decode_access_token, extract_bearer_token
from src.db import get_db
from src.models.user import User
from src.services.agent_service import AgentService
from src.services.session_service import SessionService
from src.services.user_service import get_user_by_id


def get_agent_service(request: Request) -> AgentService:
    return request.app.state.agent_service


def get_session_service(request: Request) -> SessionService:
    return request.app.state.session_service


async def get_optional_user(
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User | None:
    token = extract_bearer_token(request)
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except HTTPException:
        return None
    user_id = payload.get("sub")
    if not user_id:
        return None
    return await get_user_by_id(db, user_id)


async def get_current_user(
    user: Annotated[User | None, Depends(get_optional_user)],
) -> User:
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    return user
