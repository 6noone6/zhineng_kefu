from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_optional_user
from src.core.rate_limit import limiter
from src.core.security import verify_api_key, validate_session_id
from src.db import get_db
from src.models.feedback import MessageFeedback
from src.models.user import User
from src.services.session_service import SessionService
from src.api.deps import get_session_service

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1", tags=["feedback"])


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str | None = None
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = Field(default=None, max_length=1000)


@router.post("/chat/feedback")
@limiter.limit("30/minute")
async def submit_feedback(
    request: Request,
    body: FeedbackRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    sessions: Annotated[SessionService, Depends(get_session_service)],
    user: Annotated[User | None, Depends(get_optional_user)],
    _: Annotated[str, Depends(verify_api_key)],
):
    if not validate_session_id(body.session_id):
        raise HTTPException(status_code=400, detail="Invalid session_id format")

    session = await sessions.get_session(db, body.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    feedback = MessageFeedback(
        session_id=body.session_id,
        message_id=body.message_id,
        rating=body.rating,
        comment=body.comment,
    )
    db.add(feedback)
    await db.flush()

    logger.info(
        "feedback_recorded",
        session_id=body.session_id,
        rating=body.rating,
        user_id=user.id if user else None,
    )
    return {
        "ok": True,
        "feedback_id": feedback.id,
        "session_id": body.session_id,
        "rating": body.rating,
    }
