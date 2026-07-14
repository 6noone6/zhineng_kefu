from __future__ import annotations

from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_optional_user
from src.core.auth import create_access_token
from src.core.config import get_settings
from src.core.rate_limit import limiter
from src.db import get_db
from src.models.user import User
from src.services.user_service import get_or_create_user

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class LoginRequest(BaseModel):
    email: EmailStr
    name: str | None = Field(default=None, max_length=128)


class OAuthLoginRequest(BaseModel):
    provider: str = Field(..., pattern="^(google|github|dev)$")
    email: EmailStr
    name: str | None = None
    subject: str = Field(..., min_length=1, max_length=128)
    id_token: str | None = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict


def _user_dict(user: User) -> dict:
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "oauth_provider": user.oauth_provider,
    }


@router.post("/login", response_model=AuthResponse)
@limiter.limit("20/minute")
async def login(
    request: Request,
    body: LoginRequest,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Email login (development / passwordless demo)."""
    settings = get_settings()
    user = await get_or_create_user(db, body.email, name=body.name, oauth_provider="email")
    token = create_access_token(user.id, user.email, settings=settings)
    response.set_cookie(
        key=settings.jwt_cookie_name,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.jwt_expire_minutes * 60,
    )
    return AuthResponse(access_token=token, user=_user_dict(user))


@router.post("/oauth", response_model=AuthResponse)
@limiter.limit("20/minute")
async def oauth_login(
    request: Request,
    body: OAuthLoginRequest,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """OAuth-style login. Production should verify id_token; dev accepts trusted payload."""
    settings = get_settings()
    if settings.env == "production" and body.provider == "dev":
        raise HTTPException(status_code=400, detail="Dev provider not allowed in production")

    if body.provider == "google" and settings.oauth_google_client_id and body.id_token:
        # Placeholder: verify Google id_token with google-auth library in production
        logger.info("oauth_google_token_received", sub=body.subject)

    user = await get_or_create_user(
        db,
        body.email,
        name=body.name,
        oauth_provider=body.provider,
        oauth_sub=body.subject,
    )
    token = create_access_token(user.id, user.email, settings=settings)
    response.set_cookie(
        key=settings.jwt_cookie_name,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.jwt_expire_minutes * 60,
    )
    return AuthResponse(access_token=token, user=_user_dict(user))


@router.get("/me")
async def me(user: Annotated[User | None, Depends(get_optional_user)]):
    if not user:
        return {"authenticated": False}
    return {"authenticated": True, "user": _user_dict(user)}


@router.post("/logout")
async def logout(response: Response):
    settings = get_settings()
    response.delete_cookie(settings.jwt_cookie_name)
    return {"ok": True}
