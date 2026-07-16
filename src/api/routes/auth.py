from __future__ import annotations

import asyncio
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_optional_user
from src.core.auth import create_access_token
from src.core.config import get_settings
from src.core.rate_limit import limiter
from src.core.security import passwordless_login_allowed
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


def _set_auth_cookie(response: Response, token: str) -> None:
    settings = get_settings()
    response.set_cookie(
        key=settings.jwt_cookie_name,
        value=token,
        httponly=True,
        samesite="lax",
        max_age=settings.jwt_expire_minutes * 60,
        secure=settings.env == "production",
    )


def _verify_google_id_token_sync(token: str, client_id: str) -> dict:
    """Validate Google ID token locally via google-auth (no network round-trip to tokeninfo)."""
    from google.auth.transport import requests as google_requests
    from google.oauth2 import id_token as google_id_token

    return google_id_token.verify_oauth2_token(
        token,
        google_requests.Request(),
        audience=client_id,
    )


async def _verify_google_id_token(id_token: str, client_id: str) -> dict:
    """Validate Google ID token; requires OAUTH_GOOGLE_CLIENT_ID."""
    try:
        payload = await asyncio.to_thread(_verify_google_id_token_sync, id_token, client_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("google_id_token_verify_failed", error=str(exc))
        raise HTTPException(status_code=400, detail="Invalid Google id_token") from exc

    if payload.get("iss") not in {"accounts.google.com", "https://accounts.google.com"}:
        raise HTTPException(status_code=400, detail="Invalid Google id_token issuer")
    if payload.get("email_verified") is False:
        raise HTTPException(status_code=400, detail="Google email not verified")
    return payload


@router.post("/login", response_model=AuthResponse)
@limiter.limit("20/minute")
async def login(
    request: Request,
    body: LoginRequest,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Email login — development / passwordless demo only."""
    settings = get_settings()
    if not passwordless_login_allowed(settings):
        raise HTTPException(
            status_code=403,
            detail="Passwordless email login is disabled in production. Use OAuth.",
        )
    user = await get_or_create_user(db, body.email, name=body.name, oauth_provider="email")
    token = create_access_token(user.id, user.email, settings=settings)
    _set_auth_cookie(response, token)
    return AuthResponse(access_token=token, user=_user_dict(user))


@router.post("/oauth", response_model=AuthResponse)
@limiter.limit("20/minute")
async def oauth_login(
    request: Request,
    body: OAuthLoginRequest,
    response: Response,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """OAuth login. Production / configured Google client verify id_token; else trusted payload (dev)."""
    settings = get_settings()
    if settings.env == "production" and body.provider == "dev":
        raise HTTPException(status_code=400, detail="Dev provider not allowed in production")

    email = body.email
    name = body.name
    subject = body.subject

    if body.provider == "google":
        client_id = (settings.oauth_google_client_id or "").strip()
        # Production always requires verification; with client_id set, always require it (incl. dev).
        require_verify = settings.env == "production" or bool(client_id)
        if require_verify:
            if not body.id_token:
                raise HTTPException(status_code=400, detail="Google id_token is required")
            if not client_id:
                raise HTTPException(
                    status_code=503,
                    detail="OAUTH_GOOGLE_CLIENT_ID is not configured",
                )
            claims = await _verify_google_id_token(body.id_token, client_id)
            email = claims.get("email") or email
            subject = claims.get("sub") or subject
            name = claims.get("name") or name
    elif settings.env == "production":
        raise HTTPException(
            status_code=400,
            detail=f"OAuth provider '{body.provider}' is not supported in production",
        )

    user = await get_or_create_user(
        db,
        email,
        name=name,
        oauth_provider=body.provider,
        oauth_sub=subject,
    )
    token = create_access_token(user.id, user.email, settings=settings)
    _set_auth_cookie(response, token)
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
