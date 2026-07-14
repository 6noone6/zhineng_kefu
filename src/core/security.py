from __future__ import annotations

import secrets
import uuid

import structlog
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from src.core.config import Settings, get_settings

logger = structlog.get_logger()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

DEFAULT_API_KEY = "change-me-in-production"


def check_api_key(provided: str | None, settings: Settings | None = None) -> bool:
    settings = settings or get_settings()
    expected = settings.api_key
    if not expected:
        return settings.env == "development"
    if not provided:
        return False
    if secrets.compare_digest(provided, expected):
        return True
    previous = settings.api_key_previous
    if previous and secrets.compare_digest(provided, previous):
        return True
    return False


def validate_settings_on_startup(settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if settings.env != "production":
        return
    if not settings.api_key:
        raise RuntimeError("API_KEY must be set when ENV=production")
    if secrets.compare_digest(settings.api_key, DEFAULT_API_KEY):
        raise RuntimeError("API_KEY must not use the default value in production")


async def verify_api_key(api_key: str | None = Security(api_key_header)) -> str:
    settings = get_settings()
    if not settings.api_key:
        if settings.env == "production":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="API authentication is not configured",
            )
        return ""
    if not check_api_key(api_key, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return api_key or ""


def verify_ws_api_key(api_key: str | None) -> None:
    settings = get_settings()
    if not settings.api_key:
        if settings.env == "production":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API authentication is not configured",
            )
        return
    if not check_api_key(api_key, settings):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )


def validate_session_id(session_id: str) -> bool:
    try:
        uuid.UUID(session_id)
        return True
    except (ValueError, AttributeError, TypeError):
        return False
