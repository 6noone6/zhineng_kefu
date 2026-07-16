from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text

from src.core.config import get_settings
from src.core.security import check_api_key
from src.db import engine
from src.redis_client import get_redis
from src.utils.http_client import get_http_client

logger = structlog.get_logger()
router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "zhineng-kefu"}


def _require_deep_health_auth(request: Request) -> None:
    settings = get_settings()
    if settings.env != "production":
        return
    api_key = request.headers.get("X-API-Key")
    if not check_api_key(api_key, settings):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@router.get("/health/deep")
async def health_deep(request: Request):
    _require_deep_health_auth(request)
    settings = get_settings()
    components: dict[str, dict] = {}
    overall = "ok"

    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        components["postgres"] = {"status": "ok"}
    except Exception as exc:
        logger.warning("health_postgres_failed", error=str(exc))
        components["postgres"] = {"status": "down", "error": "unavailable"}
        overall = "degraded"

    try:
        redis = await get_redis()
        if redis is None:
            components["redis"] = {
                "status": "degraded" if settings.env != "production" else "down",
                "error": "unavailable",
            }
            overall = "degraded"
        else:
            pong = await redis.ping()
            components["redis"] = {"status": "ok" if pong else "down"}
            if not pong:
                overall = "degraded"
    except Exception as exc:
        logger.warning("health_redis_failed", error=str(exc))
        components["redis"] = {"status": "down", "error": "unavailable"}
        overall = "degraded"

    components["kimi"] = {
        "status": "configured" if settings.moonshot_api_key else "not_configured",
    }
    if not settings.moonshot_api_key:
        overall = "degraded"

    components["rag"] = {
        "backend": settings.rag_backend,
        "qwen_remote": bool(settings.qwen_inference_url),
    }

    if settings.rag_backend == "local" and settings.qwen_inference_url:
        try:
            client = get_http_client(timeout=3.0)
            resp = await client.get(
                f"{settings.qwen_inference_url.rstrip('/')}/health",
                timeout=3.0,
            )
            components["qwen_remote"] = {
                "status": "ok" if resp.status_code == 200 else "degraded",
            }
            if resp.status_code != 200:
                overall = "degraded"
        except Exception as exc:
            logger.warning("health_qwen_remote_failed", error=str(exc))
            components["qwen_remote"] = {"status": "down", "error": "unreachable"}
            overall = "degraded"

    payload = {
        "status": overall,
        "service": "zhineng-kefu",
        "components": components,
    }
    # Only expose env outside production deep checks (admin already authenticated).
    if settings.env != "production":
        payload["env"] = settings.env
    return payload
