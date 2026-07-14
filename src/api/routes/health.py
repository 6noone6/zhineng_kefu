from __future__ import annotations

import structlog
from fastapi import APIRouter
from sqlalchemy import text

from src.core.config import get_settings
from src.db import engine
from src.redis_client import get_redis

logger = structlog.get_logger()
router = APIRouter(tags=["health"])


@router.get("/health")
async def health():
    return {"status": "ok", "service": "zhineng-kefu"}


@router.get("/health/deep")
async def health_deep():
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
        pong = await redis.ping()
        components["redis"] = {"status": "ok" if pong else "down"}
        if not pong:
            overall = "degraded"
    except Exception as exc:
        logger.warning("health_redis_failed", error=str(exc))
        components["redis"] = {"status": "degraded", "error": "optional_unavailable"}

    components["kimi"] = {
        "status": "configured" if settings.moonshot_api_key else "not_configured",
    }
    if not settings.moonshot_api_key:
        overall = "degraded"

    components["rag"] = {
        "backend": settings.rag_backend,
        "qwen_inference_url": settings.qwen_inference_url or None,
    }

    if settings.rag_backend == "local" and settings.qwen_inference_url:
        try:
            import httpx

            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{settings.qwen_inference_url.rstrip('/')}/health")
                components["qwen_remote"] = {
                    "status": "ok" if resp.status_code == 200 else "degraded",
                }
                if resp.status_code != 200:
                    overall = "degraded"
        except Exception as exc:
            logger.warning("health_qwen_remote_failed", error=str(exc))
            components["qwen_remote"] = {"status": "down", "error": "unreachable"}
            overall = "degraded"

    return {
        "status": overall,
        "service": "zhineng-kefu",
        "env": settings.env,
        "components": components,
    }
