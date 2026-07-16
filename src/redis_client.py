from __future__ import annotations

import redis.asyncio as aioredis
import structlog

from src.core.config import get_settings

logger = structlog.get_logger()

_redis_client: aioredis.Redis | None = None
_redis_connect_failed = False


async def get_redis() -> aioredis.Redis | None:
    global _redis_client, _redis_connect_failed
    settings = get_settings()
    if _redis_client is not None:
        return _redis_client
    if _redis_connect_failed and settings.env != "production":
        return None
    try:
        client = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
        )
        await client.ping()
        _redis_client = client
        _redis_connect_failed = False
        return _redis_client
    except Exception as exc:
        _redis_client = None
        _redis_connect_failed = True
        logger.error(
            "redis_unavailable",
            error=str(exc),
            redis_url=settings.redis_url,
            env=settings.env,
        )
        if settings.env == "production":
            raise RuntimeError(
                "Redis is required in production (session cache / return workflow). "
                f"Connection failed: {exc}"
            ) from exc
        return None


async def close_redis() -> None:
    global _redis_client, _redis_connect_failed
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
    _redis_connect_failed = False
