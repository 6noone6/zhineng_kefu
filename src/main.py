from __future__ import annotations

import json
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.routes import admin, auth, chat, feedback, health, inference, knowledge, orders
from src.core.config import get_settings
from src.core.rate_limit import limiter
from src.core.security import check_api_key, validate_settings_on_startup
from src.core.tracing import setup_tracing
from src.db import engine
from src.models import Base
from src.rag.retriever import build_retriever
from src.rag.embeddings import release_ml_resources
from src.redis_client import close_redis, get_redis
from src.services.agent_service import AgentService
from src.services.llm.kimi_client import KimiClient
from src.services.llm.local_qwen import LocalQwenService
from src.services.session_service import SessionService

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    validate_settings_on_startup(settings)
    logger.info("starting_app", rag_backend=settings.rag_backend, env=settings.env)

    retriever = build_retriever(settings)
    kimi = KimiClient(settings)

    qwen = None
    if settings.rag_backend == "local":
        if settings.qwen_inference_url:
            from src.services.llm.remote_qwen import RemoteQwenClient

            qwen = RemoteQwenClient(settings)
            logger.info("qwen_remote_mode", url=settings.qwen_inference_url)
        else:
            try:
                qwen = LocalQwenService.get_instance(settings)
                qwen.load()
                await qwen.start_worker()
                app.state.qwen_service = qwen
                logger.info("qwen_model_loaded")
            except Exception as e:
                logger.warning("qwen_load_failed", error=str(e))

    redis = await get_redis()
    app.state.agent_service = AgentService(kimi, retriever, qwen)
    app.state.session_service = SessionService(redis)
    app.state.retriever = retriever

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield

    if qwen and hasattr(qwen, "shutdown"):
        await qwen.shutdown()
    elif qwen and hasattr(qwen, "unload"):
        qwen.unload()
    await close_redis()
    await engine.dispose()
    release_ml_resources()
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="智能客服 API",
        description="Production intelligent customer service system",
        version="1.0.0",
        lifespan=lifespan,
    )

    if setup_tracing(app):
        logger.info("otel_tracing_enabled", service=settings.otel_service_name)

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    @app.middleware("http")
    async def metrics_auth_middleware(request: Request, call_next):
        if request.url.path == "/metrics" and settings.metrics_require_auth:
            api_key = request.headers.get("X-API-Key")
            if not check_api_key(api_key, settings):
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"},
                )
        return await call_next(request)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(chat.router)
    app.include_router(feedback.router)
    app.include_router(orders.router)
    app.include_router(knowledge.router)
    app.include_router(admin.router)
    app.include_router(inference.router)

    @app.get("/config.js")
    async def frontend_config_js():
        payload = {
            "apiKey": settings.api_key or "",
            "wsPath": "/api/v1/ws/chat",
        }
        return Response(
            content=f"window.__KEFU_CONFIG__ = {json.dumps(payload)};",
            media_type="application/javascript",
        )

    @app.get("/metrics")
    async def metrics_endpoint():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    try:
        from pathlib import Path

        admin_dir = Path(__file__).parent.parent / "frontend-admin" / "dist"
        if admin_dir.exists():
            app.mount(
                "/admin",
                StaticFiles(directory=str(admin_dir), html=True),
                name="admin",
            )

        frontend_dir = Path(__file__).parent.parent / "frontend"
        if frontend_dir.exists():
            app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    except Exception:
        pass

    return app


app = create_app()


def run():
    import os

    import uvicorn

    settings = get_settings()
    try:
        uvicorn.run(
            "src.main:app",
            host=settings.api_host,
            port=settings.api_port,
            reload=False,
            timeout_graceful_shutdown=5,
        )
    finally:
        # PyTorch / sentence-transformers spawn non-daemon threads that block exit on Windows.
        release_ml_resources()
        os._exit(0)


if __name__ == "__main__":
    run()
