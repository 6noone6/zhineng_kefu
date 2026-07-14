from __future__ import annotations

import structlog
from fastapi import FastAPI

from src.core.config import get_settings
from src.rag.retriever import build_retriever

logger = structlog.get_logger()


def refresh_app_retriever(app: FastAPI) -> None:
    """Rebuild BM25 + Chroma (synced) and hot-swap in-memory retriever."""
    settings = get_settings()
    retriever = build_retriever(settings)
    app.state.retriever = retriever
    app.state.agent_service.retriever = retriever
    logger.info("retriever_refreshed", retriever_type=settings.retriever_type)
