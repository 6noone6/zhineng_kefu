from __future__ import annotations

from datetime import datetime
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.rate_limit import limiter
from src.core.security import verify_api_key
from src.db import get_db
from src.core.app_state import refresh_app_retriever
from src.core.metrics import KNOWLEDGE_UPLOADS
from src.models.knowledge import KnowledgeDoc
from src.rag.indexer import rebuild_index_async, save_uploaded_file

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])


@router.post("/upload")
@limiter.limit("10/minute")
async def upload_knowledge(
    request: Request,
    file: UploadFile = File(...),
    db: Annotated[AsyncSession, Depends(get_db)] = ...,
    _: Annotated[str, Depends(verify_api_key)] = ...,
):
    content = await file.read()
    try:
        path = save_uploaded_file(file.filename or "upload.txt", content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    chunk_count = await rebuild_index_async()
    refresh_app_retriever(request.app)
    KNOWLEDGE_UPLOADS.inc()

    doc = KnowledgeDoc(
        filename=path.name,
        chunk_count=chunk_count,
        indexed_at=datetime.utcnow(),
    )
    db.add(doc)

    return {
        "filename": path.name,
        "chunk_count": chunk_count,
        "message": "Knowledge base indexed successfully",
    }


@router.post("/rebuild")
@limiter.limit("5/minute")
async def rebuild_knowledge_index(
    request: Request,
    _: Annotated[str, Depends(verify_api_key)] = None,
):
    count = await rebuild_index_async()
    refresh_app_retriever(request.app)
    return {"chunk_count": count, "message": "Index rebuilt"}
