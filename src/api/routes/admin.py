from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.app_state import refresh_app_retriever
from src.core.config import get_settings
from src.core.security import verify_api_key
from src.db import get_db
from src.models.complaint import Complaint
from src.models.knowledge import KnowledgeDoc
from src.rag.indexer import rebuild_index_async
from src.services.complaint_service import (
    ticket_id_for,
    update_complaint_status as apply_complaint_status,
)

logger = structlog.get_logger()
router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


class ComplaintStatusUpdate(BaseModel):
    status: str = Field(..., pattern="^(Received|InReview|Resolved)$")


@router.get("/knowledge")
async def list_knowledge(
  _: Annotated[str, Depends(verify_api_key)],
  db: Annotated[AsyncSession, Depends(get_db)],
):
    settings = get_settings()
    files: list[dict] = []
    if settings.knowledge_dir.exists():
        for path in sorted(settings.knowledge_dir.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            stat = path.stat()
            files.append(
                {
                    "filename": path.name,
                    "size_bytes": stat.st_size,
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                }
            )

    result = await db.execute(select(KnowledgeDoc).order_by(KnowledgeDoc.indexed_at.desc()))
    docs = result.scalars().all()
    return {
        "files": files,
        "indexed_docs": [
            {
                "filename": d.filename,
                "chunk_count": d.chunk_count,
                "indexed_at": d.indexed_at.isoformat() if d.indexed_at else None,
            }
            for d in docs
        ],
    }


@router.delete("/knowledge/{filename}")
async def delete_knowledge_file(
    filename: str,
    request: Request,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    settings = get_settings()
    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename")

    target = settings.knowledge_dir / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    target.unlink()
    chunk_count = await rebuild_index_async(settings)
    refresh_app_retriever(request.app)
    logger.info("admin_knowledge_deleted", filename=safe_name, chunk_count=chunk_count)
    return {"filename": safe_name, "chunk_count": chunk_count, "message": "Deleted and reindexed"}


@router.get("/complaints")
async def list_complaints(
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = 50,
    offset: int = 0,
):
    limit = min(max(limit, 1), 200)
    offset = max(offset, 0)
    result = await db.execute(
        select(Complaint).order_by(Complaint.created_at.desc()).limit(limit).offset(offset)
    )
    items = result.scalars().all()
    return {
        "items": [
            {
                "id": c.id,
                "session_id": c.session_id,
                "details": c.details,
                "status": c.status,
                "ticket_id": ticket_id_for(c.id),
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in items
        ],
        "limit": limit,
        "offset": offset,
    }


@router.patch("/complaints/{complaint_id}")
async def update_complaint_status(
    complaint_id: str,
    body: ComplaintStatusUpdate,
    _: Annotated[str, Depends(verify_api_key)],
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        complaint = await apply_complaint_status(db, complaint_id, body.status)
    except LookupError:
        raise HTTPException(status_code=404, detail="Complaint not found") from None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "id": complaint.id,
        "status": complaint.status,
        "ticket_id": ticket_id_for(complaint.id),
    }
