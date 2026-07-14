from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import httpx
import structlog

from src.core.config import get_settings
from src.core.metrics import COMPLAINTS_RECORDED
from src.tools import ToolResult

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger()

COMPLAINT_STATUSES = ("Received", "InReview", "Resolved")


async def record_user_complaint(
    complaint_details: str,
    session_id: str | None = None,
    db: AsyncSession | None = None,
) -> ToolResult:
    settings = get_settings()
    record = {
        "complaint_details": complaint_details,
        "complaint_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "Received",
        "session_id": session_id,
    }

    if db is not None:
        from src.models.complaint import Complaint

        try:
            complaint = Complaint(
                session_id=session_id,
                details=complaint_details,
                status="Received",
            )
            db.add(complaint)
            await db.flush()
            record["id"] = str(complaint.id)
            record["ticket_id"] = f"TKT-{str(complaint.id).replace('-', '')[:8].upper()}"
        except Exception as exc:
            await db.rollback()
            logger.warning("complaint_db_failed", error=str(exc))

    if settings.complaint_webhook_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(settings.complaint_webhook_url, json=record)
        except Exception as exc:
            logger.warning("complaint_webhook_failed", error=str(exc))

    ticket_id = record.get("ticket_id", "TKT-PENDING")
    COMPLAINTS_RECORDED.inc()
    return ToolResult(
        success=True,
        data={
            "message": (
                f"您的投诉已记录，工单号 {ticket_id}，状态：Received。"
                f"我们会尽快处理并回复您。"
            ),
            "ticket_id": ticket_id,
            "status": "Received",
            "status_workflow": list(COMPLAINT_STATUSES),
            "record": record,
        },
    )
