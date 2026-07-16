"""Complaint ticket workflow with validated status transitions."""

from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.models.complaint import Complaint
from src.utils.http_client import get_http_client

logger = structlog.get_logger()

COMPLAINT_STATUSES = ("Received", "InReview", "Resolved")

ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "Received": frozenset({"InReview", "Resolved"}),
    "InReview": frozenset({"Resolved", "Received"}),
    "Resolved": frozenset({"InReview"}),
}


def ticket_id_for(complaint_id: str) -> str:
    return f"TKT-{complaint_id.replace('-', '')[:8].upper()}"


def can_transition(current: str, new_status: str) -> bool:
    if new_status not in COMPLAINT_STATUSES:
        return False
    if current == new_status:
        return True
    return new_status in ALLOWED_TRANSITIONS.get(current, frozenset())


async def _notify_status_change(complaint: Complaint, previous: str) -> None:
    settings = get_settings()
    if not settings.complaint_webhook_url:
        return
    payload = {
        "event": "complaint_status_changed",
        "id": complaint.id,
        "ticket_id": ticket_id_for(complaint.id),
        "previous_status": previous,
        "status": complaint.status,
        "session_id": complaint.session_id,
        "details": complaint.details,
    }
    try:
        client = get_http_client(timeout=10.0)
        await client.post(settings.complaint_webhook_url, json=payload, timeout=10.0)
    except Exception as exc:
        logger.warning("complaint_status_webhook_failed", error=str(exc))


async def update_complaint_status(
    db: AsyncSession,
    complaint_id: str,
    new_status: str,
) -> Complaint:
    result = await db.execute(select(Complaint).where(Complaint.id == complaint_id))
    complaint = result.scalar_one_or_none()
    if not complaint:
        raise LookupError("Complaint not found")

    previous = complaint.status
    if not can_transition(previous, new_status):
        raise ValueError(
            f"Cannot transition from {previous} to {new_status}. "
            f"Allowed: {sorted(ALLOWED_TRANSITIONS.get(previous, frozenset()))}"
        )

    if previous != new_status:
        complaint.status = new_status
        await db.flush()
        await _notify_status_change(complaint, previous)
        logger.info(
            "complaint_status_updated",
            complaint_id=complaint_id,
            previous=previous,
            status=new_status,
        )

    return complaint
