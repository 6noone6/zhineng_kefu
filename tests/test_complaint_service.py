import pytest

from src.services.complaint_service import can_transition


def test_complaint_transitions_received():
    assert can_transition("Received", "InReview")
    assert can_transition("Received", "Resolved")
    assert not can_transition("Received", "Invalid")


def test_complaint_transitions_in_review():
    assert can_transition("InReview", "Resolved")
    assert can_transition("InReview", "Received")


def test_complaint_transitions_resolved_reopen():
    assert can_transition("Resolved", "InReview")
    assert not can_transition("Resolved", "Received")


@pytest.mark.asyncio
async def test_update_complaint_status_invalid_transition(db_session):
    from src.models.complaint import Complaint
    from src.services.complaint_service import update_complaint_status

    complaint = Complaint(details="test complaint", status="Resolved")
    db_session.add(complaint)
    await db_session.flush()

    with pytest.raises(ValueError, match="Cannot transition"):
        await update_complaint_status(db_session, complaint.id, "Received")
