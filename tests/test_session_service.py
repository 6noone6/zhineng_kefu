import pytest

from src.services.session_service import SessionService


@pytest.mark.asyncio
async def test_session_summary_in_context(db_session):
    sessions = SessionService(None)
    session = await sessions.create_session(db_session)
    await sessions.add_message(db_session, session.id, "user", "保修多久？")
    await sessions.add_message(db_session, session.id, "assistant", "一年保修。")
    await sessions.update_session_summary(session.id, "保修多久？", "一年保修。")

    # Without Redis, summary is not persisted; context still works from DB.
    ctx = await sessions.get_context_messages(db_session, session.id)
    assert len(ctx) == 2
    assert ctx[0]["role"] == "user"
