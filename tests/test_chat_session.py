import pytest

from src.api.routes.chat import _resolve_websocket_session
from src.services.session_service import SessionService


@pytest.mark.asyncio
async def test_resolve_websocket_session_uses_client_session(db_session):
    sessions = SessionService(None)
    created = await sessions.create_session(db_session)
    sid, is_new = await _resolve_websocket_session(
        db_session,
        sessions,
        client_session_id=created.id,
        connection_session_id=None,
        user_id=None,
    )
    assert sid == created.id
    assert is_new is False


@pytest.mark.asyncio
async def test_resolve_websocket_session_creates_when_stale(db_session):
    """Valid but unknown client UUID is create-if-missing (same id)."""
    sessions = SessionService(None)
    stale_id = "550e8400-e29b-41d4-a716-446655440000"
    sid, is_new = await _resolve_websocket_session(
        db_session,
        sessions,
        client_session_id=stale_id,
        connection_session_id=stale_id,
        user_id=None,
    )
    assert is_new is True
    assert sid == stale_id
    assert await sessions.get_session(db_session, sid) is not None


@pytest.mark.asyncio
async def test_resolve_websocket_session_creates_random_without_client_id(db_session):
    sessions = SessionService(None)
    stale_connection = "550e8400-e29b-41d4-a716-446655440099"
    sid, is_new = await _resolve_websocket_session(
        db_session,
        sessions,
        client_session_id=None,
        connection_session_id=stale_connection,
        user_id=None,
    )
    assert is_new is True
    # Connection-only stale id is not reused (avoids binding to a ghost session).
    assert sid != stale_connection
    assert await sessions.get_session(db_session, sid) is not None


@pytest.mark.asyncio
async def test_resolve_websocket_session_prefers_valid_client_over_stale_connection(
    db_session,
):
    sessions = SessionService(None)
    created = await sessions.create_session(db_session)
    stale_id = "550e8400-e29b-41d4-a716-446655440099"
    sid, is_new = await _resolve_websocket_session(
        db_session,
        sessions,
        client_session_id=created.id,
        connection_session_id=stale_id,
        user_id=None,
    )
    assert sid == created.id
    assert is_new is False
