from __future__ import annotations

import json
import uuid

import structlog
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.metrics import SESSIONS_CREATED
from src.models.message import Message
from src.models.session import ChatSession

logger = structlog.get_logger()


class SessionService:
    def __init__(self, redis_client=None):
        self.redis = redis_client
        self.settings = get_settings()

    async def create_session(
        self, db: AsyncSession, user_id: str | None = None
    ) -> ChatSession:
        session = ChatSession(id=str(uuid.uuid4()), user_id=user_id, status="active")
        db.add(session)
        await db.flush()
        SESSIONS_CREATED.inc()
        return session

    async def get_session(
        self, db: AsyncSession, session_id: str
    ) -> ChatSession | None:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id)
        )
        return result.scalar_one_or_none()

    async def bind_user(
        self, db: AsyncSession, session_id: str, user_id: str
    ) -> None:
        session = await self.get_session(db, session_id)
        if session and not session.user_id:
            session.user_id = user_id
            await db.flush()

    async def get_messages(
        self, db: AsyncSession, session_id: str, limit: int = 50
    ) -> list[Message]:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def add_message(
        self,
        db: AsyncSession,
        session_id: str,
        role: str,
        content: str,
        tool_calls: dict | None = None,
        citations: list | None = None,
    ) -> Message:
        message = Message(
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            citations=citations,
        )
        db.add(message)
        await db.flush()

        if self.redis:
            cache_key = f"session:{session_id}:history"
            history = await self._get_cached_history(session_id)
            history.append({"role": role, "content": content})
            max_hist = self.settings.session_max_history
            if len(history) > max_hist:
                history = history[-max_hist:]
            await self.redis.setex(
                cache_key,
                self.settings.session_redis_ttl,
                json.dumps(history, ensure_ascii=False),
            )

        return message

    async def _get_cached_history(self, session_id: str) -> list[dict]:
        if not self.redis:
            return []
        cache_key = f"session:{session_id}:history"
        data = await self.redis.get(cache_key)
        if data:
            return json.loads(data)
        return []

    async def _get_session_summary(self, session_id: str) -> str | None:
        if not self.redis:
            return None
        data = await self.redis.get(f"session:{session_id}:summary")
        return data if data else None

    async def update_session_summary(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Rolling short summary for multi-turn context (Redis, max ~500 chars)."""
        if not self.redis:
            return
        key = f"session:{session_id}:summary"
        existing = await self.redis.get(key) or ""
        snippet_user = user_message.strip()[:120]
        snippet_assistant = assistant_message.strip()[:180]
        line = f"Q: {snippet_user} | A: {snippet_assistant}"
        combined = f"{existing}\n{line}".strip() if existing else line
        if len(combined) > 500:
            combined = combined[-500:]
        await self.redis.setex(
            key,
            self.settings.session_redis_ttl,
            combined,
        )

    async def get_context_messages(
        self, db: AsyncSession, session_id: str
    ) -> list[dict[str, str]]:
        cached = await self._get_cached_history(session_id)
        history = cached if cached else [
            {"role": m.role, "content": m.content}
            for m in await self.get_messages(db, session_id)
        ]

        summary = await self._get_session_summary(session_id)
        if summary:
            return [
                {
                    "role": "system",
                    "content": (
                        "Earlier conversation summary (for context only, do not repeat verbatim):\n"
                        f"{summary}"
                    ),
                },
                *history,
            ]
        return history

    async def delete_session(self, db: AsyncSession, session_id: str) -> bool:
        session = await self.get_session(db, session_id)
        if not session:
            return False
        await db.execute(delete(Message).where(Message.session_id == session_id))
        await db.delete(session)
        await db.flush()
        if self.redis:
            await self.redis.delete(f"session:{session_id}:history")
            await self.redis.delete(f"session:{session_id}:summary")
            await self.redis.delete(f"session:{session_id}:workflow")
        logger.info("session_deleted", session_id=session_id)
        return True

    def _workflow_key(self, session_id: str) -> str:
        return f"session:{session_id}:workflow"

    async def get_workflow_state(self, session_id: str) -> dict | None:
        if not self.redis:
            return None
        data = await self.redis.get(self._workflow_key(session_id))
        if not data:
            return None
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return None

    async def set_workflow_state(self, session_id: str, state: dict) -> None:
        if not self.redis:
            return
        await self.redis.setex(
            self._workflow_key(session_id),
            self.settings.session_redis_ttl,
            json.dumps(state, ensure_ascii=False),
        )

    async def update_workflow_state(self, session_id: str, **updates: object) -> None:
        current = await self.get_workflow_state(session_id) or {}
        current.update(updates)
        await self.set_workflow_state(session_id, current)

    async def clear_workflow_state(self, session_id: str) -> None:
        if not self.redis:
            return
        await self.redis.delete(self._workflow_key(session_id))
