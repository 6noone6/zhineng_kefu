from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.user import User


async def get_user_by_id(db: AsyncSession, user_id: str) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email.lower()))
    return result.scalar_one_or_none()


async def get_or_create_user(
    db: AsyncSession,
    email: str,
    *,
    name: str | None = None,
    oauth_provider: str | None = None,
    oauth_sub: str | None = None,
) -> User:
    email = email.lower().strip()
    user = await get_user_by_email(db, email)
    if user:
        if name and not user.name:
            user.name = name
        return user

    user = User(
        id=str(uuid.uuid4()),
        email=email,
        name=name,
        oauth_provider=oauth_provider,
        oauth_sub=oauth_sub,
    )
    db.add(user)
    await db.flush()
    return user
