"""Shared pytest fixtures."""

from __future__ import annotations

import os

# Configure test environment before application imports.
os.environ.setdefault("ENV", "development")
os.environ.setdefault("API_KEY", "test-api-key")
os.environ.setdefault("MOONSHOT_API_KEY", "test-moonshot-key")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("RAG_BACKEND", "cloud")

from src.core.config import get_settings

get_settings.cache_clear()

import pytest_asyncio  # noqa: E402
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: E402

from src.models import Base  # noqa: E402


@pytest_asyncio.fixture
async def db_engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    factory = async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        yield session
        await session.rollback()
