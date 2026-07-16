"""Postgres + Redis + Alembic integration smoke.

Prefers Docker testcontainers when available. Otherwise falls back to local
Postgres/Redis (see INTEGRATION_* env vars) if reachable. Skips with a clear
reason when neither backend can run.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _docker_available() -> bool:
    try:
        import docker

        client = docker.from_env()
        client.ping()
        return True
    except Exception:
        return False


def _sync_pg_url(async_url: str) -> str:
    return async_url.replace("postgresql+asyncpg://", "postgresql://", 1)


def _replace_db_name(url: str, db_name: str) -> str:
    parsed = urlparse(url)
    return urlunparse(parsed._replace(path=f"/{db_name}"))


def _run_alembic_upgrade(database_url: str) -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    # Fresh subprocess so alembic/env.py picks up DATABASE_URL (get_settings cache).
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"alembic upgrade failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )


def _assert_migrated_tables(database_url: str) -> None:
    import asyncio

    import asyncpg

    async def _check() -> list[str]:
        conn = await asyncpg.connect(_sync_pg_url(database_url))
        try:
            rows = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY 1"
            )
            return [r["tablename"] for r in rows]
        finally:
            await conn.close()

    tables = asyncio.run(_check())
    for required in ("alembic_version", "sessions", "users", "message_feedback"):
        assert required in tables, f"missing table {required}; got {tables}"


def _redis_ping(redis_url: str) -> None:
    from redis import Redis

    client = Redis.from_url(redis_url, socket_connect_timeout=2)
    assert client.ping() is True
    client.close()


@pytest.mark.integration
def test_postgres_redis_alembic_smoke():
    """Alembic upgrade head + Redis ping via testcontainers or local services."""
    pytest.importorskip("psycopg2")

    docker_ok = _docker_available()
    if docker_ok:
        pytest.importorskip("testcontainers")
        from testcontainers.postgres import PostgresContainer
        from testcontainers.redis import RedisContainer

        with PostgresContainer("postgres:16-alpine") as pg, RedisContainer("redis:7-alpine") as rd:
            # testcontainers returns sync postgres URL (psycopg2-compatible).
            sync_url = pg.get_connection_url()
            async_url = sync_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            redis_url = rd.get_connection_url()
            _run_alembic_upgrade(async_url)
            _assert_migrated_tables(async_url)
            _redis_ping(redis_url)
        return

    # Local fallback: real Postgres/Redis on the developer machine or CI services.
    base_async = os.environ.get(
        "INTEGRATION_DATABASE_URL",
        "postgresql+asyncpg://kefu:kefu@localhost:5432/postgres",
    )
    redis_url = os.environ.get("INTEGRATION_REDIS_URL", "redis://localhost:6379/15")

    parsed = urlparse(_sync_pg_url(base_async))
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    redis_parsed = urlparse(redis_url)
    redis_host = redis_parsed.hostname or "localhost"
    redis_port = redis_parsed.port or 6379

    if not _tcp_open(host, port) or not _tcp_open(redis_host, redis_port):
        pytest.skip(
            "Docker unavailable for testcontainers, and local Postgres/Redis not reachable "
            f"({host}:{port} / {redis_host}:{redis_port}). "
            "Set INTEGRATION_DATABASE_URL / INTEGRATION_REDIS_URL or start Docker."
        )

    import asyncio

    import asyncpg

    smoke_db = f"kefu_smoke_{uuid.uuid4().hex[:10]}"
    admin_url = _replace_db_name(base_async, "postgres")
    smoke_url = _replace_db_name(base_async, smoke_db)

    async def _prepare_and_cleanup(prepare: bool) -> None:
        conn = await asyncpg.connect(_sync_pg_url(admin_url))
        try:
            if prepare:
                await conn.execute(f'DROP DATABASE IF EXISTS "{smoke_db}" WITH (FORCE)')
                await conn.execute(f'CREATE DATABASE "{smoke_db}"')
            else:
                await conn.execute(f'DROP DATABASE IF EXISTS "{smoke_db}" WITH (FORCE)')
        finally:
            await conn.close()

    asyncio.run(_prepare_and_cleanup(True))
    try:
        _run_alembic_upgrade(smoke_url)
        _assert_migrated_tables(smoke_url)
        _redis_ping(redis_url)
    finally:
        asyncio.run(_prepare_and_cleanup(False))
