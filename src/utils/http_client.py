"""Shared httpx AsyncClient to reuse TLS connections across tools."""

from __future__ import annotations

import httpx

_client: httpx.AsyncClient | None = None


def get_http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=timeout)
    return _client


async def close_http_client() -> None:
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
    _client = None
