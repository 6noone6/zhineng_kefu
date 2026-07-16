"""Shared Protocol for local and remote Qwen backends."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable


@runtime_checkable
class QwenBackend(Protocol):
    async def generate_from_messages(self, messages: list[dict]) -> str: ...

    def generate_from_messages_stream(
        self, messages: list[dict]
    ) -> AsyncIterator[str]: ...
