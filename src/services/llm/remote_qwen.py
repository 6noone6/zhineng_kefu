from __future__ import annotations

import structlog

from src.core.config import Settings, get_settings
from src.utils.http_client import get_http_client

logger = structlog.get_logger()


class RemoteQwenClient:
    """HTTP client for Qwen inference on a remote rag-worker."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        base = self.settings.qwen_inference_url.rstrip("/")
        self._inference_url = f"{base}/inference"

    async def generate_from_messages(self, messages: list[dict]) -> str:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.settings.api_key:
            headers["X-API-Key"] = self.settings.api_key

        try:
            client = get_http_client(timeout=120.0)
            resp = await client.post(
                self._inference_url,
                json={"messages": messages},
                headers=headers,
                timeout=120.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("answer", "").strip()
        except Exception as exc:
            logger.warning("remote_qwen_inference_failed", error=str(exc))
            raise RuntimeError("Remote Qwen inference failed") from exc

    async def generate_from_messages_stream(self, messages: list[dict]):
        """Emit full answer as one chunk — remote endpoint is not true SSE."""
        text = await self.generate_from_messages(messages)
        if text:
            yield text

    # Backward-compatible alias
    async def generate_stream(self, messages: list[dict]):
        async for token in self.generate_from_messages_stream(messages):
            yield token
