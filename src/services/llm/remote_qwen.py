from __future__ import annotations

import httpx
import structlog

from src.core.config import Settings, get_settings

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
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    self._inference_url,
                    json={"messages": messages},
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
                return data.get("answer", "").strip()
        except Exception as exc:
            logger.warning("remote_qwen_inference_failed", error=str(exc))
            raise RuntimeError("Remote Qwen inference failed") from exc

    async def generate_stream(self, messages: list[dict]):
        text = await self.generate_from_messages(messages)
        for word in text.split():
            yield word + " "
