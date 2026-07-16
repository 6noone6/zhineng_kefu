from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from openai import APIStatusError, AsyncOpenAI, BadRequestError

from src.core.config import Settings, get_settings
from src.core.metrics import KIMI_TOKENS, LLM_LATENCY
from src.rag.prompts import (
    TOOL_SELECTION_HINT,
    detect_user_language,
    is_greeting,
    language_reply_instruction,
)
from src.tools.registry import get_openai_tools


def sanitize_messages_for_api(messages: list[dict]) -> list[dict]:
    """Moonshot rejects assistant messages with empty string content."""
    cleaned: list[dict] = []
    for msg in messages:
        m = dict(msg)
        role = m.get("role")
        content = m.get("content")
        if role == "assistant" and (content is None or content == ""):
            if m.get("tool_calls"):
                m.pop("content", None)
            else:
                continue
        cleaned.append(m)
    return cleaned


def _tool_call_payload(call) -> dict[str, Any]:
    return {
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.function.name,
            "arguments": call.function.arguments,
        },
    }


def build_assistant_tool_message(message) -> dict:
    """Build assistant message for tool-call round-trips."""
    call = message.tool_calls[0]
    payload: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [_tool_call_payload(call)],
    }
    if message.content:
        payload["content"] = message.content
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        payload["reasoning_content"] = reasoning
    return payload


def build_assistant_tool_message_multi(message) -> dict:
    """Build assistant message preserving all parallel tool calls."""
    payload: dict[str, Any] = {
        "role": "assistant",
        "tool_calls": [_tool_call_payload(call) for call in message.tool_calls],
    }
    if message.content:
        payload["content"] = message.content
    # kimi-k2.6 thinking mode requires reasoning_content on tool-call turns.
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning:
        payload["reasoning_content"] = reasoning
    return payload


class KimiClient:
    """OpenAI-compatible client for Moonshot Kimi API (agent orchestration)."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._client = AsyncOpenAI(
            api_key=self.settings.moonshot_api_key,
            base_url=self.settings.moonshot_base_url,
            timeout=self.settings.kimi_timeout_seconds,
        )
        self._semaphore = asyncio.Semaphore(self.settings.kimi_max_concurrency)

    @staticmethod
    def _record_usage(usage: Any, operation: str) -> None:
        if not usage:
            return
        prompt = getattr(usage, "prompt_tokens", None) or 0
        completion = getattr(usage, "completion_tokens", None) or 0
        total = getattr(usage, "total_tokens", None) or (prompt + completion)
        if prompt:
            KIMI_TOKENS.labels(token_type="prompt").inc(prompt)
        if completion:
            KIMI_TOKENS.labels(token_type="completion").inc(completion)
        if total:
            KIMI_TOKENS.labels(token_type="total").inc(total)

    def _fixed_temperature_models(self) -> set[str]:
        raw = self.settings.moonshot_fixed_temperature_models or ""
        return {part.strip().lower() for part in raw.split(",") if part.strip()}

    def _model_requires_fixed_temperature(self) -> bool:
        model = (self.settings.moonshot_model or "").strip().lower()
        if not model:
            return False
        fixed = self._fixed_temperature_models()
        return model in fixed or any(model.startswith(f"{m}-") or model == m for m in fixed)

    def _thinking_enabled(self) -> bool:
        return (self.settings.moonshot_thinking or "disabled").strip().lower() == "enabled"

    def _resolve_temperature(self, temperature: float | None) -> float:
        """Resolve API temperature.

        For kimi-k2.6-family models Moonshot fixes temperature by thinking mode:
        enabled → 1.0, disabled → 0.6. Callers cannot override that.
        """
        if self._model_requires_fixed_temperature():
            return 1.0 if self._thinking_enabled() else 0.6
        if temperature is not None:
            return float(temperature)
        return float(self.settings.moonshot_temperature)

    def _thinking_extra_body(self) -> dict[str, Any] | None:
        """Pass thinking switch for models that support it (kimi-k2.6)."""
        if not self._model_requires_fixed_temperature():
            return None
        return {
            "thinking": {
                "type": "enabled" if self._thinking_enabled() else "disabled",
            }
        }

    @staticmethod
    def _readable_api_error(exc: APIStatusError) -> RuntimeError:
        detail = getattr(exc, "message", None) or str(exc)
        status = getattr(exc, "status_code", None)
        prefix = f"Kimi API error ({status})" if status is not None else "Kimi API error"
        return RuntimeError(f"{prefix}: {detail}")

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        stream: bool = False,
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        *,
        operation: str = "chat",
        temperature: float | None = None,
    ):
        temperature = self._resolve_temperature(temperature)
        kwargs: dict[str, Any] = {
            "model": self.settings.moonshot_model,
            "messages": sanitize_messages_for_api(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        extra_body = self._thinking_extra_body()
        if extra_body is not None:
            kwargs["extra_body"] = extra_body

        start = time.perf_counter()
        async with self._semaphore:
            try:
                response = await self._client.chat.completions.create(**kwargs)
            except BadRequestError as exc:
                raise self._readable_api_error(exc) from exc
            except APIStatusError as exc:
                if 400 <= (exc.status_code or 0) < 500:
                    raise self._readable_api_error(exc) from exc
                raise
        LLM_LATENCY.labels(operation=operation).observe(time.perf_counter() - start)

        if stream:
            return response
        self._record_usage(getattr(response, "usage", None), operation)
        message = response.choices[0].message
        if message.tool_calls:
            return message
        return message.content or ""

    async def chat_stream(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        *,
        operation: str = "chat_stream",
    ) -> AsyncIterator[str]:
        start = time.perf_counter()
        stream = await self.chat(
            messages, max_tokens=max_tokens, stream=True, operation=operation
        )
        first_token = True
        async for chunk in stream:
            if first_token:
                LLM_LATENCY.labels(operation=operation).observe(time.perf_counter() - start)
                first_token = False
            if not chunk.choices:
                continue
            if getattr(chunk, "usage", None):
                self._record_usage(chunk.usage, operation)
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    async def select_tool(
        self,
        question: str,
        context: list[dict[str, str]] | None = None,
    ) -> tuple[str | None, dict]:
        if is_greeting(question):
            return None, {}

        system_prompt = (
            f"{TOOL_SELECTION_HINT}\n"
            "根据用户问题选择合适的工具。若无需工具（如纯问候），不要调用任何工具。"
        )
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]
        if context:
            messages.extend(context[-4:])
        messages.append({"role": "user", "content": question})

        message = await self.chat(
            messages,
            tools=get_openai_tools(),
            tool_choice="auto",
        )
        if isinstance(message, str):
            return self._parse_legacy_tool_json(message)

        if hasattr(message, "tool_calls") and message.tool_calls:
            call = message.tool_calls[0]
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            return name, args
        return None, {}

    @staticmethod
    def _parse_legacy_tool_json(raw: str) -> tuple[str | None, dict]:
        cleaned = raw.strip()
        cleaned = cleaned.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(cleaned)
            tool = data.get("tool")
            if tool:
                return tool, data.get("tool_input", {})
        except json.JSONDecodeError:
            pass
        return None, {}

    async def synthesize_answer(
        self,
        question: str,
        tool_name: str,
        tool_result: dict,
        context: list[dict[str, str]] | None = None,
    ) -> str:
        lang_instruction = language_reply_instruction(detect_user_language(question))
        prompt = (
            f"User question: {question}\n"
            f"Tool {tool_name} returned: {tool_result}\n\n"
            f"{lang_instruction}\n"
            "Answer the user in a concise, friendly tone. "
            "If the tool result is in a different language than the user's question, "
            "translate it before replying."
        )
        messages: list[dict] = [{"role": "system", "content": lang_instruction}]
        if context:
            messages.extend(context[-4:])
        messages.append({"role": "user", "content": prompt})
        result = await self.chat(messages)
        return result if isinstance(result, str) else ""

    async def synthesize_answer_stream(
        self,
        question: str,
        tool_name: str,
        tool_result: dict,
        context: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        lang_instruction = language_reply_instruction(detect_user_language(question))
        prompt = (
            f"User question: {question}\n"
            f"Tool {tool_name} returned: {tool_result}\n\n"
            f"{lang_instruction}\n"
            "Answer the user in a concise, friendly tone."
        )
        messages: list[dict] = [{"role": "system", "content": lang_instruction}]
        if context:
            messages.extend(context[-4:])
        messages.append({"role": "user", "content": prompt})
        async for token in self.chat_stream(messages):
            yield token

    async def synthesize_multi_answer(
        self,
        question: str,
        steps: list[dict],
        context: list[dict[str, str]] | None = None,
    ) -> str:
        lang_instruction = language_reply_instruction(detect_user_language(question))
        prompt = (
            f"User question: {question}\n"
            f"Tools executed (in order):\n{json.dumps(steps, ensure_ascii=False)}\n\n"
            f"{lang_instruction}\n"
            "Synthesize one concise, friendly answer using all tool results. "
            "If logistics was queried after an order lookup, combine order and tracking info. "
            "Reply with plain text only; do not use thinking or reasoning tags."
        )
        messages: list[dict] = [{"role": "system", "content": lang_instruction}]
        if context:
            messages.extend(context[-4:])
        messages.append({"role": "user", "content": prompt})
        result = await self.chat(messages)
        return result if isinstance(result, str) else ""

    async def synthesize_multi_stream(
        self,
        question: str,
        steps: list[dict],
        context: list[dict[str, str]] | None = None,
    ) -> AsyncIterator[str]:
        lang_instruction = language_reply_instruction(detect_user_language(question))
        prompt = (
            f"User question: {question}\n"
            f"Tools executed (in order):\n{json.dumps(steps, ensure_ascii=False)}\n\n"
            f"{lang_instruction}\n"
            "Synthesize one concise, friendly answer using all tool results. "
            "Reply with plain text only; do not use thinking or reasoning tags."
        )
        messages: list[dict] = [{"role": "system", "content": lang_instruction}]
        if context:
            messages.extend(context[-4:])
        messages.append({"role": "user", "content": prompt})
        async for token in self.chat_stream(messages):
            yield token
