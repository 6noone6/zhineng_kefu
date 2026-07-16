from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import BadRequestError

from src.core.config import Settings
from src.services.llm.kimi_client import KimiClient


def _settings(**overrides) -> Settings:
    defaults = {
        "moonshot_api_key": "test-key",
        "moonshot_model": "kimi-k2.6",
        "moonshot_temperature": 1.0,
        "moonshot_thinking": "disabled",
        "env": "development",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def _mock_completion(content: str = "ok"):
    message = MagicMock()
    message.content = content
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.usage = None
    return response


@pytest.mark.asyncio
async def test_chat_disables_thinking_and_uses_temp_06():
    settings = _settings(moonshot_thinking="disabled")
    client = KimiClient(settings)
    create = AsyncMock(return_value=_mock_completion())
    client._client.chat.completions.create = create

    await client.chat([{"role": "user", "content": "hi"}])

    assert create.await_count == 1
    kwargs = create.await_args.kwargs
    assert kwargs["temperature"] == 0.6
    assert kwargs["model"] == "kimi-k2.6"
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_chat_thinking_enabled_uses_temp_1():
    settings = _settings(moonshot_thinking="enabled")
    client = KimiClient(settings)
    create = AsyncMock(return_value=_mock_completion())
    client._client.chat.completions.create = create

    await client.chat([{"role": "user", "content": "hi"}])

    kwargs = create.await_args.kwargs
    assert kwargs["temperature"] == 1.0
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}


@pytest.mark.asyncio
async def test_chat_with_tools_still_sends_thinking_flag():
    settings = _settings()
    client = KimiClient(settings)
    create = AsyncMock(return_value=_mock_completion())
    client._client.chat.completions.create = create

    await client.chat(
        [{"role": "user", "content": "查订单"}],
        tools=[{"type": "function", "function": {"name": "query_order"}}],
        tool_choice="auto",
    )

    assert create.await_args.kwargs["temperature"] == 0.6
    assert create.await_args.kwargs["extra_body"]["thinking"]["type"] == "disabled"


@pytest.mark.asyncio
async def test_chat_ignores_caller_temperature_for_fixed_model():
    settings = _settings(moonshot_thinking="enabled")
    client = KimiClient(settings)
    create = AsyncMock(return_value=_mock_completion())
    client._client.chat.completions.create = create

    await client.chat([{"role": "user", "content": "hi"}], temperature=0.2)

    assert create.await_args.kwargs["temperature"] == 1.0


@pytest.mark.asyncio
async def test_chat_stream_sends_thinking_disabled():
    settings = _settings()
    client = KimiClient(settings)

    async def _empty_stream():
        if False:  # pragma: no cover
            yield None

    create = AsyncMock(return_value=_empty_stream())
    client._client.chat.completions.create = create

    stream = await client.chat(
        [{"role": "user", "content": "hi"}],
        stream=True,
        operation="chat_stream",
    )
    assert stream is not None
    assert create.await_args.kwargs["temperature"] == 0.6
    assert create.await_args.kwargs["stream"] is True
    assert create.await_args.kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_non_fixed_model_allows_explicit_temperature():
    settings = _settings(
        moonshot_model="gpt-like",
        moonshot_fixed_temperature_models="kimi-k2.6",
    )
    client = KimiClient(settings)
    create = AsyncMock(return_value=_mock_completion())
    client._client.chat.completions.create = create

    await client.chat([{"role": "user", "content": "hi"}], temperature=0.5)

    kwargs = create.await_args.kwargs
    assert kwargs["temperature"] == 0.5
    assert "extra_body" not in kwargs


@pytest.mark.asyncio
async def test_bad_request_surfaces_readable_error():
    settings = _settings()
    client = KimiClient(settings)
    http_resp = httpx.Response(
        400,
        request=httpx.Request("POST", "https://api.moonshot.cn/v1/chat/completions"),
    )
    err = BadRequestError(
        message="invalid temperature: only 1 is allowed for this model",
        response=http_resp,
        body={"error": {"message": "invalid temperature: only 1 is allowed for this model"}},
    )
    client._client.chat.completions.create = AsyncMock(side_effect=err)

    with pytest.raises(RuntimeError, match="invalid temperature"):
        await client.chat([{"role": "user", "content": "hi"}])


def test_resolve_temperature_follows_thinking_mode():
    disabled = KimiClient(_settings(moonshot_thinking="disabled"))
    assert disabled._resolve_temperature(None) == 0.6
    assert disabled._resolve_temperature(0.7) == 0.6

    enabled = KimiClient(_settings(moonshot_thinking="enabled"))
    assert enabled._resolve_temperature(None) == 1.0
    assert enabled._resolve_temperature(0.7) == 1.0
