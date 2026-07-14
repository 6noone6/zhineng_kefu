from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.agent_service import AgentService


@pytest.mark.asyncio
async def test_greeting_short_circuit_skips_react():
    kimi = MagicMock()
    kimi.chat = AsyncMock(return_value="您好！有什么可以帮您？")
    retriever = MagicMock()
    agent = AgentService(kimi, retriever, None)

    response = await agent.process("你好")

    assert "您好" in response.answer or response.answer
    kimi.chat.assert_awaited_once()
    assert response.tool_name is None
    assert response.tools_used == []


@pytest.mark.asyncio
async def test_parallel_tool_calls_in_single_step():
    kimi = MagicMock()
    retriever = MagicMock()

    # Mock two parallel tool calls in one assistant message
    call_order = MagicMock()
    call_order.function.name = "query_order"
    call_order.function.arguments = '{"order_id": "ORD-1001"}'
    call_order.id = "call_order"

    call_logistics = MagicMock()
    call_logistics.function.name = "fetch_logistics_information"
    call_logistics.function.arguments = '{"logistics_number": "DHL123"}'
    call_logistics.id = "call_logistics"

    assistant_msg = MagicMock()
    assistant_msg.content = ""
    assistant_msg.tool_calls = [call_order, call_logistics]

    call_count = 0

    async def chat_side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return assistant_msg
        return "订单与物流已查询。"

    kimi.chat = AsyncMock(side_effect=chat_side_effect)
    kimi.synthesize_multi_answer = AsyncMock(return_value="订单与物流已查询。")

    agent = AgentService(kimi, retriever, None)
    response = await agent.process("查订单 ORD-1001 和物流")

    assert len(response.tools_used) == 2
    assert "query_order" in response.tools_used
    assert "fetch_logistics_information" in response.tools_used
    kimi.synthesize_multi_answer.assert_awaited_once()
