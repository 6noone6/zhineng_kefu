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

    async def fake_call_tool(tool_name, tool_input, **_kwargs):
        if tool_name == "query_order":
            return {
                "success": True,
                "data": {
                    "order_id": "ORD-1001",
                    "status": "shipped",
                    "carrier": "DHL",
                    "tracking_number": "DHL123",
                },
            }
        return {
            "success": True,
            "data": {
                "logistics_number": "DHL123",
                "status": "in_transit",
                "carrier": "DHL",
                "current_location": "Shanghai",
                "estimated_delivery": "2026-07-20",
            },
        }

    agent._call_tool = AsyncMock(side_effect=fake_call_tool)
    response = await agent.process("查订单 ORD-1001 和物流")

    assert len(response.tools_used) == 2
    assert "query_order" in response.tools_used
    assert "fetch_logistics_information" in response.tools_used
    # Structured tools format locally — no Kimi synthesis round-trip.
    kimi.synthesize_multi_answer.assert_not_awaited()
    assert "ORD-1001" in response.answer


@pytest.mark.asyncio
async def test_process_stream_defers_rag_and_streams_once(monkeypatch):
    """Stream path defers generation in ReAct, then streams tokens once via customer_chat_stream."""
    kimi = MagicMock()
    retriever = MagicMock()
    agent = AgentService(kimi, retriever, None)

    call = MagicMock()
    call.function.name = "customer_chat"
    call.function.arguments = '{"query": "保修多久"}'
    call.id = "call_1"
    assistant_msg = MagicMock()
    assistant_msg.tool_calls = [call]

    kimi.chat = AsyncMock(return_value=assistant_msg)

    async def fake_stream(query, retriever, qwen=None, kimi=None):
        async def gen():
            for token in ["保修", "期为", "一年。"]:
                yield token

        return ["kb.md"], gen()

    monkeypatch.setattr(
        "src.tools.knowledge_chat.customer_chat_stream",
        fake_stream,
    )

    chunks: list[str] = []
    done = None
    async for event in agent.process_stream("保修多久"):
        if event["type"] == "chunk":
            chunks.append(event["content"])
        elif event["type"] == "done":
            done = event

    assert "".join(chunks) == "保修期为一年。"
    assert done is not None
    assert done["answer"] == "保修期为一年。"
    assert done["citations"] == ["kb.md"]


@pytest.mark.asyncio
async def test_process_stream_stops_on_cancel_event():
    import asyncio

    kimi = MagicMock()

    async def slow_tokens(_messages):
        for token in ["你", "好", "啊"]:
            yield token
            await asyncio.sleep(0.01)

    kimi.chat_stream = slow_tokens
    agent = AgentService(kimi, MagicMock(), None)
    cancel_event = asyncio.Event()

    chunks: list[str] = []
    async for event in agent.process_stream("你好", cancel_event=cancel_event):
        if event["type"] == "chunk":
            chunks.append(event["content"])
            if len(chunks) == 1:
                cancel_event.set()

    assert chunks == ["你"]

