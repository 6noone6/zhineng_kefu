from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.agent_service import AgentService
from src.services.session_service import SessionService
from src.services.workflows.return_workflow import try_handle_return_workflow


@pytest.mark.asyncio
async def test_return_workflow_asks_for_order_id():
    kimi = MagicMock()
    retriever = MagicMock()
    agent = AgentService(kimi, retriever, None)

    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    sessions = SessionService(redis)

    response = await try_handle_return_workflow(
        "我要退货",
        agent,
        sessions,
        "session-1",
    )

    assert response is not None
    assert "订单" in response.answer or "order" in response.answer.lower()
    redis.setex.assert_awaited()


@pytest.mark.asyncio
async def test_return_workflow_with_order_id_calls_tool():
    kimi = MagicMock()
    retriever = MagicMock()
    agent = AgentService(kimi, retriever, None)
    agent._call_tool = AsyncMock(
        return_value={
            "success": True,
            "data": {"answer": "退货政策说明"},
            "citations": ["returns_refund_en.txt"],
        }
    )

    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock()
    sessions = SessionService(redis)

    response = await try_handle_return_workflow(
        "我要退货 ORD-1001",
        agent,
        sessions,
        "session-2",
    )

    assert response is not None
    assert response.answer == "退货政策说明"
    agent._call_tool.assert_awaited_once()
    assert agent._call_tool.await_args[0][0] == "create_return_request"
