import pytest

from src.tools.logistics import fetch_logistics_information
from src.tools.order import query_order, list_my_orders


@pytest.mark.asyncio
async def test_mock_logistics():
    result = await fetch_logistics_information("DHL123")
    assert result.success
    assert result.data["logistics_number"] == "DHL123"
    assert result.data.get("_mock") is True


@pytest.mark.asyncio
async def test_mock_order_query():
    result = await query_order("ORD-9999")
    assert result.success
    assert result.data["order_id"] == "ORD-9999"
    assert "tracking_number" in result.data


@pytest.mark.asyncio
async def test_mock_my_orders():
    result = await list_my_orders(user_id="u1", email="demo@gulf.ae")
    assert result.success
    assert result.data["count"] >= 1
    assert result.data["orders"][0]["order_id"] == "ORD-1001"
