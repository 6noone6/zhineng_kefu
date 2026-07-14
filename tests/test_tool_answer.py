from src.utils.tool_answer import format_tool_steps_answer


def test_format_order_and_logistics_zh():
    steps = [
        {
            "tool": "query_order",
            "result": {
                "success": True,
                "data": {
                    "order_id": "ORD-1002",
                    "status": "Delivered",
                    "carrier": "Aramex",
                    "tracking_number": "ARX123456789",
                },
            },
        },
        {
            "tool": "fetch_logistics_information",
            "result": {
                "success": True,
                "data": {
                    "logistics_number": "ARX123456789",
                    "status": "In Transit",
                    "carrier": "DHL Express",
                    "current_location": "Dubai",
                    "estimated_delivery": "2024-07-15",
                },
            },
        },
    ]
    answer = format_tool_steps_answer("查询订单 ORD-1002 的物流信息", steps)
    assert "ORD-1002" in answer
    assert "ARX123456789" in answer
    assert "物流" in answer or "状态" in answer


def test_format_logistics_only_en():
    steps = [
        {
            "tool": "fetch_logistics_information",
            "result": {
                "success": True,
                "data": {
                    "logistics_number": "ARX123456789",
                    "status": "In Transit",
                    "carrier": "Aramex",
                    "current_location": "Riyadh",
                    "estimated_delivery": "2024-07-20",
                },
            },
        },
    ]
    answer = format_tool_steps_answer("Track ARX123456789", steps)
    assert "ARX123456789" in answer
    assert "In Transit" in answer
