from __future__ import annotations

from src.core.config import get_settings
from src.tools import ToolResult
from src.utils.http_client import get_http_client

# Demo orders bound to user email (development / mock)
_MOCK_USER_ORDERS: dict[str, list[dict]] = {
    "demo@gulf.ae": [
        {
            "order_id": "ORD-1001",
            "status": "Shipped",
            "destination_country": "AE",
            "currency": "AED",
            "total": 1299.00,
            "tracking_number": "DHL987654321",
            "carrier": "DHL Express",
            "created_at": "2024-07-01 10:00:00",
        },
        {
            "order_id": "ORD-1002",
            "status": "Delivered",
            "destination_country": "SA",
            "currency": "SAR",
            "total": 899.00,
            "tracking_number": "ARX123456789",
            "carrier": "Aramex",
            "created_at": "2024-06-15 14:30:00",
        },
    ],
}


def _mock_order_data(order_id: str) -> dict:
    return {
        "order_id": order_id,
        "status": "Shipped",
        "destination_country": "SA",
        "currency": "USD",
        "items": [{"name": "智能手机 / Smartphone", "quantity": 1}],
        "total": 2999.00,
        "created_at": "2024-07-01 10:00:00",
        "carrier": "Aramex",
        "tracking_number": "ARX123456789",
        "_mock": True,
    }


async def query_order(order_id: str, email: str | None = None) -> ToolResult:
    settings = get_settings()

    if settings.order_api_url:
        try:
            params = {"order_id": order_id}
            if email:
                params["email"] = email
            client = get_http_client(timeout=10.0)
            resp = await client.get(
                f"{settings.order_api_url}/{order_id}",
                params=params,
                headers={"Authorization": f"Bearer {settings.order_api_key}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            return ToolResult(success=True, data=resp.json())
        except Exception:
            return ToolResult(success=False, error="Order lookup failed")

    if settings.env == "production":
        return ToolResult(
            success=False,
            error="Order API is not configured. Contact support.",
        )

    data = _mock_order_data(order_id)
    if email:
        data["email_verified"] = True
    return ToolResult(success=True, data=data)


async def list_my_orders(
    user_id: str | None = None,
    email: str | None = None,
) -> ToolResult:
    """List orders for authenticated user."""
    settings = get_settings()
    email_key = (email or "").lower()

    if settings.order_api_url and user_id:
        try:
            client = get_http_client(timeout=10.0)
            resp = await client.get(
                f"{settings.order_api_url}/user/{user_id}",
                headers={"Authorization": f"Bearer {settings.order_api_key}"},
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return ToolResult(success=True, data={"orders": data, "user_id": user_id})
        except Exception:
            return ToolResult(success=False, error="Failed to fetch orders")

    if settings.env == "production":
        return ToolResult(
            success=False,
            error="Order API is not configured. Please log in via the store website.",
        )

    orders = _MOCK_USER_ORDERS.get(email_key, [])
    if not orders and email_key:
        orders = [
            {
                **_mock_order_data(f"ORD-{email_key.split('@')[0][:6].upper()}"),
                "email": email_key,
            }
        ]
    return ToolResult(
        success=True,
        data={
            "orders": orders,
            "user_id": user_id,
            "email": email_key,
            "count": len(orders),
            "_mock": True,
        },
    )
