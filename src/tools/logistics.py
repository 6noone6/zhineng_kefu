from __future__ import annotations

import datetime

import httpx

from src.core.config import get_settings
from src.tools import ToolResult


def _mock_logistics_data(logistics_number: str) -> dict:
    return {
        "logistics_number": logistics_number,
        "status": "In Transit",
        "carrier": "DHL Express",
        "destination_country": "AE",
        "customs_status": "Cleared",
        "estimated_delivery": "2024-07-15",
        "current_location": "Dubai Sorting Center, UAE",
        "last_update_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "_mock": True,
        "events": [
            {
                "event_type": "Pickup",
                "time": "2024-07-10 09:00:00",
                "location": "Shenzhen Warehouse, CN",
            },
            {
                "event_type": "Export Customs Cleared",
                "time": "2024-07-10 18:00:00",
                "location": "Shenzhen Port, CN",
            },
            {
                "event_type": "In Transit",
                "time": "2024-07-11 14:30:00",
                "location": "Dubai Sorting Center, UAE",
            },
        ],
    }


async def fetch_logistics_information(logistics_number: str) -> ToolResult:
    settings = get_settings()

    if settings.logistics_api_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    settings.logistics_api_url,
                    params={"number": logistics_number},
                    headers={"Authorization": f"Bearer {settings.logistics_api_key}"},
                )
                resp.raise_for_status()
                return ToolResult(success=True, data=resp.json())
        except Exception:
            return ToolResult(success=False, error="Logistics lookup failed")

    if settings.env == "production":
        return ToolResult(
            success=False,
            error="Logistics API is not configured. Contact support.",
        )

    return ToolResult(success=True, data=_mock_logistics_data(logistics_number))
