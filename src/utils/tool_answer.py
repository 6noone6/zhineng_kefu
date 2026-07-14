"""Format tool results into user-facing text when LLM synthesis is empty."""

from __future__ import annotations

from typing import Any

from src.rag.prompts import detect_user_language


def _step_data(step: dict) -> dict[str, Any]:
    result = step.get("result") or {}
    return result.get("data") or {}


def format_tool_steps_answer(question: str, steps: list[dict]) -> str:
    """Build a concise answer from tool step payloads (no LLM)."""
    if not steps:
        return ""

    lang = detect_user_language(question)
    order_step = next((s for s in steps if s.get("tool") == "query_order"), None)
    logistics_step = next(
        (s for s in steps if s.get("tool") == "fetch_logistics_information"), None
    )
    my_orders_step = next((s for s in steps if s.get("tool") == "query_my_orders"), None)
    complaint_step = next((s for s in steps if s.get("tool") == "record_user_complaint"), None)

    if order_step and logistics_step:
        od = _step_data(order_step)
        ld = _step_data(logistics_step)
        if lang == "en":
            return (
                f"Order {od.get('order_id', '')}: status {od.get('status', '')}, "
                f"carrier {od.get('carrier', '')}, tracking {od.get('tracking_number', '')}.\n"
                f"Logistics status: {ld.get('status', '')}. "
                f"Current location: {ld.get('current_location', '')}. "
                f"Estimated delivery: {ld.get('estimated_delivery', '')}."
            )
        return (
            f"订单 {od.get('order_id', '')}：状态 {od.get('status', '')}，"
            f"承运商 {od.get('carrier', '')}，运单号 {od.get('tracking_number', '')}。\n"
            f"物流状态：{ld.get('status', '')}，"
            f"当前位置 {ld.get('current_location', '')}，"
            f"预计送达 {ld.get('estimated_delivery', '')}。"
        )

    if logistics_step and not order_step:
        ld = _step_data(logistics_step)
        tracking = ld.get("logistics_number", "")
        if lang == "en":
            return (
                f"Tracking {tracking}: {ld.get('status', '')}. "
                f"Carrier {ld.get('carrier', '')}. "
                f"Location: {ld.get('current_location', '')}. "
                f"ETA: {ld.get('estimated_delivery', '')}."
            )
        return (
            f"运单 {tracking}：{ld.get('status', '')}，"
            f"承运商 {ld.get('carrier', '')}，"
            f"当前位置 {ld.get('current_location', '')}，"
            f"预计送达 {ld.get('estimated_delivery', '')}。"
        )

    if order_step:
        od = _step_data(order_step)
        if lang == "en":
            return (
                f"Order {od.get('order_id', '')}: status {od.get('status', '')}, "
                f"carrier {od.get('carrier', '')}, "
                f"tracking {od.get('tracking_number', '')}."
            )
        return (
            f"订单 {od.get('order_id', '')}：状态 {od.get('status', '')}，"
            f"承运商 {od.get('carrier', '')}，运单号 {od.get('tracking_number', '')}。"
        )

    if my_orders_step:
        data = _step_data(my_orders_step)
        orders = data.get("orders") or []
        if not orders:
            return "暂无订单。" if lang != "en" else "No orders found."
        lines = []
        for o in orders[:5]:
            lines.append(
                f"{o.get('order_id', '')} | {o.get('status', '')} | "
                f"{o.get('carrier', '')} {o.get('tracking_number', '')}"
            )
        header = "您的订单：" if lang != "en" else "Your orders:"
        return header + "\n" + "\n".join(lines)

    if complaint_step:
        data = _step_data(complaint_step)
        return data.get("message", "") or (
            "投诉已记录。" if lang != "en" else "Your complaint has been recorded."
        )

    # Generic: use embedded answer from RAG-like tools
    for step in steps:
        data = _step_data(step)
        if data.get("answer"):
            return str(data["answer"])

    return ""
