from __future__ import annotations

from typing import TYPE_CHECKING

from src.tools import ToolResult

if TYPE_CHECKING:
    from src.rag.retriever import Retriever
    from src.services.llm.kimi_client import KimiClient
    from src.services.llm.local_qwen import LocalQwenService


async def create_return_request(
    query: str,
    retriever: Retriever,
    order_id: str | None = None,
    qwen: LocalQwenService | None = None,
    kimi: KimiClient | None = None,
) -> ToolResult:
    """Answer return/refund questions via returns policy knowledge (RAG)."""
    from src.tools import knowledge_chat

    rag_query = query.strip()
    if order_id:
        rag_query = f"{rag_query} 订单号: {order_id.strip()}"

    result = await knowledge_chat.customer_chat(
        f"退换货退货退款政策: {rag_query}",
        retriever,
        qwen=qwen,
        kimi=kimi,
    )
    if order_id and result.data:
        result.data["order_id"] = order_id.strip()
        result.data["return_request"] = True
    elif result.data:
        result.data["return_request"] = True
    return result
