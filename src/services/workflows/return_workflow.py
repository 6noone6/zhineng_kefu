"""Explicit return/refund workflow state machine (Redis-backed per session)."""

from __future__ import annotations

import re
from enum import Enum
from typing import TYPE_CHECKING, Any

from src.rag.prompts import detect_user_language

if TYPE_CHECKING:
    from src.services.agent_service import AgentService
    from src.services.session_service import SessionService

RETURN_KEYWORDS = (
    "退货",
    "退款",
    "换货",
    "退换货",
    "退一下",
    "不想要了",
    "return",
    "refund",
    "exchange",
    "money back",
)

ORDER_ID_RE = re.compile(r"\b(ORD-[A-Z0-9]+)\b", re.IGNORECASE)

PROMPTS = {
    "awaiting_order_zh": (
        "了解您需要办理退货/退款。请提供订单号（例如 ORD-1001），"
        "我会根据退换货政策为您说明具体步骤。"
    ),
    "awaiting_order_en": (
        "I can help with your return or refund. Please provide your order ID "
        "(e.g. ORD-1001) so I can explain the policy and next steps."
    ),
    "awaiting_order_ar": (
        "يمكنني مساعدتك في الإرجاع أو الاسترداد. يرجى تقديم رقم الطلب "
        "(مثل ORD-1001) لشرح السياسة والخطوات."
    ),
}


class ReturnWorkflowState(str, Enum):
    IDLE = "idle"
    AWAITING_ORDER = "awaiting_order"
    COMPLETED = "completed"


def is_return_intent(message: str) -> bool:
    text = message.lower()
    return any(kw in message or kw in text for kw in RETURN_KEYWORDS)


def extract_order_id(message: str) -> str | None:
    match = ORDER_ID_RE.search(message)
    return match.group(1).upper() if match else None


def _prompt_for_lang(key: str, lang: str) -> str:
    suffix = lang if lang in ("en", "ar") else "zh"
    return PROMPTS.get(f"{key}_{suffix}", PROMPTS[f"{key}_zh"])


async def try_handle_return_workflow(
    message: str,
    agent: AgentService,
    sessions: SessionService | None,
    session_id: str | None,
) -> Any | None:
    """
    Run return workflow when intent detected or session already in workflow.
    Returns AgentResponse if handled, None to fall through to ReAct agent.
    """
    from src.services.agent_service import AgentResponse

    if not session_id or not sessions:
        if is_return_intent(message) and not extract_order_id(message):
            lang = detect_user_language(message)
            return AgentResponse(answer=_prompt_for_lang("awaiting_order", lang))
        return None

    workflow = await sessions.get_workflow_state(session_id)
    active = workflow.get("type") == "return" if workflow else False
    intent = is_return_intent(message)
    order_id = extract_order_id(message)

    if not active and not intent:
        return None

    lang = detect_user_language(message)
    state = str(workflow.get("state", ReturnWorkflowState.IDLE.value)) if workflow else ReturnWorkflowState.IDLE.value
    stored_order = workflow.get("order_id") if workflow else None

    if not active and intent:
        await sessions.set_workflow_state(
            session_id,
            {
                "type": "return",
                "state": ReturnWorkflowState.AWAITING_ORDER.value,
                "order_id": order_id,
            },
        )
        if not order_id:
            return AgentResponse(answer=_prompt_for_lang("awaiting_order", lang))
        stored_order = order_id

    elif active and state == ReturnWorkflowState.AWAITING_ORDER.value:
        if order_id:
            stored_order = order_id
            await sessions.update_workflow_state(session_id, order_id=order_id)
        elif not stored_order:
            return AgentResponse(answer=_prompt_for_lang("awaiting_order", lang))

    elif active and state == ReturnWorkflowState.COMPLETED.value:
        if not intent:
            await sessions.clear_workflow_state(session_id)
            return None
        stored_order = order_id or stored_order

    effective_order = stored_order or order_id
    if not effective_order:
        return AgentResponse(answer=_prompt_for_lang("awaiting_order", lang))

    result = await agent._call_tool(
        "create_return_request",
        {"query": message, "order_id": effective_order},
        session_id=session_id,
    )
    answer = result.get("data", {}).get("answer", "")
    citations = result.get("citations", [])

    await sessions.set_workflow_state(
        session_id,
        {
            "type": "return",
            "state": ReturnWorkflowState.COMPLETED.value,
            "order_id": effective_order,
        },
    )

    return AgentResponse(
        answer=answer,
        tool_name="create_return_request",
        tool_result=result,
        citations=citations,
        tools_used=["create_return_request"],
    )
