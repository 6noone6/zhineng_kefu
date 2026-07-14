from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import get_settings
from src.core.metrics import TOOL_CALLS, TOOL_LATENCY, TOOL_SELECTION
from src.rag.prompts import (
    AGENT_SYSTEM_PROMPT,
    REACT_SYSTEM_HINT,
    TOOL_SELECTION_HINT,
    build_greeting_messages,
    is_greeting,
)
from src.rag.retriever import Retriever
from src.services.llm.kimi_client import (
    KimiClient,
    build_assistant_tool_message_multi,
)
from src.services.llm.local_qwen import LocalQwenService
from src.services.session_service import SessionService
from src.tools import complaint, knowledge_chat, logistics, order, returns
from src.core.tracing import trace_span
from src.services.workflows.return_workflow import try_handle_return_workflow
from src.utils.text import sanitize_assistant_reply
from src.utils.tool_answer import format_tool_steps_answer
from src.tools.registry import get_openai_tools

logger = structlog.get_logger()

_STREAM_CHUNK_SIZE = 20

# These tools already return structured fields — format locally instead of waiting
# on a long Kimi synthesis round-trip (often 30–60s before the first WebSocket chunk).
_STRUCTURED_REPLY_TOOLS = frozenset(
    {
        "query_order",
        "query_my_orders",
        "fetch_logistics_information",
        "record_user_complaint",
    }
)


def _structured_tool_answer(question: str, steps: list[ToolStep]) -> str:
    if not steps:
        return ""
    if not any(s.tool_name in _STRUCTURED_REPLY_TOOLS for s in steps):
        return ""
    payload = [
        {
            "tool": s.tool_name,
            "input": s.tool_input,
            "result": s.tool_result,
        }
        for s in steps
    ]
    return format_tool_steps_answer(question, payload)


@dataclass
class ToolStep:
    tool_name: str
    tool_input: dict
    tool_result: dict


@dataclass
class AgentResponse:
    answer: str
    tool_name: str | None = None
    tool_result: dict | None = None
    citations: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)


def _done_event(
    answer: str,
    tool_name: str | None = None,
    citations: list[str] | None = None,
    tools_used: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "done",
        "answer": answer,
        "tool_name": tool_name,
        "citations": citations or [],
        "tools_used": tools_used or [],
    }


def _collect_citations(steps: list[ToolStep]) -> list[str]:
    citations: list[str] = []
    for step in steps:
        citations.extend(step.tool_result.get("citations", []))
    return citations


class AgentService:
    def __init__(
        self,
        kimi: KimiClient,
        retriever: Retriever,
        qwen: LocalQwenService | None = None,
    ):
        self.kimi = kimi
        self.retriever = retriever
        self.qwen = qwen

    async def _call_tool(
        self,
        tool_name: str,
        tool_input: dict,
        session_id: str | None = None,
        db: AsyncSession | None = None,
        user_id: str | None = None,
        user_email: str | None = None,
    ) -> dict:
        TOOL_CALLS.labels(tool_name=tool_name).inc()
        start = time.perf_counter()
        try:
            if tool_name == "fetch_logistics_information":
                result = await logistics.fetch_logistics_information(
                    tool_input.get("logistics_number", "")
                )
            elif tool_name == "record_user_complaint":
                result = await complaint.record_user_complaint(
                    tool_input.get("complaint_details", ""),
                    session_id=session_id,
                    db=db,
                )
            elif tool_name == "create_return_request":
                result = await returns.create_return_request(
                    tool_input.get("query", ""),
                    order_id=tool_input.get("order_id"),
                    retriever=self.retriever,
                    qwen=self.qwen,
                    kimi=self.kimi,
                )
            elif tool_name == "customer_chat":
                result = await knowledge_chat.customer_chat(
                    tool_input.get("query", ""),
                    self.retriever,
                    qwen=self.qwen,
                    kimi=self.kimi,
                )
            elif tool_name == "query_order":
                result = await order.query_order(
                    tool_input.get("order_id", ""),
                    email=tool_input.get("email") or user_email,
                )
            elif tool_name == "query_my_orders":
                if not user_id:
                    return {
                        "success": False,
                        "error": "Please log in to view your orders.",
                        "login_required": True,
                    }
                result = await order.list_my_orders(user_id=user_id, email=user_email)
            else:
                return {"success": False, "error": "Unknown tool"}

            return result.to_dict()
        finally:
            TOOL_LATENCY.labels(tool_name=tool_name).observe(time.perf_counter() - start)

    async def _get_context(
        self,
        session_id: str | None,
        db: AsyncSession | None,
        sessions: SessionService | None,
    ) -> list[dict[str, str]]:
        if not session_id or not db or not sessions:
            return []
        return await sessions.get_context_messages(db, session_id)

    async def _run_react(
        self,
        question: str,
        context: list[dict[str, str]],
        *,
        session_id: str | None = None,
        db: AsyncSession | None = None,
        user_id: str | None = None,
        user_email: str | None = None,
    ) -> list[ToolStep]:
        settings = get_settings()
        steps: list[ToolStep] = []
        react_messages: list[dict] = [
            {
                "role": "system",
                "content": f"{REACT_SYSTEM_HINT}\n{TOOL_SELECTION_HINT}",
            },
        ]
        if context:
            react_messages.extend(context[-4:])
        react_messages.append({"role": "user", "content": question})

        for _ in range(settings.agent_max_steps):
            message = await self.kimi.chat(
                react_messages,
                tools=get_openai_tools(),
                tool_choice="auto",
            )
            if isinstance(message, str) or not getattr(message, "tool_calls", None):
                break

            calls = message.tool_calls
            react_messages.append(build_assistant_tool_message_multi(message))

            async def _invoke(call):
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                logger.info("react_tool_step", tool=name, session_id=session_id)
                TOOL_SELECTION.labels(tool_name=name).inc()
                tool_result = await self._call_tool(
                    name,
                    args,
                    session_id=session_id,
                    db=db,
                    user_id=user_id,
                    user_email=user_email,
                )
                return call, name, args, tool_result

            if len(calls) == 1:
                call, name, args, tool_result = await _invoke(calls[0])
                results = [(call, name, args, tool_result)]
            else:
                gathered = await asyncio.gather(*[_invoke(c) for c in calls])
                results = list(gathered)

            for call, name, args, tool_result in results:
                steps.append(ToolStep(name, args, tool_result))
                react_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

            if any(
                r[1] in ("customer_chat", "create_return_request") for r in results
            ):
                break

        return steps

    def _steps_payload(self, steps: list[ToolStep]) -> list[dict]:
        return [
            {
                "tool": s.tool_name,
                "input": s.tool_input,
                "result": s.tool_result,
            }
            for s in steps
        ]

    async def process(
        self,
        question: str,
        session_id: str | None = None,
        db: AsyncSession | None = None,
        sessions: SessionService | None = None,
        user_id: str | None = None,
        user_email: str | None = None,
    ) -> AgentResponse:
        with trace_span(
            "agent.process",
            session_id=session_id or "",
            user_id=user_id or "",
        ):
            context = await self._get_context(session_id, db, sessions)

            if is_greeting(question):
                answer = await self.kimi.chat(build_greeting_messages(question))
                if not isinstance(answer, str):
                    answer = getattr(answer, "content", "") or ""
                return AgentResponse(answer=answer)

            workflow_response = await try_handle_return_workflow(
                question, self, sessions, session_id
            )
            if workflow_response is not None:
                return workflow_response

            steps = await self._run_react(
                question,
                context,
                session_id=session_id,
                db=db,
                user_id=user_id,
                user_email=user_email,
            )

            if not steps:
                messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
                messages.extend(context[-6:])
                messages.append({"role": "user", "content": question})
                answer = await self.kimi.chat(messages)
                if not isinstance(answer, str):
                    answer = getattr(answer, "content", "") or ""
                return AgentResponse(answer=answer)

            tools_used = [s.tool_name for s in steps]
            citations = _collect_citations(steps)
            payload = self._steps_payload(steps)

            if len(steps) == 1 and steps[0].tool_name in (
                "customer_chat",
                "create_return_request",
            ):
                data = steps[0].tool_result.get("data", {})
                answer = data.get("answer", "")
                if answer:
                    return AgentResponse(
                        answer=answer,
                        tool_name=steps[0].tool_name,
                        tool_result=steps[0].tool_result,
                        citations=citations,
                        tools_used=tools_used,
                    )

            structured = _structured_tool_answer(question, steps)
            if structured:
                return AgentResponse(
                    answer=structured,
                    tool_name=tools_used[-1] if tools_used else None,
                    tool_result=steps[-1].tool_result if steps else None,
                    citations=citations,
                    tools_used=tools_used,
                )

            answer = await self.kimi.synthesize_multi_answer(
                question, payload, context=context
            )
            answer = sanitize_assistant_reply(answer)
            if not answer:
                answer = format_tool_steps_answer(question, payload)
            return AgentResponse(
                answer=answer,
                tool_name=tools_used[-1] if tools_used else None,
                tool_result=steps[-1].tool_result if steps else None,
                citations=citations,
                tools_used=tools_used,
            )

    async def process_stream(
        self,
        question: str,
        session_id: str | None = None,
        db: AsyncSession | None = None,
        sessions: SessionService | None = None,
        user_id: str | None = None,
        user_email: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        with trace_span(
            "agent.process_stream",
            session_id=session_id or "",
            user_id=user_id or "",
        ):
            context = await self._get_context(session_id, db, sessions)
            parts: list[str] = []

            if is_greeting(question):
                async for token in self.kimi.chat_stream(build_greeting_messages(question)):
                    parts.append(token)
                    yield {"type": "chunk", "content": token}
                yield _done_event("".join(parts))
                return

            workflow_response = await try_handle_return_workflow(
                question, self, sessions, session_id
            )
            if workflow_response is not None:
                answer = workflow_response.answer
                for chunk in [
                    answer[i : i + _STREAM_CHUNK_SIZE]
                    for i in range(0, len(answer), _STREAM_CHUNK_SIZE)
                ]:
                    parts.append(chunk)
                    yield {"type": "chunk", "content": chunk}
                yield _done_event(
                    answer,
                    tool_name=workflow_response.tool_name,
                    citations=workflow_response.citations,
                    tools_used=workflow_response.tools_used,
                )
                return

            steps = await self._run_react(
                question,
                context,
                session_id=session_id,
                db=db,
                user_id=user_id,
                user_email=user_email,
            )

            if not steps:
                messages: list[dict] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
                messages.extend(context[-6:])
                messages.append({"role": "user", "content": question})
                async for token in self.kimi.chat_stream(messages):
                    parts.append(token)
                    yield {"type": "chunk", "content": token}
                yield _done_event("".join(parts))
                return

            tools_used = [s.tool_name for s in steps]
            citations = _collect_citations(steps)

            if len(steps) == 1 and steps[0].tool_name in (
                "customer_chat",
                "create_return_request",
            ):
                query = steps[0].tool_input.get("query", question)
                if steps[0].tool_name == "create_return_request":
                    query = f"退换货退货退款政策: {query}"
                    if steps[0].tool_input.get("order_id"):
                        query = f"{query} 订单号: {steps[0].tool_input['order_id']}"
                step_citations, token_stream = await knowledge_chat.customer_chat_stream(
                    query,
                    self.retriever,
                    qwen=self.qwen,
                    kimi=self.kimi,
                )
                if step_citations:
                    citations = step_citations
                async for token in token_stream:
                    parts.append(token)
                    yield {"type": "chunk", "content": token}
            else:
                payload = self._steps_payload(steps)
                structured = _structured_tool_answer(question, steps)
                if structured:
                    for chunk in [
                        structured[i : i + _STREAM_CHUNK_SIZE]
                        for i in range(0, len(structured), _STREAM_CHUNK_SIZE)
                    ]:
                        parts.append(chunk)
                        yield {"type": "chunk", "content": chunk}
                else:
                    async for token in self.kimi.synthesize_multi_stream(
                        question, payload, context=context
                    ):
                        parts.append(token)
                        yield {"type": "chunk", "content": token}
                    answer_text = sanitize_assistant_reply("".join(parts))
                    if not answer_text:
                        fallback = format_tool_steps_answer(question, payload)
                        if fallback:
                            parts = [fallback]
                            yield {"type": "chunk", "content": fallback}

            yield _done_event(
                "".join(parts),
                tool_name=tools_used[-1] if tools_used else None,
                citations=citations,
                tools_used=tools_used,
            )
