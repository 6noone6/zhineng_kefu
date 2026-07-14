from __future__ import annotations

import time
from typing import TYPE_CHECKING, AsyncIterator

from src.core.metrics import RAG_HIT_RATE, RAG_LATENCY, RAG_QUERIES
from src.core.config import get_settings
from src.rag.multilingual import retrieve_multilingual
from src.rag.prompts import (
    LANGUAGE_MATCH_INSTRUCTION,
    build_rag_messages,
    detect_user_language,
)
from src.tools import ToolResult

if TYPE_CHECKING:
    from src.rag import Chunk
    from src.rag.retriever import Retriever
    from src.services.llm.local_qwen import LocalQwenService
    from src.services.llm.kimi_client import KimiClient


async def retrieve_chunks(query: str, retriever: Retriever) -> list[Chunk]:
    settings = get_settings()

    RAG_QUERIES.inc()
    start = time.perf_counter()
    try:
        fetch_k = max(settings.rag_top_k * 3, settings.rag_top_k + 2)
        chunks = await retrieve_multilingual(
            query,
            retriever,
            top_k=settings.rag_top_k,
            fetch_k=fetch_k,
            rrf_k=settings.hybrid_rrf_k,
        )
        RAG_HIT_RATE.labels(hit="true" if chunks else "false").inc()
        return chunks
    finally:
        RAG_LATENCY.observe(time.perf_counter() - start)


def build_citations(chunks: list) -> list[str]:
    """Show source filenames only — avoids mixing raw multilingual snippets in UI."""
    return [c.source for c in chunks]


async def generate_answer_stream(
    query: str,
    chunks: list,
    qwen: LocalQwenService | None = None,
    kimi: KimiClient | None = None,
) -> AsyncIterator[str]:
    settings = get_settings()

    if not chunks:
        if kimi is not None:
            async for token in kimi.chat_stream(
                [
                    {"role": "system", "content": LANGUAGE_MATCH_INSTRUCTION},
                    {
                        "role": "user",
                        "content": (
                            f"User question: {query}\n\n"
                            "No relevant information was found in the knowledge base. "
                            "Politely inform the user and suggest contacting human support."
                        ),
                    },
                ],
            ):
                yield token
        else:
            yield "找不到相关信息，请联系人工客服。"
        return

    info_texts = [c.text for c in chunks]
    messages = build_rag_messages(query, info_texts)

    use_local_qwen = (
        settings.rag_backend == "local"
        and qwen is not None
        and detect_user_language(query) == "zh"
    )
    if use_local_qwen:
        async for token in qwen.generate_from_messages_stream(messages):
            yield token
    elif kimi is not None:
        collected: list[str] = []
        async for token in kimi.chat_stream(messages):
            collected.append(token)
            yield token
        if not collected:
            answer = await kimi.chat(messages)
            if isinstance(answer, str) and answer:
                yield answer
    else:
        yield "找不到"


async def customer_chat(
    query: str,
    retriever: Retriever,
    qwen: LocalQwenService | None = None,
    kimi: KimiClient | None = None,
) -> ToolResult:
    chunks = await retrieve_chunks(query, retriever)
    citations = build_citations(chunks)

    if not chunks:
        if kimi is not None:
            answer = await kimi.chat(
                [
                    {"role": "system", "content": LANGUAGE_MATCH_INSTRUCTION},
                    {
                        "role": "user",
                        "content": (
                            f"User question: {query}\n\n"
                            "No relevant information was found in the knowledge base. "
                            "Politely inform the user and suggest contacting human support "
                            "or checking the FAQ for transfer-to-human options."
                        ),
                    },
                ],
            )
        else:
            answer = "找不到相关信息，请联系人工客服。"
        return ToolResult(
            success=True,
            data={"answer": answer},
            citations=[],
        )

    info_texts = [c.text for c in chunks]
    settings = get_settings()
    use_local_qwen = (
        settings.rag_backend == "local"
        and qwen is not None
        and detect_user_language(query) == "zh"
    )
    if use_local_qwen:
        messages = build_rag_messages(query, info_texts)
        answer = await qwen.generate_from_messages(messages)
    elif kimi is not None:
        messages = build_rag_messages(query, info_texts)
        answer = await kimi.chat(messages)
    else:
        answer = "找不到"

    return ToolResult(
        success=True,
        data={"answer": answer},
        citations=citations,
    )


async def customer_chat_stream(
    query: str,
    retriever: Retriever,
    qwen: LocalQwenService | None = None,
    kimi: KimiClient | None = None,
) -> tuple[list[str], AsyncIterator[str]]:
    """Retrieve once, return citations and a token stream for the answer."""
    chunks = await retrieve_chunks(query, retriever)
    citations = build_citations(chunks)
    stream = generate_answer_stream(query, chunks, qwen=qwen, kimi=kimi)
    return citations, stream
