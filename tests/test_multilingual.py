from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag import Chunk
from src.rag.multilingual import (
    build_cross_lingual_queries,
    detect_chunk_language,
    retrieve_multilingual,
)
from src.services.workflows.return_workflow import (
    extract_order_id,
    is_return_intent,
)


def test_detect_chunk_language_en_suffix():
    assert detect_chunk_language("returns_refund_en.txt", "中文") == "en"


def test_detect_chunk_language_ar_suffix():
    assert detect_chunk_language("faq_contact_ar.txt", "text") == "ar"


def test_build_cross_lingual_queries_english():
    queries = build_cross_lingual_queries("How long is warranty?", "en")
    assert len(queries) >= 2
    assert queries[0] == "How long is warranty?"


def test_build_cross_lingual_queries_gulf_warranty_zh():
    queries = build_cross_lingual_queries("海湾地区保修政策是什么？", "zh")
    assert queries[0] == "海湾地区保修政策是什么？"
    assert any("gulf warranty" in q.lower() for q in queries)


def test_return_intent_chinese():
    assert is_return_intent("我要退货")


def test_return_intent_english():
    assert is_return_intent("I need a refund")


def test_extract_order_id():
    assert extract_order_id("查订单 ORD-1001 物流") == "ORD-1001"


@pytest.mark.asyncio
async def test_retrieve_multilingual_parallel_merges():
    async def search_async(query: str, top_k: int | None = None):
        return [
            Chunk(text=f"hit:{query}", source="kb.md", score=1.0, chunk_id=query[:12])
        ]

    retriever = MagicMock()
    retriever.search_async = AsyncMock(side_effect=search_async)

    chunks = await retrieve_multilingual(
        "warranty length",
        retriever,
        top_k=2,
        fetch_k=3,
        rrf_k=60,
    )
    assert chunks
    assert retriever.search_async.await_count >= 2
