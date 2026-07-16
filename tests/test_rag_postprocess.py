"""Unit tests for RAG confidence reject and source diversity (no live LLM)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag import Chunk
from src.rag.multilingual import build_cross_lingual_queries, retrieve_multilingual
from src.rag.postprocess import (
    diversify_by_source,
    expand_domain_queries,
    is_gulf_warranty_query,
    prefer_gulf_warranty_sources,
    reject_below_min_score,
)


def _chunk(source: str, score: float, text: str = "x", chunk_id: str = "") -> Chunk:
    return Chunk(
        text=text,
        source=source,
        score=score,
        chunk_id=chunk_id or f"{source}:{score}",
    )


def test_diversify_by_source_limits_repeats():
    chunks = [
        _chunk("payment_currency.txt", 0.9, chunk_id="a"),
        _chunk("payment_currency.txt", 0.8, chunk_id="b"),
        _chunk("payment_currency.txt", 0.7, chunk_id="c"),
        _chunk("faq_payment_mined.txt", 0.6, chunk_id="d"),
        _chunk("returns_refund.txt", 0.5, chunk_id="e"),
    ]
    out = diversify_by_source(chunks, max_per_source=2, top_k=3)
    assert len(out) == 3
    assert [c.source for c in out] == [
        "payment_currency.txt",
        "payment_currency.txt",
        "faq_payment_mined.txt",
    ]


def test_diversify_by_source_max_one():
    chunks = [
        _chunk("a.txt", 1.0, chunk_id="1"),
        _chunk("a.txt", 0.9, chunk_id="2"),
        _chunk("b.txt", 0.8, chunk_id="3"),
    ]
    out = diversify_by_source(chunks, max_per_source=1, top_k=2)
    assert [c.source for c in out] == ["a.txt", "b.txt"]


def test_reject_below_min_score_empties_low_confidence():
    chunks = [_chunk("phone.txt", 0.45), _chunk("faq.txt", 0.40)]
    assert reject_below_min_score(chunks, min_score=0.60) == []


def test_reject_below_min_score_keeps_confident():
    chunks = [_chunk("phone.txt", 0.72), _chunk("faq.txt", 0.55)]
    out = reject_below_min_score(chunks, min_score=0.60)
    assert len(out) == 2
    assert out[0].source == "phone.txt"


def test_reject_disabled_when_min_score_zero():
    chunks = [_chunk("phone.txt", 0.1)]
    assert reject_below_min_score(chunks, min_score=0.0) == chunks


def test_gulf_warranty_query_detect_and_expand():
    q = "海湾地区保修政策是什么？"
    assert is_gulf_warranty_query(q)
    variants = expand_domain_queries(q)
    assert variants[0] == q
    assert any("gulf warranty" in v.lower() for v in variants[1:])


def test_prefer_gulf_warranty_sources_boosts():
    q = "海湾地区保修政策是什么？"
    chunks = [
        _chunk("phone.txt", 0.031),
        _chunk("gulf_region_specifics.txt", 0.030),
        _chunk("gulf_warranty.txt", 0.030),
    ]
    ranked = prefer_gulf_warranty_sources(chunks, q)
    assert ranked[0].source == "gulf_warranty.txt"


def test_build_cross_lingual_includes_gulf_expand():
    queries = build_cross_lingual_queries("海湾地区保修政策是什么？", "zh")
    assert any("gulf" in q.lower() for q in queries)


@pytest.mark.asyncio
async def test_retrieve_multilingual_propagates_empty_reject():
    """When retriever confidence-rejects all variants, result is empty."""
    retriever = MagicMock()
    retriever.search_async = AsyncMock(return_value=[])

    chunks = await retrieve_multilingual(
        "今天股市涨跌？",
        retriever,
        top_k=3,
        fetch_k=6,
        rrf_k=60,
    )
    assert chunks == []


@pytest.mark.asyncio
async def test_retrieve_multilingual_applies_source_diversity():
    async def search_async(query: str, top_k: int | None = None):
        return [
            _chunk("payment_currency.txt", 0.9, chunk_id=f"{query}-1"),
            _chunk("payment_currency.txt", 0.85, chunk_id=f"{query}-2"),
            _chunk("payment_currency.txt", 0.8, chunk_id=f"{query}-3"),
            _chunk("faq_payment_mined.txt", 0.7, chunk_id=f"{query}-4"),
        ]

    retriever = MagicMock()
    retriever.search_async = AsyncMock(side_effect=search_async)

    chunks = await retrieve_multilingual(
        "支持哪些支付方式？",
        retriever,
        top_k=3,
        fetch_k=4,
        rrf_k=60,
    )
    assert chunks
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.source] = counts.get(c.source, 0) + 1
    assert all(n <= 2 for n in counts.values())
