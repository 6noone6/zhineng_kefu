"""Cross-lingual retrieval helpers for multilingual knowledge base."""

from __future__ import annotations

from src.rag import Chunk
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.language import detect_text_language
from src.rag.prompts import detect_user_language


def detect_chunk_language(source: str, text: str) -> str:
    """Infer chunk language from filename suffix or text script."""
    src = source.lower()
    if "_en." in src or src.endswith("_en.txt") or src.endswith("_en.md"):
        return "en"
    if "_ar." in src or src.endswith("_ar.txt") or src.endswith("_ar.md"):
        return "ar"
    return detect_text_language(text)


def build_cross_lingual_queries(query: str, lang: str) -> list[str]:
    """Build query variants to improve recall across zh/en/ar knowledge files."""
    from src.rag.postprocess import expand_domain_queries

    queries = expand_domain_queries(query.strip())
    if not queries:
        return []
    base = queries[0]
    if lang == "en":
        queries.append(f"English customer support: {base}")
        queries.append(f"returns refund warranty shipping payment: {base}")
    elif lang == "ar":
        queries.append(f"Arabic Gulf customer service: {base}")
        queries.append(f"إرجاع استرداد ضمان شحن: {base}")
    else:
        queries.append(f"中文客服知识库: {base}")
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)
    return unique


async def retrieve_multilingual(
    query: str,
    retriever,
    *,
    top_k: int,
    fetch_k: int,
    rrf_k: int,
) -> list[Chunk]:
    """Retrieve with language-aware query expansion and RRF merge."""
    import asyncio

    from src.core.config import get_settings
    from src.rag.language import prefer_language_chunks
    from src.rag.postprocess import diversify_by_source, prefer_gulf_warranty_sources

    lang = detect_user_language(query)
    query_variants = build_cross_lingual_queries(query, lang)
    settings = get_settings()

    if hasattr(retriever, "search_async"):

        async def _search(variant: str) -> list[Chunk]:
            return await retriever.search_async(variant, top_k=fetch_k)

    else:

        async def _search(variant: str) -> list[Chunk]:
            return await asyncio.to_thread(retriever.search, variant, fetch_k)

    results = await asyncio.gather(*[_search(v) for v in query_variants])
    ranked_lists = [chunks for chunks in results if chunks]

    if not ranked_lists:
        return []

    if len(ranked_lists) == 1:
        merged = ranked_lists[0]
    else:
        merged = reciprocal_rank_fusion(ranked_lists, rrf_k=rrf_k, top_k=fetch_k)

    merged = prefer_gulf_warranty_sources(merged, query)
    # HybridRetriever already confidence-gates; empty lists mean reject.
    preferred = prefer_language_chunks(merged, lang, top_k=max(top_k * 2, top_k + 2))
    return diversify_by_source(
        preferred,
        max_per_source=settings.rag_max_per_source,
        top_k=top_k,
    )
