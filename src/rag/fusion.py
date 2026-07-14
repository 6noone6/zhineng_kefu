"""Reciprocal Rank Fusion for hybrid retrieval."""

from __future__ import annotations

from src.rag import Chunk


def _chunk_key(chunk: Chunk) -> str:
    if chunk.chunk_id:
        return chunk.chunk_id
    return f"{chunk.source}:{chunk.text[:80]}"


def reciprocal_rank_fusion(
    ranked_lists: list[list[Chunk]],
    *,
    rrf_k: int = 60,
    top_k: int = 3,
) -> list[Chunk]:
    """Merge multiple ranked lists with RRF (Cormack et al.)."""
    scores: dict[str, float] = {}
    chunks_by_key: dict[str, Chunk] = {}

    for results in ranked_lists:
        for rank, chunk in enumerate(results):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            if key not in chunks_by_key:
                chunks_by_key[key] = chunk

    ranked_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
    return [
        Chunk(
            text=chunks_by_key[key].text,
            source=chunks_by_key[key].source,
            score=scores[key],
            chunk_id=chunks_by_key[key].chunk_id,
            lang=chunks_by_key[key].lang,
        )
        for key in ranked_keys[:top_k]
    ]
