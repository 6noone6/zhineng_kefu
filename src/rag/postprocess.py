"""Post-RRF helpers: source diversity and confidence gating."""

from __future__ import annotations

import re

from src.rag import Chunk

_GULF_RE = re.compile(r"海湾|gulf|gcc|中东", re.IGNORECASE)
_WARRANTY_RE = re.compile(r"保修|质保|延保|warranty|warrant", re.IGNORECASE)


def is_gulf_warranty_query(query: str) -> bool:
    return bool(_GULF_RE.search(query) and _WARRANTY_RE.search(query))


def expand_domain_queries(query: str) -> list[str]:
    """Domain-specific query variants to improve recall (e.g. gulf warranty)."""
    q = (query or "").strip()
    if not q:
        return []
    variants = [q]
    if is_gulf_warranty_query(q):
        variants.extend(
            [
                f"{q} gulf warranty 海湾地区保修",
                "海湾地区手机质保政策 gulf region phone warranty 2年",
            ]
        )
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for item in variants:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def diversify_by_source(
    chunks: list[Chunk],
    *,
    max_per_source: int = 2,
    top_k: int,
) -> list[Chunk]:
    """Limit how many chunks share the same source file in the final top_k."""
    if top_k <= 0 or not chunks:
        return []
    limit = max(1, max_per_source)
    picked: list[Chunk] = []
    counts: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.source or ""
        if counts.get(src, 0) >= limit:
            continue
        picked.append(chunk)
        counts[src] = counts.get(src, 0) + 1
        if len(picked) >= top_k:
            break
    return picked


def reject_below_min_score(chunks: list[Chunk], *, min_score: float) -> list[Chunk]:
    """Empty-result path when the best chunk score is below the confidence floor."""
    if min_score <= 0 or not chunks:
        return chunks
    if float(chunks[0].score) < min_score:
        return []
    return chunks


def prefer_gulf_warranty_sources(chunks: list[Chunk], query: str) -> list[Chunk]:
    """Light re-rank so gulf_warranty* surfaces for gulf+warranty questions."""
    if not chunks or not is_gulf_warranty_query(query):
        return chunks

    def sort_key(chunk: Chunk) -> tuple[float, float]:
        src = (chunk.source or "").lower()
        boost = 0.0
        if "gulf_warranty" in src:
            boost += 0.05
        elif "warranty" in src:
            boost += 0.02
        elif "gulf_region" in src:
            boost += 0.01
        return (float(chunk.score) + boost, boost)

    return sorted(chunks, key=sort_key, reverse=True)
