"""Language detection and chunk ranking for multilingual knowledge base."""

from __future__ import annotations

import re

from src.rag import Chunk

_AR_RE = re.compile(r"[\u0600-\u06FF]")
_ZH_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[a-zA-Z]")


def detect_text_language(text: str) -> str:
    if _AR_RE.search(text):
        return "ar"
    if _ZH_RE.search(text):
        return "zh"
    if _LATIN_RE.search(text):
        return "en"
    return "zh"


def prefer_language_chunks(chunks: list[Chunk], lang: str, top_k: int) -> list[Chunk]:
    """Re-rank retrieved chunks to prefer the user's language (filename + script)."""

    def rank_score(chunk: Chunk) -> float:
        score = float(chunk.score)
        src = chunk.source.lower()
        text_lang = detect_text_language(chunk.text)

        if lang == "en" and "_en." in src:
            score += 3.0
        elif lang == "ar" and "_ar." in src:
            score += 3.0
        elif lang == "zh" and "_en." not in src and "_ar." not in src:
            score += 1.5

        if text_lang == lang:
            score += 2.0
        elif chunk.lang and chunk.lang == lang:
            score += 2.5
        elif text_lang != lang:
            score -= 0.5

        return score

    ranked = sorted(chunks, key=rank_score, reverse=True)

    # Keep language-boosted scores; source uniqueness is applied by diversify_by_source.
    reranked = [
        Chunk(
            text=chunk.text,
            source=chunk.source,
            score=rank_score(chunk),
            chunk_id=chunk.chunk_id,
            lang=chunk.lang or detect_text_language(chunk.text),
        )
        for chunk in ranked
    ]
    return reranked[: max(top_k, 1)] if reranked else []
