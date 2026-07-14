"""Multilingual tokenization for BM25 (Chinese / English / Arabic)."""

from __future__ import annotations

import re

_ARABIC_RE = re.compile(r"[\u0600-\u06FF]+")
_LATIN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """Tokenize query or document text for BM25."""
    text = text.strip()
    if not text:
        return []

    tokens: list[str] = []

    if _CJK_RE.search(text):
        import jieba

        tokens.extend(t.strip() for t in jieba.cut(text) if t.strip())

    tokens.extend(_LATIN_RE.findall(text.lower()))
    tokens.extend(_ARABIC_RE.findall(text))

    if not tokens:
        return list(text)

    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique
