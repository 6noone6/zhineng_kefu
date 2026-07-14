from __future__ import annotations

import re


def split_text_by_word_count(text: str, max_word_count: int) -> list[str]:
    """Split Latin/space-delimited text into segments by approximate word count."""
    words = text.split()
    segments: list[str] = []
    current_segment: list[str] = []
    current_word_count = 0

    for word in words:
        current_segment.append(word)
        current_word_count += len(word) + 1

        if current_word_count > max_word_count:
            if len(current_segment) == 1:
                segments.append(current_segment[0])
                current_segment = []
                current_word_count = 0
            else:
                segments.append(" ".join(current_segment[:-1]))
                current_segment = [word]
                current_word_count = len(word) + 1

    if current_segment:
        segments.append(" ".join(current_segment))

    return segments


def split_chinese_text(text: str, max_chars: int) -> list[str]:
    """Split Chinese (or CJK-heavy) text by sentence boundaries then char count."""
    if not text.strip():
        return []

    parts = re.split(r"([。！？；\n]+)", text)
    sentences: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"[。！？；\n]+", part):
            buf += part
            if buf.strip():
                sentences.append(buf.strip())
            buf = ""
        else:
            buf += part
    if buf.strip():
        sentences.append(buf.strip())

    if not sentences:
        sentences = [text.strip()]

    segments: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                segments.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                segments.append(sentence[i : i + max_chars])
            continue

        candidate = f"{current}{sentence}" if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                segments.append(current)
            current = sentence

    if current:
        segments.append(current)

    return segments


def _is_cjk_heavy(text: str) -> bool:
    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    words = len(text.split())
    return cjk > 0 and (words == 0 or cjk >= words)


def split_text(text: str, max_size: int) -> list[str]:
    """Split text using word-count (Latin) or char/sentence (Chinese) strategy."""
    if _is_cjk_heavy(text):
        return split_chinese_text(text, max_size)
    return split_text_by_word_count(text, max_size)
