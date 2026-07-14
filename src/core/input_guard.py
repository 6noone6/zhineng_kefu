"""User input validation and lightweight prompt-injection heuristics."""

from __future__ import annotations

import re

# Common instruction-hijack patterns (heuristic, not a full safety filter).
_INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(all\s+)?(previous|above)\s+instructions",
        r"disregard\s+(all\s+)?(previous|prior)\s+",
        r"forget\s+(everything|all)\s+you\s+(were\s+)?told",
        r"you\s+are\s+now\s+(a|an)\s+",
        r"new\s+system\s+prompt",
        r"<\s*/?\s*system\s*>",
        r"<\|im_start\|>system",
        r"###\s*system",
        r"override\s+(the\s+)?(system|safety)\s+",
        r"reveal\s+(your\s+)?(system\s+)?prompt",
    )
)

_MAX_CONTROL_CHARS = 32


def contains_prompt_injection(text: str) -> bool:
    """Return True if text matches known injection heuristics."""
    if not text:
        return False
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    control = sum(1 for ch in text if ord(ch) < 32 and ch not in "\n\r\t")
    return control > _MAX_CONTROL_CHARS


def validate_user_message(text: str, *, max_length: int = 2000) -> str:
    """
    Normalize and validate user message. Raises ValueError on rejection.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Message cannot be empty")
    if len(cleaned) > max_length:
        raise ValueError(f"Message exceeds maximum length of {max_length}")
    if contains_prompt_injection(cleaned):
        raise ValueError(
            "Message contains disallowed patterns. Please rephrase your question."
        )
    return cleaned
