"""Text helpers for model output sanitization."""

from __future__ import annotations

import re

_LT, _GT = chr(60), chr(62)
_OPEN_THINK = _LT + "think" + _GT
_CLOSE_THINK = _LT + "/think" + _GT
_OPEN_RT = _LT + "redacted_thinking" + _GT
_CLOSE_RT = _LT + "/redacted_thinking" + _GT

# Kimi / reasoning models may emit think blocks in various tag styles.
# Do NOT strip bare backticks — that removes order IDs / paths / short code.
_THINK_BLOCK_RE = re.compile(
    re.escape(_OPEN_THINK)
    + r"[\s\S]*?"
    + re.escape(_CLOSE_THINK)
    + "|"
    + re.escape(_OPEN_RT)
    + r"[\s\S]*?"
    + re.escape(_CLOSE_RT)
    + "|"
    + re.escape(_OPEN_THINK)
    + r"[\s\S]*?"
    + re.escape(_CLOSE_RT)
    + "|"
    + re.escape(_OPEN_RT)
    + r"[\s\S]*?"
    + re.escape(_CLOSE_THINK),
    re.IGNORECASE,
)
_THINK_OPEN_RE = re.compile(
    re.escape(_OPEN_THINK)
    + r"[\s\S]*$"
    + "|"
    + re.escape(_OPEN_RT)
    + r"[\s\S]*$",
    re.IGNORECASE,
)
# Local Qwen may continue generating fake multi-turn dialogue after the real answer.
_LEAKED_TURN_RE = re.compile(
    r"\n(?:user|assistant)\s*(?:\n|$)|<\|im_start\|>(?:user|assistant)\b",
    re.IGNORECASE,
)


def strip_think_blocks(text: str) -> str:
    """Remove model reasoning blocks from user-visible reply text."""
    if not text:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_OPEN_RE.sub("", cleaned)
    return cleaned.strip()


def strip_leaked_dialogue(text: str) -> str:
    """Truncate fabricated follow-up turns (user/assistant) appended after the answer."""
    if not text:
        return ""
    match = _LEAKED_TURN_RE.search(text)
    if match:
        return text[: match.start()].strip()
    return text.strip()


def sanitize_assistant_reply(text: str) -> str:
    """Full cleanup for assistant messages shown to users."""
    return strip_leaked_dialogue(strip_think_blocks(text))
