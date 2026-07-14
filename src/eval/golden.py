"""Golden QA dataset loader for evaluation."""

from __future__ import annotations

import json
from pathlib import Path

GOLDEN_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "eval" / "golden_qa.jsonl"


def load_golden_cases() -> list[dict]:
    cases: list[dict] = []
    if not GOLDEN_PATH.exists():
        return cases
    with GOLDEN_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cases.append(json.loads(line))
    return cases
