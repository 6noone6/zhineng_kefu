#!/usr/bin/env python3
"""Offline RAG retrieval evaluation (recall@k, hit rate)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.core.config import PROJECT_ROOT, get_settings
from src.rag.retriever import build_retriever


def load_eval_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_hit(retrieved_sources: list[str], expected: list[str]) -> bool:
    if not expected:
        return not retrieved_sources
    retrieved = {s.lower() for s in retrieved_sources}
    return any(exp.lower() in retrieved for exp in expected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval quality")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "rag_questions.json",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override RAG_TOP_K")
    args = parser.parse_args()

    settings = get_settings()
    top_k = args.top_k or settings.rag_top_k
    retriever = build_retriever(settings)
    cases = load_eval_set(args.dataset)

    hits = 0
    recalls = 0
    negative_correct = 0
    negative_total = 0

    print(f"Evaluating {len(cases)} questions (top_k={top_k}, retriever={settings.retriever_type})")
    print("-" * 72)

    for case in cases:
        question = case["question"]
        expected = case.get("expected_sources", [])
        chunks = retriever.search(question, top_k=top_k)
        sources = [c.source for c in chunks]
        has_chunks = len(chunks) > 0

        if not expected:
            negative_total += 1
            ok = not has_chunks
            negative_correct += int(ok)
            status = "OK" if ok else "FAIL (false positive)"
        else:
            ok = source_hit(sources, expected)
            recalls += int(ok)
            hits += int(has_chunks)
            status = "HIT" if ok else "MISS"

        print(f"[{status}] {question[:40]:<40} -> {sources[:3]}")

    n_positive = len(cases) - negative_total
    hit_rate = hits / n_positive if n_positive else 0.0
    recall_at_k = recalls / n_positive if n_positive else 0.0
    neg_precision = negative_correct / negative_total if negative_total else 1.0

    print("-" * 72)
    print(f"Retrieval hit rate:     {hit_rate:.1%} ({hits}/{n_positive})")
    print(f"Source recall@{top_k}:   {recall_at_k:.1%} ({recalls}/{n_positive})")
    print(f"Negative precision:     {neg_precision:.1%} ({negative_correct}/{negative_total})")


if __name__ == "__main__":
    main()
