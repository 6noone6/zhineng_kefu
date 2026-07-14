#!/usr/bin/env python3
"""Rebuild knowledge base vector index."""

from __future__ import annotations

import os
import sys
import time

from src.core.config import get_settings
from src.rag.embeddings import release_ml_resources
from src.rag.indexer import rebuild_index


def main() -> int:
    settings = get_settings()
    print(f"Rebuilding index from {settings.knowledge_dir} ...")
    print(f"Embedding model: {settings.embedding_model} (may take 1-3 min on CPU)")
    sys.stdout.flush()

    started = time.perf_counter()
    count = rebuild_index(settings)
    elapsed = time.perf_counter() - started

    print(f"Indexed {count} chunks from {settings.knowledge_dir} ({elapsed:.1f}s)")
    sys.stdout.flush()

    release_ml_resources()
    # sentence-transformers / chromadb spawn non-daemon threads that block normal exit
    os._exit(0)


if __name__ == "__main__":
    main()
