from __future__ import annotations

import asyncio
from pathlib import Path

import chromadb

from src.core.config import Settings, get_settings
from src.rag.chunker import split_text
from src.rag.embeddings import get_chroma_collection
from src.rag.index_sync import knowledge_signature, write_manifest
from src.rag import Chunk


def rebuild_index(settings: Settings | None = None) -> int:
    """Rebuild Chroma vector index from knowledge directory."""
    settings = settings or get_settings()
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[Chunk] = []
    for path in settings.knowledge_dir.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        content = path.read_text(encoding="utf-8")
        for i, segment in enumerate(split_text(content, settings.chunk_size)):
            chunks.append(
                Chunk(
                    text=segment,
                    source=str(path.name),
                    chunk_id=f"{path.stem}_{i}",
                )
            )

    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    collection = get_chroma_collection(client, settings, recreate=True)

    if chunks:
        collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source} for c in chunks],
        )

    write_manifest(
        settings.chroma_persist_dir,
        signature=knowledge_signature(settings.knowledge_dir),
        chunk_ids=[c.chunk_id for c in chunks],
    )
    return len(chunks)


async def rebuild_index_async(settings: Settings | None = None) -> int:
    return await asyncio.to_thread(rebuild_index, settings)


def save_uploaded_file(
    filename: str, content: bytes, settings: Settings | None = None
) -> Path:
    settings = settings or get_settings()
    safe_name = Path(filename).name
    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("Invalid filename")

    ext = Path(safe_name).suffix.lower()
    if ext not in settings.allowed_upload_extensions:
        raise ValueError(f"File type not allowed. Allowed: {settings.allowed_upload_extensions}")

    if len(content) > settings.max_upload_bytes:
        raise ValueError(f"File too large. Max size: {settings.max_upload_bytes} bytes")

    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.knowledge_dir / safe_name
    dest.write_bytes(content)
    return dest
