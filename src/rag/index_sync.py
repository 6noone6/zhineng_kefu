from __future__ import annotations

import hashlib
import json
from pathlib import Path

import chromadb

from src.core.config import Settings
from src.rag import Chunk
from src.rag.embeddings import get_chroma_collection
from src.rag.multilingual import detect_chunk_language

MANIFEST_NAME = ".knowledge_manifest.json"


def knowledge_signature(knowledge_dir: Path) -> str:
    """Fingerprint knowledge files (name + mtime + size) for sync detection."""
    parts: list[str] = []
    if knowledge_dir.exists():
        for path in sorted(knowledge_dir.rglob("*")):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            stat = path.stat()
            parts.append(f"{path.name}:{stat.st_mtime_ns}:{stat.st_size}")
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return digest


def _manifest_path(chroma_dir: Path) -> Path:
    return chroma_dir / MANIFEST_NAME


def read_stored_manifest(chroma_dir: Path) -> dict | None:
    path = _manifest_path(chroma_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def write_manifest(chroma_dir: Path, *, signature: str, chunk_ids: list[str]) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "signature": signature,
        "chunk_count": len(chunk_ids),
        "chunk_ids": sorted(chunk_ids),
    }
    _manifest_path(chroma_dir).write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def chroma_chunk_ids(collection) -> set[str]:
    count = collection.count()
    if count == 0:
        return set()
    result = collection.get(include=[])
    return set(result.get("ids") or [])


def chroma_needs_rebuild(
    collection,
    chunks: list[Chunk],
    signature: str,
    chroma_dir: Path,
) -> bool:
    expected_ids = {c.chunk_id for c in chunks}
    actual_ids = chroma_chunk_ids(collection)
    manifest = read_stored_manifest(chroma_dir)

    if collection.count() == 0 and chunks:
        return True
    if not chunks and collection.count() > 0:
        return True
    if expected_ids != actual_ids:
        return True
    if not manifest:
        return True
    if manifest.get("signature") != signature:
        return True
    if manifest.get("chunk_count") != len(chunks):
        return True
    return False


def sync_chroma_collection(
    client: chromadb.PersistentClient,
    collection,
    chunks: list[Chunk],
    settings: Settings,
    *,
    force: bool = False,
) -> tuple[object, bool]:
    """
    Ensure Chroma matches on-disk knowledge. Returns (collection, rebuilt).
    BM25 is always rebuilt from files in build_retriever; this keeps vector side aligned.
    """
    signature = knowledge_signature(settings.knowledge_dir)
    rebuild = force or chroma_needs_rebuild(
        collection, chunks, signature, settings.chroma_persist_dir
    )

    if not rebuild:
        return collection, False

    collection = get_chroma_collection(client, settings, recreate=True)
    if chunks:
        collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "lang": c.lang or detect_chunk_language(c.source, c.text)} for c in chunks],
        )
    write_manifest(
        settings.chroma_persist_dir,
        signature=signature,
        chunk_ids=[c.chunk_id for c in chunks],
    )
    return collection, True
