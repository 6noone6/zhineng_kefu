from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

import chromadb
from rank_bm25 import BM25Okapi

from src.core.config import Settings, get_settings
from src.rag.chunker import split_text
from src.rag.embeddings import get_chroma_collection
from src.rag.fusion import reciprocal_rank_fusion
from src.rag.index_sync import sync_chroma_collection, write_manifest, knowledge_signature
from src.rag.multilingual import detect_chunk_language
from src.rag import Chunk
from src.rag.tokenizer import tokenize


class Retriever(Protocol):
    def search(self, query: str, top_k: int | None = None) -> list[Chunk]: ...

    async def search_async(self, query: str, top_k: int | None = None) -> list[Chunk]: ...


class BM25Retriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        corpus = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, top_k: int | None = None) -> list[Chunk]:
        if not self._bm25 or not self.chunks:
            return []
        settings = get_settings()
        k = top_k or settings.rag_top_k
        tokenized_query = tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:k]
        return [
            Chunk(
                text=c.text,
                source=c.source,
                score=float(s),
                chunk_id=c.chunk_id,
                lang=c.lang,
            )
            for c, s in ranked
            if s > 0
        ]

    async def search_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        return await asyncio.to_thread(self.search, query, top_k)


class VectorRetriever:
    def __init__(self, collection):
        self._collection = collection

    def search(self, query: str, top_k: int | None = None) -> list[Chunk]:
        settings = get_settings()
        k = top_k or settings.rag_top_k
        if self._collection.count() == 0:
            return []
        results = self._collection.query(query_texts=[query], n_results=k)
        chunks: list[Chunk] = []
        if not results["documents"] or not results["documents"][0]:
            return chunks
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i] if results["metadatas"] else {}
            dist = results["distances"][0][i] if results["distances"] else 0.0
            chunks.append(
                Chunk(
                    text=doc,
                    source=meta.get("source", ""),
                    score=1.0 - dist if dist else 0.0,
                    chunk_id=results["ids"][0][i] if results["ids"] else "",
                    lang=meta.get("lang", ""),
                )
            )
        return chunks

    async def search_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        return await asyncio.to_thread(self.search, query, top_k)


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever):
        self._bm25 = bm25
        self._vector = vector

    def search(self, query: str, top_k: int | None = None) -> list[Chunk]:
        settings = get_settings()
        k = top_k or settings.rag_top_k
        fetch_k = max(k * 2, k + 2)
        bm25_results = self._bm25.search(query, top_k=fetch_k)
        vector_results = self._vector.search(query, top_k=fetch_k)
        return reciprocal_rank_fusion(
            [bm25_results, vector_results],
            rrf_k=settings.hybrid_rrf_k,
            top_k=k,
        )

    async def search_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        return await asyncio.to_thread(self.search, query, top_k)


def load_knowledge_chunks(knowledge_dir: Path, chunk_size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    if not knowledge_dir.exists():
        return chunks
    for path in knowledge_dir.rglob("*"):
        if path.suffix.lower() not in {".txt", ".md"}:
            continue
        content = path.read_text(encoding="utf-8")
        for i, segment in enumerate(split_text(content, chunk_size)):
            source_name = path.name
            chunks.append(
                Chunk(
                    text=segment,
                    source=source_name,
                    chunk_id=f"{path.stem}_{i}",
                    lang=detect_chunk_language(source_name, segment),
                )
            )
    return chunks


def build_retriever(settings: Settings | None = None) -> HybridRetriever | BM25Retriever | VectorRetriever:
    settings = settings or get_settings()
    settings.knowledge_dir.mkdir(parents=True, exist_ok=True)
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_knowledge_chunks(settings.knowledge_dir, settings.chunk_size)
    bm25 = BM25Retriever(chunks)

    if settings.retriever_type == "bm25":
        return bm25

    client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    collection = get_chroma_collection(client, settings)
    collection, _rebuilt = sync_chroma_collection(client, collection, chunks, settings)
    if chunks and not _rebuilt and collection.count() > 0:
        write_manifest(
            settings.chroma_persist_dir,
            signature=knowledge_signature(settings.knowledge_dir),
            chunk_ids=[c.chunk_id for c in chunks],
        )

    vector = VectorRetriever(collection)

    if settings.retriever_type == "vector":
        return vector
    return HybridRetriever(bm25, vector)
