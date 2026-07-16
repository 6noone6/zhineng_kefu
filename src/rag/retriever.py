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
from src.rag.postprocess import (
    diversify_by_source,
    expand_domain_queries,
    prefer_gulf_warranty_sources,
    reject_below_min_score,
)
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
        )[: max(k * 3, k + 2)]
        chunks = [
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
        chunks = prefer_gulf_warranty_sources(chunks, query)
        return diversify_by_source(
            chunks,
            max_per_source=settings.rag_max_per_source,
            top_k=k,
        )

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
        fetch_n = max(k * 3, k + 2)
        results = self._collection.query(query_texts=[query], n_results=fetch_n)
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
        chunks = prefer_gulf_warranty_sources(chunks, query)
        chunks = reject_below_min_score(chunks, min_score=settings.rag_min_score)
        if not chunks:
            return []
        return diversify_by_source(
            chunks,
            max_per_source=settings.rag_max_per_source,
            top_k=k,
        )

    async def search_async(self, query: str, top_k: int | None = None) -> list[Chunk]:
        return await asyncio.to_thread(self.search, query, top_k)


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, vector: VectorRetriever):
        self._bm25 = bm25
        self._vector = vector

    def _raw_bm25(self, query: str, top_k: int) -> list[Chunk]:
        """BM25 hits without diversity/reject (fusion inputs)."""
        if not self._bm25._bm25 or not self._bm25.chunks:
            return []
        tokenized_query = tokenize(query)
        scores = self._bm25._bm25.get_scores(tokenized_query)
        ranked = sorted(
            zip(self._bm25.chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
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

    def _raw_vector(self, query: str, top_k: int) -> list[Chunk]:
        """Vector hits without diversity/reject (fusion + confidence inputs)."""
        collection = self._vector._collection
        if collection.count() == 0:
            return []
        results = collection.query(query_texts=[query], n_results=top_k)
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

    def search(self, query: str, top_k: int | None = None) -> list[Chunk]:
        settings = get_settings()
        k = top_k or settings.rag_top_k
        fetch_k = max(k * 3, k + 4)
        variants = expand_domain_queries(query)

        bm25_lists: list[list[Chunk]] = []
        vector_lists: list[list[Chunk]] = []
        best_vector_score = 0.0
        for variant in variants:
            bm25_lists.append(self._raw_bm25(variant, fetch_k))
            v_hits = self._raw_vector(variant, fetch_k)
            vector_lists.append(v_hits)
            if v_hits:
                best_vector_score = max(best_vector_score, float(v_hits[0].score))

        # Semantic confidence gate: off-topic queries usually peak below rag_min_score.
        if settings.rag_min_score > 0 and best_vector_score < settings.rag_min_score:
            return []

        ranked_lists = [lst for lst in bm25_lists + vector_lists if lst]
        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(
            ranked_lists,
            rrf_k=settings.hybrid_rrf_k,
            top_k=max(fetch_k * len(variants), k * 4),
        )
        fused = prefer_gulf_warranty_sources(fused, query)
        return diversify_by_source(
            fused,
            max_per_source=settings.rag_max_per_source,
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
