"""ChromaDB embedding functions (BAAI/bge-m3 per README)."""

from __future__ import annotations

import structlog

from src.core.config import Settings, get_settings

logger = structlog.get_logger()

_embedding_fn_cache: dict[str, object] = {}


def release_ml_resources() -> None:
    """Drop cached embedders so CLI scripts can exit (torch/ST keep non-daemon threads)."""
    _embedding_fn_cache.clear()
    try:
        import gc

        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

def get_embedding_function(model_name: str | None = None):
    """Return a SentenceTransformer embedding function, or None if unavailable."""
    settings = get_settings()
    name = model_name or settings.embedding_model

    if name in _embedding_fn_cache:
        return _embedding_fn_cache[name]

    try:
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        ef = SentenceTransformerEmbeddingFunction(model_name=name)
        _embedding_fn_cache[name] = ef
        logger.info("embedding_function_loaded", model=name)
        return ef
    except ImportError:
        logger.warning(
            "sentence_transformers_not_installed",
            hint="pip install sentence-transformers for BAAI/bge-m3 embeddings",
        )
    except Exception as exc:
        logger.warning("embedding_function_load_failed", model=name, error=str(exc))

    return None


def get_chroma_collection(client, settings: Settings | None = None, *, recreate: bool = False):
    """Get or create the knowledge collection with configured embeddings."""
    settings = settings or get_settings()
    name = "knowledge"
    metadata = {"hnsw:space": "cosine"}

    if recreate:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    ef = get_embedding_function(settings.embedding_model)
    if ef is not None:
        try:
            return client.get_or_create_collection(
                name=name,
                embedding_function=ef,
                metadata=metadata,
            )
        except ValueError as exc:
            # Existing collection was created with a different embedder (e.g. Chroma default).
            if "embedding function" not in str(exc).lower():
                raise
            logger.warning(
                "chroma_embedding_conflict_recreate",
                collection=name,
                model=settings.embedding_model,
                error=str(exc),
            )
            try:
                client.delete_collection(name)
            except Exception:
                pass
            return client.get_or_create_collection(
                name=name,
                embedding_function=ef,
                metadata=metadata,
            )
    return client.get_or_create_collection(name=name, metadata=metadata)
