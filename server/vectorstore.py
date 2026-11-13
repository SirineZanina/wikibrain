"""ChromaDB persistence.

The collection name is scoped by embedder (`policies-minilm`, `policies-gemini`) so that
changing `EMBEDDER` lands in a different, empty collection and triggers a clean re-ingest,
instead of silently querying a 384-dim index with a 768-dim vector.
"""

import logging

import chromadb
from chromadb.api.models.Collection import Collection

from server.config import settings
from server.embedding import get_embedder

log = logging.getLogger(__name__)

_client: chromadb.ClientAPI | None = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return _client


def collection_name() -> str:
    return f"{settings.collection_prefix}-{get_embedder().name}"


def get_collection() -> Collection:
    return _get_client().get_or_create_collection(
        name=collection_name(),
        # Explicit: the default is L2, and every score in this codebase assumes cosine.
        metadata={"hnsw:space": "cosine"},
    )


def count() -> int:
    return get_collection().count()


def add(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    """Upsert so that re-running ingest replaces rather than duplicates."""
    get_collection().upsert(
        ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas
    )


def delete_by_source(source: str) -> None:
    """Remove every chunk belonging to one document."""
    get_collection().delete(where={"source": source})


def query(embedding: list[float], top_k: int) -> dict:
    return get_collection().query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )


def reset() -> None:
    """Drop the collection for the active embedder. Used by `ingest --rebuild`."""
    try:
        _get_client().delete_collection(collection_name())
    except Exception:
        log.info("No existing collection %s to delete", collection_name())
