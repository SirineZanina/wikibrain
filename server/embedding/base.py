"""Embedder interface.

Nothing outside this package imports `sentence_transformers` or `google.genai` for
embedding. That is what makes swapping the backend a one-env-var change.

The `name` attribute is not cosmetic: it is folded into the Chroma collection name so
that two embedders can never write into the same vector space. See `vectorstore.py`.
"""

from functools import lru_cache
from typing import Protocol, runtime_checkable

from server.config import settings


@runtime_checkable
class Embedder(Protocol):
    #: Short, stable identifier. Becomes part of the collection name.
    name: str
    #: Output dimensionality. Used only for logging and sanity checks.
    dim: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for storage."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query."""
        ...


@lru_cache(maxsize=None)
def get_embedder() -> Embedder:
    """Return the configured embedder. Cached so models load at most once."""
    choice = settings.embedder.strip().lower()

    if choice == "local":
        from server.embedding.local import LocalEmbedder

        return LocalEmbedder()

    if choice == "gemini":
        from server.embedding.gemini import GeminiEmbedder

        return GeminiEmbedder()

    raise ValueError(
        f"Unknown EMBEDDER={settings.embedder!r}. Expected 'local' or 'gemini'."
    )
