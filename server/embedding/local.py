"""Local sentence-transformers embedder.

Runs in-process on CPU. No API key, no network at query time, no rate limits.
The model is ~90MB and is loaded once per process.
"""

import logging

from sentence_transformers import SentenceTransformer

from server.config import settings

log = logging.getLogger(__name__)


class LocalEmbedder:
    name = "minilm"

    def __init__(self) -> None:
        log.info("Loading local embedding model %s", settings.embed_model_local)
        self._model = SentenceTransformer(settings.embed_model_local)
        # Renamed in sentence-transformers 3.x; keep working on older versions too.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]
