"""Gemini embedding adapter (`gemini-embedding-001`).

Written up front so that switching `EMBEDDER=gemini` is a config change rather than a
code change. Not the default: it needs network and an API key even to ingest.

Two details that are easy to get wrong:

* **Task types matter.** Gemini embeddings are asymmetric — documents and queries must be
  embedded with different task types (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) or
  retrieval quality drops noticeably.
* **Sub-3072 dimensions need re-normalising.** `gemini-embedding-001` is trained with
  Matryoshka representation learning; truncated outputs are not unit-length, and cosine
  distance in Chroma assumes they are.
"""

import logging
import math

from google import genai
from google.genai import types

from server.config import settings

log = logging.getLogger(__name__)

_BATCH = 100


def _unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    return [v / norm for v in vec] if norm else vec


class GeminiEmbedder:
    name = "gemini"

    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("EMBEDDER=gemini requires GEMINI_API_KEY to be set.")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.embed_model_gemini
        self.dim = settings.embed_dim_gemini

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            batch = texts[start : start + _BATCH]
            resp = self._client.models.embed_content(
                model=self._model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.dim,
                ),
            )
            out.extend(_unit(list(e.values)) for e in resp.embeddings)
        return out

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], "RETRIEVAL_QUERY")[0]
