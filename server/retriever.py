"""Question -> top-k policy chunks.

Chroma returns *distances*; everything upstream reads *similarity*. The conversion
happens here, once, so that "higher is better" holds everywhere else.
"""

from dataclasses import dataclass

from server import vectorstore
from server.config import settings
from server.embedding import get_embedder


@dataclass(frozen=True)
class Hit:
    text: str
    source: str
    heading: str
    score: float


def retrieve(question: str, top_k: int | None = None) -> list[Hit]:
    k = top_k or settings.top_k
    embedding = get_embedder().embed_query(question)
    result = vectorstore.query(embedding, top_k=k)

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    return [
        Hit(
            text=doc,
            source=meta.get("source", "unknown"),
            heading=meta.get("heading", ""),
            # Chroma cosine distance is 1 - cosine_similarity.
            score=round(1.0 - float(dist), 4),
        )
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]
