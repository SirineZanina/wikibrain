"""Markdown -> chunks -> embeddings -> Chroma.

Splitting is structure-first: policy documents are already organised into numbered
sections, and those sections are the natural retrieval unit. We only fall back to
character splitting for sections that are too long to embed well on their own.

Keeping the heading attached to its body matters twice over: it gives the embedding
topical context it would otherwise lose, and it gives the answer a citation string.

Run:  python -m server.ingest [--rebuild]
"""

import argparse
import logging
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter

from server import vectorstore
from server.config import settings
from server.embedding import get_embedder

log = logging.getLogger(__name__)

# Split before any level-2 or level-3 ATX heading, keeping the heading itself.
_HEADING_RE = re.compile(r"(?m)^(?=#{2,3} )")


def split_sections(markdown: str) -> list[tuple[str, str]]:
    """Return (heading, text) pairs. Text includes the heading line."""
    blocks = [b.strip() for b in _HEADING_RE.split(markdown) if b.strip()]
    sections: list[tuple[str, str]] = []
    for block in blocks:
        first_line = block.splitlines()[0].strip()
        heading = (
            first_line.lstrip("#").strip()
            if first_line.startswith("#")
            else "Document header"
        )
        sections.append((heading, block))
    return sections


def chunk_document(markdown: str, source: str) -> list[dict]:
    """Split one document into chunk records ready for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    records: list[dict] = []
    for heading, text in split_sections(markdown):
        pieces = (
            [text]
            if len(text) <= settings.chunk_size
            else splitter.split_text(text)
        )
        for piece in pieces:
            # Re-attach the heading to continuation pieces, which would otherwise
            # lose all topical context and retrieve poorly.
            body = piece if piece.lstrip().startswith("#") else f"## {heading}\n\n{piece}"
            index = len(records)
            records.append(
                {
                    "id": f"{source}::{index}",
                    "text": body,
                    "metadata": {
                        "source": source,
                        "heading": heading,
                        "chunk_index": index,
                    },
                }
            )
    return records


def build_records() -> list[dict]:
    paths = sorted(settings.knowledge_dir.glob("*.md"))
    if not paths:
        raise RuntimeError(f"No markdown files found in {settings.knowledge_dir}")

    records: list[dict] = []
    for path in paths:
        doc_records = chunk_document(path.read_text(encoding="utf-8"), path.name)
        log.info("%s -> %d chunks", path.name, len(doc_records))
        records.extend(doc_records)
    return records


def ingest(rebuild: bool = False) -> int:
    """Embed and store every policy document. Returns the chunk count."""
    if rebuild:
        vectorstore.reset()

    records = build_records()

    # Upsert alone is not enough to make re-ingest idempotent. Chunk ids are positional
    # (`source::3`), so a document edited down to fewer chunks leaves its surplus ids
    # behind holding stale text - still indexed, still retrievable. Clear each document
    # before re-adding it.
    if not rebuild:
        for source in sorted({r["metadata"]["source"] for r in records}):
            vectorstore.delete_by_source(source)

    embedder = get_embedder()
    log.info("Embedding %d chunks with %s (dim=%d)", len(records), embedder.name, embedder.dim)

    embeddings = embedder.embed_texts([r["text"] for r in records])
    vectorstore.add(
        ids=[r["id"] for r in records],
        documents=[r["text"] for r in records],
        embeddings=embeddings,
        metadatas=[r["metadata"] for r in records],
    )

    total = vectorstore.count()
    log.info("Collection %s now holds %d chunks", vectorstore.collection_name(), total)
    return total


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest policy documents into Chroma.")
    parser.add_argument(
        "--rebuild", action="store_true", help="Drop the collection before ingesting."
    )
    args = parser.parse_args()

    total = ingest(rebuild=args.rebuild)

    sample = vectorstore.get_collection().peek(limit=1)
    print(f"\nTotal chunks: {total}")
    if sample["documents"]:
        print(f"Sample metadata: {sample['metadatas'][0]}")
        print("Sample text:\n" + sample["documents"][0][:400])


if __name__ == "__main__":
    main()
