"""FastAPI application."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from server import vectorstore
from server.config import settings
from server.llm.base import LLMError

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger("wikibrain")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from server.ingest import ingest

    existing = vectorstore.count()
    if existing == 0:
        log.info("Collection %s is empty - ingesting", vectorstore.collection_name())
        existing = ingest()
    # Logged explicitly: a silent zero here is the difference between "nothing to do"
    # and "ingest is broken", and they look identical from the outside.
    log.info("Ready. Collection=%s chunks=%d", vectorstore.collection_name(), existing)
    yield


app = FastAPI(title="WikiBrain", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    source: str
    heading: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "collection": vectorstore.collection_name(),
        "chunks": vectorstore.count(),
        "embedder": settings.embedder,
        "model": settings.gemini_model,
    }


@app.post("/ingest")
def run_ingest(rebuild: bool = False) -> dict:
    from server.ingest import ingest

    return {"chunks": ingest(rebuild=rebuild)}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    from server.rag import answer

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Question must not be empty.")

    try:
        result = answer(question)
    except LLMError as exc:
        # Upstream failure, not a client error - and never leak a stack trace.
        log.error("Generation failed: %s", exc)
        raise HTTPException(status_code=502, detail="The language model is unavailable.")

    return ChatResponse(**result)


@app.get("/debug/retrieve")
def debug_retrieve(q: str = Query(min_length=1), k: int | None = None) -> dict:
    """Retrieval without generation. The first place to look when an answer is wrong."""
    from server.retriever import retrieve

    hits = retrieve(q, top_k=k)
    return {
        "question": q,
        "hits": [
            {
                "source": h.source,
                "heading": h.heading,
                "score": h.score,
                "preview": h.text[:300],
            }
            for h in hits
        ],
    }
