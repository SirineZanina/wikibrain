"""Retrieve -> assemble grounded context -> generate.

The system prompt is the safety rail: without an explicit instruction to stay inside the
context and to admit ignorance, the model will happily answer policy questions from its
own priors, which is exactly the failure this project exists to avoid.
"""

import logging
import re

from server.config import settings
from server.llm import get_provider
from server.retriever import Hit, retrieve

log = logging.getLogger(__name__)

IDK = "I don't know based on the policies I have."

SYSTEM_PROMPT = f"""You are WikiBrain, an assistant that answers questions about company \
policy for Aster Dynamics employees.

Rules you must follow:
1. Answer ONLY using the numbered context passages provided. Never use outside knowledge \
about how companies usually work.
2. Cite the source filename in brackets after each fact you state, e.g. [pto-and-leave-policy.md].
3. If the context does not contain the answer, reply exactly: "{IDK}" \
Do not guess, and do not offer a general answer instead.
4. If the context partly answers the question, give the part that is supported and say \
plainly which part is not covered.
5. Quote exact figures, deadlines, and thresholds as written. Never round or approximate them.
6. Be concise and direct. No preamble."""


def build_context(hits: list[Hit]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {h.source} — section: {h.heading})\n{h.text}"
        for i, h in enumerate(hits, start=1)
    )


def build_user_prompt(question: str, context: str) -> str:
    return f"CONTEXT:\n{context}\n\nQUESTION: {question}"


def is_refusal(text: str) -> bool:
    """Did the model decline to answer?

    Compared loosely on purpose. An exact match would miss a stray period or a
    capitalisation change, and attaching citations to a refusal is the worst outcome
    available - it makes "I don't know" look sourced.
    """
    normalized = re.sub(r"[^a-z ]+", "", text.lower()).strip()
    return normalized.startswith("i dont know based on the policies")


def dedupe_sources(hits: list[Hit]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    sources: list[dict] = []
    for h in hits:
        key = (h.source, h.heading)
        if key not in seen:
            seen.add(key)
            sources.append({"source": h.source, "heading": h.heading, "score": h.score})
    return sources


def answer(question: str, top_k: int | None = None) -> dict:
    hits = retrieve(question, top_k=top_k)

    if not hits:
        return {"answer": IDK, "sources": []}

    context = build_context(hits)
    user_prompt = build_user_prompt(question, context)

    if settings.debug_prompt:
        # Most RAG bugs are visible in this string before they look like model failures.
        log.info("----- PROMPT -----\n%s\n%s\n------------------", SYSTEM_PROMPT, user_prompt)

    text = get_provider().generate(SYSTEM_PROMPT, user_prompt)

    # If the model declined, the citations would be misleading.
    sources = [] if is_refusal(text) else dedupe_sources(hits)
    return {"answer": text, "sources": sources}
