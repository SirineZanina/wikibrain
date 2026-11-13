"""LLM provider interface.

Only one implementation exists today (Gemini). The interface still earns its place:
`rag.py` never imports an SDK, so the generation backend is replaceable without
touching the retrieval or prompt-assembly logic.
"""

from functools import lru_cache
from typing import Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Raised when the upstream provider fails. Surfaced to the API as a 502."""


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def generate(self, system: str, user: str) -> str:
        ...


@lru_cache(maxsize=None)
def get_provider() -> LLMProvider:
    from server.llm.gemini import GeminiProvider

    return GeminiProvider()
