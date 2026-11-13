"""Gemini generation adapter."""

import logging

from google import genai
from google.genai import types

from server.config import settings
from server.llm.base import LLMError

log = logging.getLogger(__name__)


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise LLMError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_model

    def generate(self, system: str, user: str) -> str:
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    # Grounded extraction, not creative writing.
                    temperature=0.1,
                ),
            )
        except Exception as exc:  # noqa: BLE001 - normalise every SDK failure
            log.exception("Gemini generation failed")
            raise LLMError(f"Gemini request failed: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise LLMError("Gemini returned an empty response.")
        return text
