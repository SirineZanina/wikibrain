"""Single source of truth for configuration.

Everything tunable lives here and is read from the environment (or `.env`).
No other module should call `os.getenv`.
"""

from functools import cached_property
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # --- LLM (generation) ---
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # --- Embeddings ---
    # "local" -> sentence-transformers | "gemini" -> gemini-embedding-001
    embedder: str = "local"
    embed_model_local: str = "all-MiniLM-L6-v2"
    embed_model_gemini: str = "gemini-embedding-001"
    embed_dim_gemini: int = 768

    # --- Storage ---
    knowledge_dir: Path = SERVER_DIR / "knowledge"
    chroma_path: Path = SERVER_DIR / "chroma"
    collection_prefix: str = "policies"

    # --- Retrieval tuning ---
    top_k: int = 4
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- API ---
    cors_origins: str = "http://localhost:3000"
    debug_prompt: bool = False

    @cached_property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
