from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- OpenAI ---------------------------------------------------------
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o"
    embedding_dimensions: int = 1536

    # --- Pinecone -------------------------------------------------------
    pinecone_api_key: str = ""
    pinecone_index_name: str = "financial-research"

    # --- Chunking -------------------------------------------------------
    chunk_size: int = 600
    chunk_overlap: int = 100

    # --- Retrieval ------------------------------------------------------
    retrieval_top_k: int = 5
    rerank_initial_k: int = 10
    rerank_enabled: bool = True

    # --- Cache ----------------------------------------------------------
    cache_max_size: int = 128
    cache_ttl_seconds: int = 600

    # --- Security -------------------------------------------------------
    api_key: str = ""
    cors_origins: str = "*"
    max_upload_mb: int = 50

    # --- App ------------------------------------------------------------
    log_level: str = "INFO"
    app_title: str = "Financial Research Assistant"
    app_version: str = "1.0.0"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()
