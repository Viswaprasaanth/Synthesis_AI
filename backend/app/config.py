from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── LLM ──
    groq_api_key: str
    llm_model: str = "llama-3.1-8b-instant"

    # ── Embeddings ──
    embedding_model: str = "all-MiniLM-L6-v2"

    # ── Qdrant ──
    qdrant_url: str
    qdrant_api_key: str
    collection_name: str = "synthesis_papers"

    # ── Auth ──
    api_key: str = "default-key"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()