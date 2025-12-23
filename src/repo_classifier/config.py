"""
Configuration management for Repo Classifier RAG Agent.
Uses pydantic-settings to load from environment variables and .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Required - OpenAI API key for embeddings and LLM
    openai_api_key: str

    # Optional - GitHub token for higher rate limits (5000 vs 60 req/hour)
    github_token: str | None = None

    # Optional - Wallet address for x402 payments
    wallet_address: str | None = None

    # Pricing configuration
    free_tier_enabled: bool = True
    free_tier_requests: int = 5  # Requests per minute in free tier
    price_per_request: float = 0.02  # USD per request

    # Rate limiting
    rate_limit_window: int = 60  # Seconds

    # Performance tuning
    cache_ttl_seconds: int = 3600  # 1 hour default cache TTL
    readme_max_chars: int = 8000  # Truncate README to this length
    chroma_db_path: str = "./data/chroma_db"

    # RAG configuration
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    rag_k_neighbors: int = 3  # Number of similar repos to retrieve


# Global settings instance - imported by other modules
settings = Settings()

