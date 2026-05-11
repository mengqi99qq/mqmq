"""Application settings powered by pydantic-settings."""

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment-backed settings with safe defaults."""

    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=6523)
    cors_origins: str = Field(default="http://localhost:5569,http://127.0.0.1:5569")
    max_upload_mb: int = Field(default=10)
    interview_db_path: str = Field(default="../backend/data/interviews.db")
    mock_interview_chroma_path: str = Field(default="data/mock_interview_chroma")
    mock_interview_rag_topk: int = Field(default=5)
    mock_interview_rag_candidate_topk: int = Field(default=12)
    mock_interview_embedding_model: str = Field(default="text-embedding-3-small")
    mock_interview_session_ttl_minutes: int = Field(default=90)
    mock_interview_rag: bool = Field(default=False)
    mock_interview_plan_timeout_seconds: int = Field(default=120)

    model_provider: Literal["openai", "google_genai", "anthropic"] = Field(default="openai")
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)
    openai_model: str = Field(default="gpt-4o-mini")
    dashscope_ocr_model: str = Field(default="qwen-vl-plus")
    google_api_key: str | None = Field(default=None)
    google_model: str = Field(default="gemini-2.0-flash")
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(default="claude-sonnet-4-5-20250929")

    rate_limit_requests_per_second: float = Field(default=20)
    rate_limit_check_every_n_seconds: float = Field(default=5)
    rate_limit_max_bucket_size: int = Field(default=20)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    def get_cors_origins(self) -> list[str]:
        """Parse comma-separated CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def get_provider_config(self, provider: Literal["openai", "google_genai", "anthropic"]) -> dict:
        """Return model config for the specified provider."""
        if provider == "openai":
            return {
                "model_provider": "openai",
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "model": self.openai_model,
            }
        if provider == "google_genai":
            if self.google_api_key:
                os.environ["GOOGLE_API_KEY"] = self.google_api_key
            return {
                "model_provider": "google_genai",
                "api_key": self.google_api_key,
                "base_url": None,
                "model": self.google_model,
            }
        if provider == "anthropic":
            if self.anthropic_api_key:
                os.environ["ANTHROPIC_API_KEY"] = self.anthropic_api_key
            return {
                "model_provider": "anthropic",
                "api_key": self.anthropic_api_key,
                "base_url": None,
                "model": self.anthropic_model,
            }
        raise ValueError(f"Unsupported model provider: {provider}")

    def get_active_config(self) -> dict:
        """Return currently active model configuration."""
        return self.get_provider_config(self.model_provider)


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
