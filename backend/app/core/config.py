from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "reco-commerce-api"
    api_v1_prefix: str = "/api/v1"
    environment: str = "development"

    # Default is for local Docker Postgres
    database_url: str = "postgresql+psycopg2://postgres:admin@localhost:5432/recommendly"

    # Frontend dev server later
    cors_origins: List[str] = ["http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()