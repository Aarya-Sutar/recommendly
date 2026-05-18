from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "reco-commerce-api"
    api_v1_prefix: str = "/api/v1"
    environment: str = "development"

    database_url: str = "postgresql+psycopg2://postgres:admin@localhost:5432/recommendly"

    cors_origins: List[str] = ["http://localhost:3000"]

    # Project paths
    project_root: Path = Path(__file__).resolve().parents[3]
    ml_root: Path = project_root / "ml"
    ml_models_dir: Path = ml_root / "models" / "als"
    ml_splits_dir: Path = ml_root / "data" / "splits"
    ml_artifacts_dir: Path = ml_root / "data" / "processed" / "artifacts"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()