"""
Configuration settings for the FastAPI backend.
"""
import os
from typing import List, Optional
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "demand-forecasting-api"
    app_env: str = Field(default="development", alias="APP_ENV")
    debug: bool = Field(default=True, alias="DEBUG")

    # API
    api_host: str = Field(default="127.0.0.1", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_prefix: str = "/api"

    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:8501"],
        alias="CORS_ORIGINS"
    )

    # File upload
    max_upload_size: int = Field(default=100 * 1024 * 1024, alias="MAX_UPLOAD_SIZE")  # 100MB
    upload_dir: str = Field(default="./uploads", alias="UPLOAD_DIR")

    # Job queue
    job_queue_max_size: int = Field(default=100, alias="JOB_QUEUE_MAX_SIZE")
    job_worker_count: int = Field(default=2, alias="JOB_WORKER_COUNT")

    # Model registry
    model_registry_dir: Optional[str] = Field(default=None, alias="MODEL_REGISTRY_DIR")

    # External services (for future phases)
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")
    redis_url: Optional[str] = Field(default=None, alias="REDIS_URL")
    s3_endpoint: Optional[str] = Field(default=None, alias="S3_ENDPOINT")
    s3_bucket: Optional[str] = Field(default=None, alias="S3_BUCKET")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export commonly used settings
settings = get_settings()