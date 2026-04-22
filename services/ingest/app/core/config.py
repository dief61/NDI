# services/ingest/app/core/config.py
# Zentrale Konfiguration – liest Werte aus .env

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL
    postgres_user: str = "mnr"
    postgres_password: str
    postgres_db: str = "mnr_db"
    postgres_port: int = 5432
    postgres_host: str = "localhost"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # MinIO
    minio_root_user: str = "mnr_admin"
    minio_root_password: str
    minio_port: int = 9000
    minio_host: str = "localhost"
    minio_default_bucket: str = "mnr-artefakte"

    @property
    def minio_endpoint(self) -> str:
        return f"{self.minio_host}:{self.minio_port}"

    # Apache Tika
    tika_port: int = 9998
    tika_host: str = "localhost"

    @property
    def tika_server_url(self) -> str:
        return f"http://{self.tika_host}:{self.tika_port}"

    # Embedding
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dimension: int = 1024

    # Allgemein
    project_name: str = "ndi"
    log_level: str = "INFO"


settings = Settings()
