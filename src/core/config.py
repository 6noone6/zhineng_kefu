import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing_extensions import Self

# src/core/config.py -> project root (zhineng_kefu/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # Moonshot / Kimi (agent orchestration)
    moonshot_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("MOONSHOT_API_KEY", "KIMI_API_KEY"),
    )
    moonshot_base_url: str = Field(
        default="https://api.moonshot.cn/v1",
        validation_alias=AliasChoices("MOONSHOT_BASE_URL"),
    )
    moonshot_model: str = Field(
        default="kimi-k2.6",
        validation_alias=AliasChoices("MOONSHOT_MODEL"),
    )

    # Hugging Face Hub (embeddings, transformers downloads)
    hf_token: str = Field(
        default="",
        validation_alias=AliasChoices("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"),
    )

    # Local Qwen
    qwen_model_path: str = "Qwen/Qwen3.5-2B"
    lora_path: str = "./models/qwen35_2b_lora"
    rag_backend: Literal["local", "cloud"] = "local"
    max_new_tokens: int = 256

    # RAG
    rag_top_k: int = 3
    chunk_size: int = 128
    knowledge_dir: Path = Path("./data/knowledge")
    chroma_persist_dir: Path = Path("./data/chroma")
    retriever_type: Literal["bm25", "vector", "hybrid"] = "hybrid"
    embedding_model: str = Field(
        default="BAAI/bge-m3",
        validation_alias=AliasChoices("EMBEDDING_MODEL"),
    )
    hybrid_rrf_k: int = Field(
        default=60,
        validation_alias=AliasChoices("HYBRID_RRF_K"),
    )
    agent_max_steps: int = Field(
        default=4,
        validation_alias=AliasChoices("AGENT_MAX_STEPS"),
    )

    # JWT / OAuth
    jwt_secret: str = Field(
        default="change-me-jwt-secret",
        validation_alias=AliasChoices("JWT_SECRET"),
    )
    jwt_algorithm: str = Field(
        default="HS256",
        validation_alias=AliasChoices("JWT_ALGORITHM"),
    )
    jwt_expire_minutes: int = Field(
        default=1440,
        validation_alias=AliasChoices("JWT_EXPIRE_MINUTES"),
    )
    jwt_cookie_name: str = Field(
        default="kefu_token",
        validation_alias=AliasChoices("JWT_COOKIE_NAME"),
    )
    oauth_google_client_id: str = Field(
        default="",
        validation_alias=AliasChoices("OAUTH_GOOGLE_CLIENT_ID"),
    )

    # Database
    database_url: str = "postgresql+asyncpg://kefu:kefu@localhost:5432/kefu"
    redis_url: str = "redis://localhost:6379/0"

    # Environment
    env: Literal["development", "production"] = "development"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_key: str = "change-me-in-production"
    api_key_previous: str = Field(
        default="",
        validation_alias=AliasChoices("API_KEY_PREVIOUS"),
    )
    max_upload_bytes: int = 1_048_576  # 1 MB
    allowed_upload_extensions: tuple[str, ...] = (".txt", ".md")
    metrics_require_auth: bool = True
    qwen_inference_url: str = ""
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
        ]
    )

    # Session
    session_max_history: int = 6
    session_redis_ttl: int = 86400

    # External APIs
    logistics_api_url: str = ""
    logistics_api_key: str = ""
    order_api_url: str = ""
    order_api_key: str = ""
    complaint_webhook_url: str = ""

    # Logging
    log_level: str = "INFO"

    # OpenTelemetry (optional)
    otel_enabled: bool = Field(
        default=False,
        validation_alias=AliasChoices("OTEL_ENABLED"),
    )
    otel_service_name: str = Field(
        default="zhineng-kefu",
        validation_alias=AliasChoices("OTEL_SERVICE_NAME"),
    )
    otel_exporter_endpoint: str = Field(
        default="",
        validation_alias=AliasChoices("OTEL_EXPORTER_OTLP_ENDPOINT"),
    )

    @model_validator(mode="after")
    def resolve_project_paths(self) -> Self:
        for name in ("knowledge_dir", "chroma_persist_dir"):
            path = getattr(self, name)
            if not path.is_absolute():
                setattr(self, name, PROJECT_ROOT / path)
        lora = Path(self.lora_path)
        if not lora.is_absolute():
            self.lora_path = str(PROJECT_ROOT / lora)
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)
            os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", self.hf_token)
        return self

    @property
    def lora_path_exists(self) -> bool:
        return Path(self.lora_path).exists()


@lru_cache
def get_settings() -> Settings:
    return Settings()
