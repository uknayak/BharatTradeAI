from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "BharatTradeAI"
    environment: str = "development"
    log_level: str = "INFO"

    postgres_db: str = "bharattradeai"
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    database_url: str = "postgresql://postgres:postgres@localhost:5432/bharattradeai"

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0

    nse_base_url: str = "https://www.nseindia.com"
    bse_base_url: str = "https://www.bseindia.com"
    news_data_api_key: str = ""

    discord_webhook_url: str = ""
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    broker_provider: str = "upstox"
    upstox_api_key: str = ""
    upstox_api_secret: str = ""
    upstox_redirect_uri: str = ""
    upstox_code: str = ""
    upstox_access_token: str = ""

    llm_provider: str = "ollama"
    model_name: str = Field(
        default="gemma3:latest",
        validation_alias=AliasChoices("OLLAMA_MODEL", "MODEL_NAME"),
    )
    ollama_base_url: str = "http://localhost:11434"
    rag_embedding_model: str = "nomic-embed-text"
    rag_vector_store_path: Path = BASE_DIR / ".data" / "rag" / "chroma"

    secret_key: str = "change-me-in-production"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
        protected_namespaces=("settings_",),
    )


settings = Settings()
