"""Trading Intelligence Platform - Core Configuration Engine.

Handles environment-driven settings, path resolution, validation,
and default application configurations.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global system configuration settings with environment variable overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Base Paths
    BASE_DIR: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    # Application Environment
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = "development"
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_JSON: bool = False
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # TimescaleDB / PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "trading_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres_password_dev"
    POSTGRES_POOL_SIZE: int = 10
    POSTGRES_MAX_OVERFLOW: int = 20
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres_password_dev@localhost:5432/trading_db"

    # SQLite fallback for standalone local development / unit tests
    SQLITE_DATABASE_PATH: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "data" / "trading.db"
    )
    MODEL_STORAGE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "models_store"
    )

    # Redis Cache & Bus
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""
    USE_IN_MEMORY_CACHE_FALLBACK: bool = True

    # MLflow Tracking
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_ARTIFACT_ROOT: str = "./mlartifacts"

    # Market Data Ingestion
    DEFAULT_DATA_PROVIDER: str = "coinbase_ws"
    DEFAULT_SYMBOLS: list[str] | str = ["BTC-USD", "ETH-USD", "SOL-USD"]
    STALE_DATA_THRESHOLD_SECONDS: float = 15.0
    WS_RECONNECT_INITIAL_DELAY_SECONDS: float = 1.0
    WS_RECONNECT_MAX_DELAY_SECONDS: float = 30.0
    WS_MAX_RECONNECT_ATTEMPTS: int = 10
    PING_INTERVAL_SECONDS: float = 20.0
    PING_TIMEOUT_SECONDS: float = 10.0

    # Candle Engine
    PRIMARY_TIMEFRAME: str = "1m"
    ENABLED_TIMEFRAMES: list[str] | str = ["1m", "5m", "15m", "1h", "4h", "1d"]

    # Risk Engine Limits
    MAX_RISK_PER_TRADE_PCT: float = 0.01
    MAX_PORTFOLIO_EXPOSURE_PCT: float = 0.20
    MAX_DAILY_LOSS_PCT: float = 0.03
    MAX_DRAWDOWN_LIMIT_PCT: float = 0.10
    CIRCUIT_BREAKER_STALE_SECS: float = 30.0

    @field_validator("DEFAULT_SYMBOLS", mode="after")
    @classmethod
    def parse_symbols(cls, v: list[str] | str) -> list[str]:
        if isinstance(v, str):
            return [s.strip().upper() for s in v.split(",") if s.strip()]
        return [s.upper() for s in v]

    @field_validator("ENABLED_TIMEFRAMES", mode="after")
    @classmethod
    def parse_timeframes(cls, v: list[str] | str) -> list[str]:
        if isinstance(v, str):
            return [tf.strip() for tf in v.split(",") if tf.strip()]
        return v

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_testing(self) -> bool:
        return self.ENVIRONMENT == "test"


# Global singleton instance
settings = Settings()
