"""
app/core/config.py

Centralized application configuration.

All configuration values are loaded from environment variables (and a local
.env file during development) using pydantic-settings. This avoids hardcoded
values scattered throughout the codebase and makes the application portable
across dev / staging / production environments and Docker containers.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    APP_NAME: str = "PhaseGuard-Layer2"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000

    # --- Database ---
    POSTGRES_USER: str = "phaseguard"
    POSTGRES_PASSWORD: str = "phaseguard_secret"
    POSTGRES_DB: str = "phaseguard_db"
    POSTGRES_HOST: str = "db"
    POSTGRES_PORT: int = 5432

    DATABASE_URL: str = (
        "postgresql+asyncpg://phaseguard:phaseguard_secret@db:5432/phaseguard_db"
    )
    DATABASE_URL_SYNC: str = (
        "postgresql+psycopg2://phaseguard:phaseguard_secret@db:5432/phaseguard_db"
    )

    # --- ML / SpeechBrain ---
    ECAPA_MODEL_SOURCE: str = "evaluation/model_checkpoints/trained_ecapa"
    ECAPA_MODEL_SAVE_DIR: str = "evaluation/model_checkpoints/trained_ecapa"
    EMBEDDING_DIM: int = 192
    TARGET_SAMPLE_RATE: int = 16000

    # --- Verification & Fallback Thresholds ---
    SIMILARITY_THRESHOLD: float = 0.63977
    STEP_UP_LOWER_BOUND: float = 0.45499
    HARD_FAIL_THRESHOLD: float = 0.18061
    MIN_ENROLLMENT_RECORDINGS: int = 3

    # --- Adaptive Score Normalisation (s-norm) ---
    SNORM_PASS_THRESHOLD: float = 3.0
    SNORM_STEP_UP_THRESHOLD: float = 1.5
    SNORM_HARD_FAIL_THRESHOLD: float = 0.0
    IMPOSTOR_COHORT_SIZE: int = 200

    # --- BioHashing (Cancellable Biometrics) ---
    BIOHASH_SEED_KEY: str = "PhaseGuardSecretKey2026"
    BIOHASH_DIM: int = 256

    # --- FAISS Scalable Vector Search ---
    FAISS_INDEX_PATH: str = "./data/faiss_ivfpq.index"
    FAISS_NPROBE: int = 32
    FAISS_M: int = 16
    FAISS_NBITS: int = 8

    # --- Audio Preprocessing & Quality Gating ---
    ENABLE_AUDIO_QUALITY_CHECK: bool = True
    MIN_SPEECH_DURATION_SEC: float = 2.0
    EMBEDDING_CONFIDENCE_THRESHOLD: float = 0.5
    TARGET_NORM_SCALE: float = 10.0

    # --- Liveness & Replay Protection ---
    ENABLE_CHALLENGE_ASR: bool = True
    ENABLE_PERCEPTUAL_REPLAY_CHECK: bool = True
    CHALLENGE_TTL_SEC: int = 60
    AUDIO_HASH_TTL_SEC: int = 86400  # 24 hours

    # --- Observability & Telemetry ---
    ENABLE_OPENTELEMETRY: bool = True
    OTEL_EXPORTER_OTLP_ENDPOINT: str = "http://localhost:4317"

    # --- Risk Engine ---
    LAYER1_FRAUD_THRESHOLD: float = 0.70
    LAYER1_LIVENESS_CHECK_ENABLED: bool = True

    # --- Redis ---
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # --- File handling ---
    TEMP_UPLOAD_DIR: str = "./data/temp_uploads"
    MAX_UPLOAD_SIZE_MB: int = 20

    # --- Logging ---
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        Path(self.TEMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.LOG_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.ECAPA_MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache ensures the .env file is parsed only once and the same
    settings object is reused across the application (cheap dependency
    injection for FastAPI routes).
    """
    settings = Settings()
    settings.ensure_directories()
    return settings


# Convenience module-level singleton for non-FastAPI contexts (e.g. scripts,
# Alembic env.py, Streamlit app).
settings = get_settings()
