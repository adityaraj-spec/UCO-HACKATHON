"""
app/core/config.py

Centralized application configuration for PhaseGuard Layer 2.

All configuration values are loaded from environment variables (and a local
.env file during development) using pydantic-settings. This covers all Layer 2
components: voice auth, encryption, Redis, FAISS, anti-spoofing, JWT, Kafka,
consent, emergency access, and fraud integration.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings, loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application                                                          #
    # ------------------------------------------------------------------ #
    APP_NAME: str = "PhaseGuard-Layer2"
    APP_ENV: str = "development"
    APP_DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    APP_VERSION: str = "2.0.0"

    # ------------------------------------------------------------------ #
    # Database                                                             #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # ML / SpeechBrain / ECAPA-TDNN                                       #
    # ------------------------------------------------------------------ #
    ECAPA_MODEL_SOURCE: str = "speechbrain/spkrec-ecapa-voxceleb"
    ECAPA_MODEL_SAVE_DIR: str = "./pretrained_models/ecapa"
    EMBEDDING_DIM: int = 192
    TARGET_SAMPLE_RATE: int = 16000

    # ------------------------------------------------------------------ #
    # Verification / Thresholds                                            #
    # ------------------------------------------------------------------ #
    SIMILARITY_THRESHOLD: float = 0.65
    HIGH_CONFIDENCE_THRESHOLD: float = 0.92          # Rolling update trigger
    STEP_UP_LOWER_BOUND: float = 0.05                # [threshold-x, threshold] = step-up zone
    DRIFT_ALERT_DISTANCE: float = 0.15               # Cosine drift from anchor = alert

    # ------------------------------------------------------------------ #
    # Enrollment                                                           #
    # ------------------------------------------------------------------ #
    MIN_ENROLLMENT_SAMPLES: int = 5                  # Required samples for enrollment
    MIN_ENROLLMENT_RECORDINGS: int = 3               # Backward compat alias
    ENROLLMENT_SESSION_TTL_SECONDS: int = 600        # 10-minute enrollment session
    MAX_ROLLING_EMBEDDINGS: int = 10                 # Rolling pool size
    ROLLING_UPDATE_CONSECUTIVE_REQUIRED: int = 3     # Consecutive successes before update
    ROLLING_UPDATE_WINDOW_HOURS: int = 24

    # ------------------------------------------------------------------ #
    # Audio Quality                                                        #
    # ------------------------------------------------------------------ #
    AUDIO_MIN_DURATION_SECONDS: float = 1.5
    AUDIO_MAX_DURATION_SECONDS: float = 10.0
    AUDIO_MIN_SNR_DB: float = 10.0                   # Minimum SNR for acceptance
    AUDIO_ENROLLMENT_MIN_SNR_DB: float = 20.0        # Stricter for enrollment
    AUDIO_MIN_VOICE_RATIO: float = 0.4               # Min fraction of voiced frames
    VAD_AGGRESSIVENESS: int = 2                      # WebRTC VAD aggressiveness 0-3

    # ------------------------------------------------------------------ #
    # Anti-Spoofing                                                        #
    # ------------------------------------------------------------------ #
    ANTISPOOF_PROVIDER: Literal["mock", "aasist", "rawboost"] = "mock"
    ANTISPOOF_REJECT_THRESHOLD: float = 0.5          # Spoof probability > this = reject
    CHALLENGE_TTL_SECONDS: int = 180                 # 3-minute challenge expiry

    # ------------------------------------------------------------------ #
    # Cancelable Biometrics / BioHash                                      #
    # ------------------------------------------------------------------ #
    BIOHASH_MASTER_SECRET: str = "CHANGE_ME_IN_PRODUCTION_BIOHASH_SECRET_32CHARS"
    BIOHASH_PROJECTION_DIM: int = 256                # Output dimension after projection

    # ------------------------------------------------------------------ #
    # Encryption / HSM                                                     #
    # ------------------------------------------------------------------ #
    HSM_PROVIDER: Literal["mock", "aws_cloudhsm", "pkcs11"] = "mock"
    HSM_MASTER_KEY_ID: str = "phaseguard-master-key-v1"
    ENCRYPTION_KEY_ROTATION_DAYS: int = 90
    # For mock HSM only — symmetric key (32 bytes for AES-256)
    MOCK_HSM_KEY: str = "CHANGE_ME_IN_PRODUCTION_AES256_KEY_32B"

    # ------------------------------------------------------------------ #
    # JWT / Authentication                                                 #
    # ------------------------------------------------------------------ #
    JWT_SECRET_KEY: str = "CHANGE_ME_IN_PRODUCTION_JWT_SECRET_KEY"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 1
    INTERNAL_SERVICE_TOKEN: str = "CHANGE_ME_INTERNAL_SERVICE_TOKEN"

    # ------------------------------------------------------------------ #
    # Redis                                                                #
    # ------------------------------------------------------------------ #
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_SESSION_TTL_SECONDS: int = 300             # 5-minute auth session
    REDIS_CHALLENGE_TTL_SECONDS: int = 180           # 3-minute challenge
    REDIS_EMBEDDING_CACHE_TTL_SECONDS: int = 3600    # 1-hour embedding cache
    REDIS_ENABLED: bool = True

    # ------------------------------------------------------------------ #
    # FAISS                                                                #
    # ------------------------------------------------------------------ #
    FAISS_INDEX_PATH: str = "./data/faiss_index.bin"
    FAISS_HNSW_M: int = 32
    FAISS_HNSW_EF_CONSTRUCTION: int = 200
    FAISS_ENABLED: bool = True

    # ------------------------------------------------------------------ #
    # Kafka / Event Bus                                                    #
    # ------------------------------------------------------------------ #
    KAFKA_ENABLED: bool = False
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:9092"
    KAFKA_VOICE_AUTH_TOPIC: str = "voice-auth-events"
    KAFKA_ENROLLMENT_TOPIC: str = "voice-enrollment-events"
    KAFKA_TOPIC: str = "voice-auth-events"   # convenience alias used by KafkaEventPublisher

    # ------------------------------------------------------------------ #
    # Fraud Engine                                                         #
    # ------------------------------------------------------------------ #
    FRAUD_ENGINE_PROVIDER: Literal["mock", "nice", "fico"] = "mock"
    FRAUD_ENGINE_URL: str = "http://fraud-engine:8080"
    FRAUD_HIGH_RISK_THRESHOLD: float = 0.7
    FRAUD_REQUIRE_CLEARANCE_FOR_EMBEDDING_UPDATE: bool = True

    # ------------------------------------------------------------------ #
    # Consent / DPDP                                                       #
    # ------------------------------------------------------------------ #
    CONSENT_TOKEN_TTL_SECONDS: int = 3600            # 1-hour consent token
    CONSENT_DELETION_GRACE_DAYS: int = 7             # Days to complete deletion
    CONSENT_REQUIRED_FOR_ENROLLMENT: bool = True

    # ------------------------------------------------------------------ #
    # Emergency Access                                                     #
    # ------------------------------------------------------------------ #
    EMERGENCY_ACCESS_TTL_HOURS: int = 72
    EMERGENCY_MAX_CONTACTS: int = 2
    EMERGENCY_OTP_PROVIDER: Literal["mock", "sms", "ivr"] = "mock"

    # ------------------------------------------------------------------ #
    # Risk Engine (Layer 1 integration)                                   #
    # ------------------------------------------------------------------ #
    LAYER1_FRAUD_THRESHOLD: float = 0.70

    # ------------------------------------------------------------------ #
    # Rate Limiting                                                        #
    # ------------------------------------------------------------------ #
    RATE_LIMIT_ENROLLMENT_PER_HOUR: int = 3
    RATE_LIMIT_AUTH_PER_MINUTE: int = 10
    RATE_LIMIT_ADMIN_PER_MINUTE: int = 100

    # ------------------------------------------------------------------ #
    # File handling                                                        #
    # ------------------------------------------------------------------ #
    TEMP_UPLOAD_DIR: str = "./data/temp_uploads"
    MAX_UPLOAD_SIZE_MB: int = 20
    DATA_DIR: str = "./data"

    # ------------------------------------------------------------------ #
    # Logging                                                              #
    # ------------------------------------------------------------------ #
    LOG_LEVEL: str = "INFO"
    LOG_DIR: str = "./logs"
    LOG_JSON_FORMAT: bool = False                    # Enable for production

    # ------------------------------------------------------------------ #
    # Notification Service                                                 #
    # ------------------------------------------------------------------ #
    NOTIFICATION_PROVIDER: Literal["mock", "sms", "email"] = "mock"
    NOTIFICATION_SERVICE_URL: str = "http://notification-service:8080"

    # ------------------------------------------------------------------ #
    # CBS / IAM Integration                                                #
    # ------------------------------------------------------------------ #
    CBS_PROVIDER: Literal["mock", "finacle", "bancs"] = "mock"
    CBS_API_URL: str = "http://cbs:8080"
    IAM_PROVIDER: Literal["mock", "okta", "forgerock"] = "mock"
    IAM_API_URL: str = "http://iam:8080"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create runtime directories if they do not already exist."""
        dirs = [
            self.TEMP_UPLOAD_DIR,
            self.LOG_DIR,
            self.ECAPA_MODEL_SAVE_DIR,
            self.DATA_DIR,
            "./data/faiss",
            "./data/enrollment_sessions",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Return a cached Settings instance.

    Using lru_cache ensures the .env file is parsed only once and the same
    settings object is reused across the application.
    """
    s = Settings()
    s.ensure_directories()
    return s


# Convenience module-level singleton for non-FastAPI contexts.
settings = get_settings()
