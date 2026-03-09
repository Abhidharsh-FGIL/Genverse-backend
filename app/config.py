from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    APP_NAME: str = "Genverse.ai"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database — individual fields; URLs are built dynamically
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "eduverse_db"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""

    @property
    def database_url(self) -> str:
        """Async URL for SQLAlchemy (asyncpg driver)."""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @property
    def sync_database_url(self) -> str:
        """Sync URL for Alembic migrations (psycopg2 driver)."""
        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    # JWT
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30

    # Google OAuth
    GOOGLE_CLIENT_ID: str = "314078026446-sovdc3nrc2sg6rbv1mns1c2f2a08r3tp.apps.googleusercontent.com"

    # AI Providers
    GOOGLE_GEMINI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    AI_PRIMARY_MODEL: str = "gemini-2.5-flash"
    AI_FALLBACK_MODEL: str = "gpt-4o-mini"
    AI_DOCUMENT_MODEL: str = "claude-sonnet-4-20250514"

    # YouTube Data API
    YOUTUBE_API_KEY: str = ""

    # RapidAPI — Google News (google-news13.p.rapidapi.com)
    RAPIDAPI_KEY: str = ""
    RAPIDAPI_NEWS_HOST: str = "google-news13.p.rapidapi.com"

    # Storage
    STORAGE_ROOT: str = "./uploads"
    MAX_UPLOAD_SIZE_MB: int = 50

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Email
    MAIL_USERNAME: str = ""
    MAIL_PASSWORD: str = ""
    MAIL_FROM: str = "noreply@genverse.ai"
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.gmail.com"
    MAIL_FROM_NAME: str = "Genverse.ai"
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False

    # Payment Gateways — PhonePe Standard Checkout v2
    PHONEPE_CLIENT_ID: str = ""
    PHONEPE_CLIENT_SECRET: str = ""
    PHONEPE_CLIENT_VERSION: int = 1
    PHONEPE_ENV: str = "SANDBOX"  # SANDBOX or PRODUCTION

    # Stripe (international cards)
    STRIPE_API_KEY: str = ""
    STRIPE_ENDPOINT_SECRET: str = ""

    @property
    def phonepe_base_url(self) -> str:
        if self.PHONEPE_ENV == "PRODUCTION":
            return "https://api.phonepe.com/apis/pg"
        return "https://api-preprod.phonepe.com/apis/pg-sandbox"

    # Frontend / Backend URLs
    FRONTEND_URL: str = "http://localhost:8080"
    BACKEND_URL: str = "http://localhost:8000"

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Content Safety
    CONTENT_GUARD_ENABLED: bool = True
    CONTENT_GUARD_LLM_CLASSIFIER: bool = True

    # Feature flags
    ENABLE_EMAIL_NOTIFICATIONS: bool = False
    ENABLE_BACKGROUND_TASKS: bool = False

    @property
    def cors_origins_list(self) -> List[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024


settings = Settings()
