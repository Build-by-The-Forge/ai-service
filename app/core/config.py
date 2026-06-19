from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "ai-service"
    ENV: str = "development"

    DATABASE_URL: str

    OPENAI_API_KEY: str | None = None

    VECTOR_DB_TYPE: str = "faiss"

    LOG_LEVEL: str = "INFO"

settings = Settings()
