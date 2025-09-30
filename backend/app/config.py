import os

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://raguser:ragpass@db:5432/ragdb")
    API_KEY = os.getenv("API_KEY", "")
    EMBEDDING_PROVIDER_BASE_URL = os.getenv("EMBEDDING_PROVIDER_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    CHAT_PROVIDER_BASE_URL = os.getenv("CHAT_PROVIDER_BASE_URL", "https://api.openai.com/v1")
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
    TOP_K = int(os.getenv("TOP_K", "8"))
    RERANK = os.getenv("RERANK", "false").lower() == "true"
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

settings = Settings()
