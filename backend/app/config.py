import os

class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://raguser:ragpass@db:5432/ragdb")
    API_KEY = os.getenv("API_KEY", "")

    EMBEDDING_PROVIDER_BASE_URL = os.getenv("EMBEDDING_PROVIDER_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    EMBEDDING_TARGET_DIM = int(os.getenv("EMBEDDING_TARGET_DIM", "3072"))
    HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))

    CHAT_PROVIDER_BASE_URL = os.getenv("CHAT_PROVIDER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1")
    CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
    CHAT_TOP_P = float(os.getenv("CHAT_TOP_P", "1.0"))
    CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "2048"))

    MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", os.getenv("MAX_CHUNK_TOKENS", "900")))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    TOP_K = int(os.getenv("TOP_K", "16"))
    MAX_CANDIDATE_CHUNKS = int(os.getenv("MAX_CANDIDATE_CHUNKS", "24"))
    RERANK = os.getenv("RERANK", "false").lower() == "true"
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")


def _as_positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


settings = Settings()

# Validate numeric settings eagerly so deployment issues are caught at startup.
settings.EMBEDDING_BATCH_SIZE = _as_positive_int("EMBEDDING_BATCH_SIZE", settings.EMBEDDING_BATCH_SIZE)
settings.EMBEDDING_TARGET_DIM = _as_positive_int("EMBEDDING_TARGET_DIM", settings.EMBEDDING_TARGET_DIM)
settings.CHAT_MAX_TOKENS = _as_positive_int("CHAT_MAX_TOKENS", settings.CHAT_MAX_TOKENS)
settings.MAX_CHUNK_CHARS = _as_positive_int("MAX_CHUNK_CHARS", settings.MAX_CHUNK_CHARS)
settings.CHUNK_OVERLAP = max(0, settings.CHUNK_OVERLAP)
settings.TOP_K = _as_positive_int("TOP_K", settings.TOP_K)
settings.MAX_CANDIDATE_CHUNKS = _as_positive_int("MAX_CANDIDATE_CHUNKS", settings.MAX_CANDIDATE_CHUNKS)
settings.HTTP_TIMEOUT_SECONDS = max(1.0, settings.HTTP_TIMEOUT_SECONDS)

if settings.CHUNK_OVERLAP >= settings.MAX_CHUNK_CHARS:
    settings.CHUNK_OVERLAP = max(0, settings.MAX_CHUNK_CHARS // 4)

if settings.TOP_K > settings.MAX_CANDIDATE_CHUNKS:
    settings.MAX_CANDIDATE_CHUNKS = settings.TOP_K
