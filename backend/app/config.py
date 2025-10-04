import os


class Settings:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg://raguser:ragpass@db:5432/ragdb")
    API_KEY = os.getenv("API_KEY", "")

    EMBEDDING_PROVIDER_BASE_URL = os.getenv("EMBEDDING_PROVIDER_BASE_URL", "https://api.openai.com/v1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
    EMBEDDING_TARGET_DIM = int(os.getenv("EMBEDDING_TARGET_DIM", "3072"))
    HTTP_TIMEOUT_SECONDS = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))
    # Vertex may require longer per-request timeout than general HTTP
    VERTEX_HTTP_TIMEOUT_SECONDS = float(os.getenv("VERTEX_HTTP_TIMEOUT_SECONDS", os.getenv("HTTP_TIMEOUT_SECONDS", "120")))

    CHAT_PROVIDER_BASE_URL = os.getenv("CHAT_PROVIDER_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1")
    CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
    CHAT_TOP_P = float(os.getenv("CHAT_TOP_P", "1.0"))
    # Soft per-call output size. Overall responses are not limited when
    # auto-continue is enabled (see CHAT_AUTO_CONTINUE / CHAT_CONTINUE_MAX_CALLS).
    CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "2048"))
    # Automatically continue generation across calls when the provider
    # stops due to length/MaxTokens. Ensures complete responses.
    CHAT_AUTO_CONTINUE = os.getenv("CHAT_AUTO_CONTINUE", "true").lower() in {"1", "true", "yes"}
    CHAT_CONTINUE_MAX_CALLS = int(os.getenv("CHAT_CONTINUE_MAX_CALLS", "12"))
    CHAT_CONTINUE_PROMPT = os.getenv(
        "CHAT_CONTINUE_PROMPT",
        # Keep ASCII and concise to avoid token bloat
        "Continue exactly where you left off. Do not repeat earlier text.",
    )
    CHAT_MAX_RETRIES = int(os.getenv("CHAT_MAX_RETRIES", "5"))
    CHAT_BACKOFF_BASE = float(os.getenv("CHAT_BACKOFF_BASE", "0.6"))
    # Which chat provider to use by default when a request does not specify one.
    # Accepts values like 'openai' or 'gpt' for OpenAI, and 'deepseek' for DeepSeek.
    DEFAULT_CHAT_PROVIDER = os.getenv(
        "DEFAULT_CHAT_PROVIDER",
        os.getenv(
            "CHAT_PROVIDER",
            os.getenv("LLM_PROVIDER", "openai"),
        ),
    )
    # Provider-specific configuration (also re-exported at module level below)
    OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    DEEPSEEK_API_BASE = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
    DEEPSEEK_EMBEDDING_MODEL = os.getenv("DEEPSEEK_EMBEDDING_MODEL", "deepseek-embedder")
    ENABLE_PROVIDER_FALLBACK = os.getenv("ENABLE_PROVIDER_FALLBACK", "true").lower() in {"1","true","yes"}
    # Gemini provider
    GEMINI_API_BASE = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-pro")
    GEMINI_ENABLE_CONTEXT_CACHE = os.getenv("GEMINI_ENABLE_CONTEXT_CACHE", "true").lower() in {"1", "true", "yes"}
    GEMINI_CACHE_TTL_SECONDS = int(os.getenv("GEMINI_CACHE_TTL_SECONDS", "1800"))
    GEMINI_CACHE_MIN_CHARS = int(os.getenv("GEMINI_CACHE_MIN_CHARS", "4000"))
    # Cap for Gemini max output tokens per request
    GEMINI_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "8192"))
    # Soft limit for system/context chars when retrying after MAX_TOKENS
    GEMINI_TRUNCATE_SYSTEM_CHARS = int(os.getenv("GEMINI_TRUNCATE_SYSTEM_CHARS", "60000"))

    # xAI Grok provider
    GROK_API_BASE = os.getenv("GROK_API_BASE", "https://api.x.ai/v1")
    GROK_API_KEY = os.getenv("GROK_API_KEY", "")
    GROK_CHAT_MODEL = os.getenv("GROK_CHAT_MODEL", "grok-3")
    GROK_MAX_TOKENS = int(os.getenv("GROK_MAX_TOKENS", "16384"))
    GROK_CONTEXT_WINDOW = int(os.getenv("GROK_CONTEXT_WINDOW", "131072"))
    GROK_ENABLE_CONTEXT_CACHE = os.getenv("GROK_ENABLE_CONTEXT_CACHE", "false").lower() in {"1", "true", "yes"}
    GROK_CACHE_TTL_SECONDS = int(os.getenv("GROK_CACHE_TTL_SECONDS", "1800"))
    GROK_CACHE_MIN_CHARS = int(os.getenv("GROK_CACHE_MIN_CHARS", "4000"))
    # Vertex-specific chat cap (preferred over global chat cap)
    VERTEX_CHAT_MAX_TOKENS = int(os.getenv("VERTEX_CHAT_MAX_TOKENS", os.getenv("VERTEX_MAX_TOKENS", "8192")))

    MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", os.getenv("MAX_CHUNK_TOKENS", "1100")))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "240"))
    TOP_K = int(os.getenv("TOP_K", "16"))
    MAX_CANDIDATE_CHUNKS = int(os.getenv("MAX_CANDIDATE_CHUNKS", "96"))
    RERANK = os.getenv("RERANK", "false").lower() == "true"
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")

    AUTH_TOKEN_SECRET = os.getenv("AUTH_TOKEN_SECRET", "change-me")
    AUTH_TOKEN_TTL_SECONDS = int(os.getenv("AUTH_TOKEN_TTL_SECONDS", "86400"))
    AUTH_USERS = os.getenv(
        "AUTH_USERS",
        "username_changeme:password_changeme:role_changeme,username2_changeme:password2_changeme:role2_changeme",
    )
    AUTH_HASH_ITERATIONS = int(os.getenv("AUTH_HASH_ITERATIONS", "120000"))


def _as_positive_int(name: str, value: int) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be positive; got {value}")
    return value


settings = Settings()

# Validate numeric settings eagerly so deployment issues are caught at startup.
settings.EMBEDDING_BATCH_SIZE = _as_positive_int("EMBEDDING_BATCH_SIZE", settings.EMBEDDING_BATCH_SIZE)
settings.EMBEDDING_TARGET_DIM = _as_positive_int("EMBEDDING_TARGET_DIM", settings.EMBEDDING_TARGET_DIM)
settings.CHAT_MAX_TOKENS = _as_positive_int("CHAT_MAX_TOKENS", settings.CHAT_MAX_TOKENS)
settings.CHAT_CONTINUE_MAX_CALLS = _as_positive_int("CHAT_CONTINUE_MAX_CALLS", settings.CHAT_CONTINUE_MAX_CALLS)
settings.MAX_CHUNK_CHARS = _as_positive_int("MAX_CHUNK_CHARS", settings.MAX_CHUNK_CHARS)
settings.CHUNK_OVERLAP = max(0, settings.CHUNK_OVERLAP)
settings.TOP_K = _as_positive_int("TOP_K", settings.TOP_K)
settings.MAX_CANDIDATE_CHUNKS = _as_positive_int("MAX_CANDIDATE_CHUNKS", settings.MAX_CANDIDATE_CHUNKS)
settings.HTTP_TIMEOUT_SECONDS = max(1.0, settings.HTTP_TIMEOUT_SECONDS)
settings.VERTEX_HTTP_TIMEOUT_SECONDS = max(1.0, settings.VERTEX_HTTP_TIMEOUT_SECONDS)
settings.AUTH_TOKEN_TTL_SECONDS = _as_positive_int("AUTH_TOKEN_TTL_SECONDS", settings.AUTH_TOKEN_TTL_SECONDS)
settings.AUTH_HASH_ITERATIONS = _as_positive_int("AUTH_HASH_ITERATIONS", settings.AUTH_HASH_ITERATIONS)
settings.GROK_MAX_TOKENS = _as_positive_int("GROK_MAX_TOKENS", settings.GROK_MAX_TOKENS)
settings.GROK_CONTEXT_WINDOW = _as_positive_int("GROK_CONTEXT_WINDOW", settings.GROK_CONTEXT_WINDOW)
settings.GROK_CACHE_TTL_SECONDS = _as_positive_int("GROK_CACHE_TTL_SECONDS", settings.GROK_CACHE_TTL_SECONDS)
settings.GROK_CACHE_MIN_CHARS = _as_positive_int("GROK_CACHE_MIN_CHARS", settings.GROK_CACHE_MIN_CHARS)
settings.VERTEX_CHAT_MAX_TOKENS = _as_positive_int("VERTEX_CHAT_MAX_TOKENS", settings.VERTEX_CHAT_MAX_TOKENS)

if settings.CHUNK_OVERLAP >= settings.MAX_CHUNK_CHARS:
    settings.CHUNK_OVERLAP = max(0, settings.MAX_CHUNK_CHARS // 4)

if settings.TOP_K > settings.MAX_CANDIDATE_CHUNKS:
    settings.MAX_CANDIDATE_CHUNKS = settings.TOP_K

# Normalize API keys (trim whitespace/newlines) to avoid header errors
try:
    if isinstance(settings.OPENAI_API_KEY, str):
        settings.OPENAI_API_KEY = settings.OPENAI_API_KEY.strip()
    if isinstance(settings.DEEPSEEK_API_KEY, str):
        settings.DEEPSEEK_API_KEY = settings.DEEPSEEK_API_KEY.strip()
    if isinstance(settings.GEMINI_API_KEY, str):
        settings.GEMINI_API_KEY = settings.GEMINI_API_KEY.strip()
    if isinstance(settings.GROK_API_KEY, str):
        settings.GROK_API_KEY = settings.GROK_API_KEY.strip()
except Exception:
    pass

# Re-export provider-specific settings for backward compatibility
OPENAI_BASE_URL = settings.OPENAI_BASE_URL
OPENAI_API_KEY = settings.OPENAI_API_KEY
DEEPSEEK_API_BASE = settings.DEEPSEEK_API_BASE
DEEPSEEK_API_KEY = settings.DEEPSEEK_API_KEY
DEEPSEEK_CHAT_MODEL = settings.DEEPSEEK_CHAT_MODEL
DEEPSEEK_EMBEDDING_MODEL = settings.DEEPSEEK_EMBEDDING_MODEL
ENABLE_PROVIDER_FALLBACK = settings.ENABLE_PROVIDER_FALLBACK
GEMINI_API_BASE = settings.GEMINI_API_BASE
GEMINI_API_KEY = settings.GEMINI_API_KEY
GEMINI_CHAT_MODEL = settings.GEMINI_CHAT_MODEL
GEMINI_ENABLE_CONTEXT_CACHE = settings.GEMINI_ENABLE_CONTEXT_CACHE
GEMINI_CACHE_TTL_SECONDS = settings.GEMINI_CACHE_TTL_SECONDS
GEMINI_CACHE_MIN_CHARS = settings.GEMINI_CACHE_MIN_CHARS
GEMINI_MAX_TOKENS = settings.GEMINI_MAX_TOKENS

# xAI Grok re-exports
GROK_API_BASE = settings.GROK_API_BASE
GROK_API_KEY = settings.GROK_API_KEY
GROK_CHAT_MODEL = settings.GROK_CHAT_MODEL
GROK_MAX_TOKENS = settings.GROK_MAX_TOKENS
GROK_CONTEXT_WINDOW = settings.GROK_CONTEXT_WINDOW
GROK_ENABLE_CONTEXT_CACHE = settings.GROK_ENABLE_CONTEXT_CACHE
GROK_CACHE_TTL_SECONDS = settings.GROK_CACHE_TTL_SECONDS
GROK_CACHE_MIN_CHARS = settings.GROK_CACHE_MIN_CHARS

# Vertex AI (OAuth-based) configuration
class _VertexCfg:
    pass

# Prefer explicit Vertex-prefixed environment variables for clarity
settings.VERTEX_PROJECT_ID = os.getenv("VERTEX_PROJECT_ID", os.getenv("GCP_PROJECT_ID", ""))
settings.VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", os.getenv("GCP_LOCATION", "us-east1"))
settings.VERTEX_PUBLISHER = os.getenv("VERTEX_PUBLISHER", "google")
settings.VERTEX_CHAT_MODEL = os.getenv("VERTEX_CHAT_MODEL", "gemini-1.5-pro-002")
settings.VERTEX_ENABLE_CONTEXT_CACHE = os.getenv("VERTEX_ENABLE_CONTEXT_CACHE", "true").lower() in {"1", "true", "yes"}
settings.VERTEX_CACHE_TTL_SECONDS = int(os.getenv("VERTEX_CACHE_TTL_SECONDS", "1800"))
settings.VERTEX_CACHE_MIN_CHARS = int(os.getenv("VERTEX_CACHE_MIN_CHARS", "4000"))
settings.VERTEX_MAX_TOKENS = int(os.getenv("VERTEX_MAX_TOKENS", "8192"))
settings.VERTEX_TRUNCATE_SYSTEM_CHARS = int(os.getenv("VERTEX_TRUNCATE_SYSTEM_CHARS", "60000"))
settings.VERTEX_USE_OAUTH = os.getenv("VERTEX_USE_OAUTH", "true").lower() in {"1", "true", "yes"}
settings.VERTEX_ENABLE_COUNT_TOKENS = os.getenv("VERTEX_ENABLE_COUNT_TOKENS", "false").lower() in {"1", "true", "yes"}
GEMINI_TRUNCATE_SYSTEM_CHARS = settings.GEMINI_TRUNCATE_SYSTEM_CHARS
