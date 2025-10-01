# app/embeddings.py
from __future__ import annotations
from typing import List
import asyncio
import logging

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

EMBED_BASE_URL = settings.EMBEDDING_PROVIDER_BASE_URL.rstrip('/')
EMBED_BATCH_SIZE = settings.EMBEDDING_BATCH_SIZE
REQUEST_TIMEOUT = settings.HTTP_TIMEOUT_SECONDS
TARGET_DIM = settings.EMBEDDING_TARGET_DIM
API_KEY = settings.API_KEY

MODEL_DIMENSIONS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


def _candidate_models() -> List[str]:
    preferred = [
        settings.EMBEDDING_MODEL,
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-ada-002",
    ]
    models: List[str] = []
    for name in preferred:
        if not name or name in models:
            continue
        dim = MODEL_DIMENSIONS.get(name)
        if dim is not None and dim != TARGET_DIM:
            logger.warning(
                "Skipping embedding model %s because dimension %s does not match target %s",
                name,
                dim,
                TARGET_DIM,
            )
            continue
        models.append(name)
    if not models:
        default = "text-embedding-3-small" if TARGET_DIM == 1536 else "text-embedding-3-large"
        logger.warning("No embedding models matched target dim %s; defaulting to %s", TARGET_DIM, default)
        models.append(default)
    return models


_MODEL_QUEUE = _candidate_models()
_CURRENT_MODEL = _MODEL_QUEUE[0]
_FALLBACK_MODELS = _MODEL_QUEUE[1:]


def _switch_to_next_model() -> bool:
    global _CURRENT_MODEL, _FALLBACK_MODELS
    while _FALLBACK_MODELS:
        candidate = _FALLBACK_MODELS.pop(0)
        if candidate:
            logger.warning(
                "Embedding model %s unavailable or incompatible; falling back to %s",
                _CURRENT_MODEL,
                candidate,
            )
            _CURRENT_MODEL = candidate
            return True
    logger.error("No further embedding model fallbacks are available")
    return False


def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if not API_KEY:
        raise RuntimeError("OPENAI_API_KEY or API_KEY environment variable is required for embeddings")
    headers["Authorization"] = f"Bearer {API_KEY}"
    return headers


async def _embed_chunk(client: httpx.AsyncClient, inputs: List[str]) -> List[List[float]]:
    url = f"{EMBED_BASE_URL}/embeddings"
    payload = {"model": _CURRENT_MODEL, "input": inputs}
    try:
        response = await client.post(url, headers=_headers(), json=payload)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail = ""
        try:
            data = exc.response.json()
            detail = data.get("error", {}).get("message") or data.get("message") or str(data)
        except Exception:
            detail = exc.response.text
        raise RuntimeError(
            f"Embedding request failed ({exc.response.status_code}) for model '{_CURRENT_MODEL}': {detail}"
        ) from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"Embedding request failed: {exc}") from exc

    data = response.json()
    if "data" not in data:
        raise RuntimeError(f"Embedding response missing data field: {data}")

    embeddings: List[List[float]] = []
    for item in data["data"]:
        embedding = item.get("embedding")
        if embedding is None:
            raise RuntimeError(f"Embedding item missing vector for model '{_CURRENT_MODEL}': {item}")
        embeddings.append(embedding)

    if embeddings:
        actual_dim = len(embeddings[0])
        if actual_dim != TARGET_DIM:
            raise RuntimeError(
                f"Embedding dimension mismatch for model '{_CURRENT_MODEL}': expected {TARGET_DIM}, got {actual_dim}"
            )
    return embeddings


async def embed_texts(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []

    embeddings: List[List[float]] = []
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            delay = 1.0
            for attempt in range(5):
                try:
                    vectors = await _embed_chunk(client, batch)
                    embeddings.extend(vectors)
                    break
                except RuntimeError as exc:
                    message = str(exc).lower()
                    if ("invalid model" in message or "dimension mismatch" in message) and _switch_to_next_model():
                        logger.info("Retrying embeddings with fallback model '%s'", _CURRENT_MODEL)
                        delay = 1.0
                        continue
                    if attempt == 4:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 10.0)
                except (httpx.TimeoutException, httpx.RemoteProtocolError):
                    if attempt == 4:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 10.0)
    return embeddings
