# app/embeddings.py
from __future__ import annotations
from typing import List
import asyncio

import httpx

from app.config import settings

EMBED_BASE_URL = settings.EMBEDDING_PROVIDER_BASE_URL.rstrip('/')
EMBED_MODEL = settings.EMBEDDING_MODEL
EMBED_BATCH_SIZE = settings.EMBEDDING_BATCH_SIZE
REQUEST_TIMEOUT = settings.HTTP_TIMEOUT_SECONDS
API_KEY = settings.API_KEY

def _headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    return headers

async def _embed_chunk(client: httpx.AsyncClient, inputs: List[str]) -> List[List[float]]:
    url = f"{EMBED_BASE_URL}/embeddings"
    payload = {"model": EMBED_MODEL, "input": inputs}
    response = await client.post(url, headers=_headers(), json=payload)
    response.raise_for_status()
    data = response.json()
    return [item["embedding"] for item in data["data"]]

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
                except (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.HTTPStatusError):
                    if attempt == 4:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 10.0)
    return embeddings
