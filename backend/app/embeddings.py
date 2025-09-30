# app/embeddings.py
from __future__ import annotations
from typing import List
import os, asyncio
import httpx

OPENAI_BASE = os.getenv("OPENAI_BASE_URL", os.getenv("OPENAI_BASE", "https://api.openai.com/v1"))
OPENAI_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
BATCH = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
TIMEOUT = float(os.getenv("HTTP_TIMEOUT_SECONDS", "60"))

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if OPENAI_KEY:
        h["Authorization"] = f"Bearer {OPENAI_KEY}"
    return h

async def _embed_chunk(client: httpx.AsyncClient, inputs: List[str]) -> List[List[float]]:
    url = f"{OPENAI_BASE}/embeddings"
    payload = {"model": EMBED_MODEL, "input": inputs}
    r = await client.post(url, headers=_headers(), json=payload)
    r.raise_for_status()
    data = r.json()
    return [d["embedding"] for d in data["data"]]

async def embed_texts(texts: List[str]) -> List[List[float]]:
    out: List[List[float]] = []
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i+BATCH]
            delay = 1.0
            for attempt in range(5):
                try:
                    vecs = await _embed_chunk(client, batch)
                    out.extend(vecs)
                    break
                except (httpx.TimeoutException, httpx.RemoteProtocolError, httpx.HTTPStatusError):
                    if attempt == 4:
                        raise
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 10.0)
    return out
