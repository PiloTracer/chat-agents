# app/rag.py
from __future__ import annotations
from typing import Iterable, Tuple, List
from sqlalchemy import text
from sqlalchemy.orm import Session
import re

from app.config import settings
from app.embeddings import embed_texts

# ---------- chunker ----------
def chunk_text(s: str, max_chars: int | None = None, overlap: int | None = None) -> List[str]:
    max_chars = max_chars or settings.MAX_CHUNK_CHARS
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    max_chars = max(1, max_chars)
    overlap = max(0, min(overlap, max_chars - 1))

    s = re.sub(r"\n{3,}", "\n\n", s.strip())
    parts: List[str] = []
    start = 0
    n = len(s)
    while start < n:
        end = min(n, start + max_chars)
        part = s[start:end].strip()
        if part:
            parts.append(part)
        if end == n:
            break
        start = max(0, end - overlap)
    return parts

# ---------- upsert document + chunks ----------
async def upsert_document(
    db: Session,
    agent_slug: str,
    filename: str,
    content_type: str,
    iter_sources: Iterable[Tuple[str, str]],
    chunker=chunk_text,
    doc_id: int | None = None,
) -> Tuple[int, int]:
    if doc_id is None:
        row = db.execute(
            text(
                "INSERT INTO documents (agent_slug, filename, content_type, meta) "
                "VALUES (:a, :f, :c, '{}'::jsonb) RETURNING id"
            ),
            {"a": agent_slug, "f": filename[:255], "c": content_type[:127]},
        ).first()
        doc_id = row[0]

    pairs: List[Tuple[str, str]] = list(iter_sources)
    chunks: List[Tuple[str, str]] = []
    for src, txt in pairs:
        for p in chunker(txt):
            chunks.append((src, p))

    if not chunks:
        return doc_id, 0

    texts = [body for _, body in chunks]
    # embed
    vecs = await embed_texts(texts)

    # insert chunks (explicit CAST for pgvector)
    for idx, ((ref, body), emb) in enumerate(zip(chunks, vecs)):
        db.execute(
            text(
                "INSERT INTO chunks (agent_slug, document_id, source_ref, text, embedding, ord) "
                "VALUES (:a, :d, :r, :t, CAST(:e AS vector), :o)"
            ),
            {"a": agent_slug, "d": doc_id, "r": ref[:255], "t": body, "e": emb, "o": idx},
        )

    db.commit()
    return doc_id, len(texts)

# ---------- search ----------
async def search_chunks(
    db: Session, agent_slug: str, query: str, top_k: int | None = None
) -> List[Tuple[int, str, str, int | None, str | None]]:
    desired = top_k or settings.TOP_K
    candidate_limit = max(desired, settings.MAX_CANDIDATE_CHUNKS)
    qvec = (await embed_texts([query]))[0]
    rows = db.execute(
        text(
            "SELECT c.id, c.text, c.source_ref, c.ord, d.filename "
            "FROM chunks AS c "
            "JOIN documents AS d ON d.id = c.document_id "
            "WHERE c.agent_slug = :a "
            "ORDER BY c.embedding <-> CAST(:q AS vector) "
            "LIMIT :k"
        ),
        {"a": agent_slug, "q": qvec, "k": candidate_limit},
    ).all()
    results = [(r[0], r[1], r[2], r[3], r[4]) for r in rows]
    return results[:desired]
