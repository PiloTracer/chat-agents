# app/rag.py
from __future__ import annotations
from typing import Iterable, Tuple, List
from sqlalchemy import text
from sqlalchemy.orm import Session
import re

from app.embeddings import embed_texts

# ---------- chunker ----------
def chunk_text(s: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
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
    refs: List[str] = []
    texts: List[str] = []
    for src, txt in pairs:
        for p in chunker(txt):
            refs.append(src)
            texts.append(p)

    if not texts:
        return doc_id, 0

    # embed
    vecs = await embed_texts(texts)

    # insert chunks (explicit CAST for pgvector)
    for ref, body, emb in zip(refs, texts, vecs):
        db.execute(
            text(
                "INSERT INTO chunks (agent_slug, document_id, source_ref, text, embedding) "
                "VALUES (:a, :d, :r, :t, CAST(:e AS vector))"
            ),
            {"a": agent_slug, "d": doc_id, "r": ref[:255], "t": body, "e": emb},
        )
    db.commit()
    return doc_id, len(texts)

# ---------- search ----------
async def search_chunks(db: Session, agent_slug: str, query: str, top_k: int = 8) -> List[Tuple[int, str, str]]:
    qvec = (await embed_texts([query]))[0]
    rows = db.execute(
        text(
            "SELECT id, text, source_ref "
            "FROM chunks "
            "WHERE agent_slug = :a "
            "ORDER BY embedding <-> CAST(:q AS vector) "
            "LIMIT :k"
        ),
        {"a": agent_slug, "q": qvec, "k": top_k},
    ).all()
    return [(r[0], r[1], r[2]) for r in rows]
