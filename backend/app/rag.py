# app/rag.py
from __future__ import annotations
from typing import Iterable, Tuple, List
from sqlalchemy import text
from sqlalchemy.orm import Session
import re

from app.config import settings
from app.embeddings import embed_texts

# ---------- advanced chunker ----------
# Sentence splitter that also treats an ellipsis (\u2026) as potential boundary.
_SENT_SPLIT_RE = re.compile(r"(?<=[" + "\u2026" + r".!?])\s+(?=[\(\[\"'A-Z0-9])")
_BLANKLINE_RE = re.compile(r"\n\s*\n+")


def _split_paragraphs(s: str) -> List[str]:
    s = s.replace("\r\n", "\n").replace("\r", "\n").strip()
    parts = [p.strip() for p in _BLANKLINE_RE.split(s) if p.strip()]
    return parts if parts else ([s] if s else [])


def _split_sentences(p: str) -> List[str]:
    pieces = [seg.strip() for seg in _SENT_SPLIT_RE.split(p) if seg.strip()]
    return pieces if pieces else ([p] if p else [])


def _take_tail_overlap(text: str, overlap_chars: int) -> str:
    if overlap_chars <= 0 or not text:
        return ""
    window = text[-overlap_chars:]
    m = re.search(r"[" + "\u2026" + r".!?]\s+", window)
    if m:
        return window[m.end():]
    nl = window.rfind("\n")
    if nl != -1 and nl + 1 < len(window):
        return window[nl + 1 :]
    return window


def chunk_text(s: str, max_chars: int | None = None, overlap: int | None = None) -> List[str]:
    """Greedy, structure-aware chunking.

    - Respects paragraphs and sentence boundaries when possible.
    - Uses character-based sizing for simplicity and speed.
    - Adds textual overlap between chunks for context continuity.
    """
    max_chars = max(1, (max_chars or settings.MAX_CHUNK_CHARS))
    overlap = settings.CHUNK_OVERLAP if overlap is None else overlap
    overlap = max(0, min(overlap, max_chars - 1))

    paragraphs = _split_paragraphs(s)
    pieces: List[str] = []

    carry = ""
    buf: List[str] = []
    buf_len = 0

    def flush() -> None:
        nonlocal carry, buf, buf_len
        if not buf:
            return
        joined = (carry + ("\n" if carry and buf else "") + "\n".join(buf)).strip()
        if joined:
            pieces.append(joined)
            carry = _take_tail_overlap(joined, overlap)
        else:
            carry = ""
        buf = []
        buf_len = 0

    for para in paragraphs:
        sentences = _split_sentences(para)
        for sent in sentences:
            unit = sent
            unit_len = len(unit) + (1 if buf else 0)
            if not buf and len(carry) + len(unit) > max_chars:
                start = 0
                while start < len(unit):
                    remain = max_chars - len(carry) - (1 if carry else 0)
                    chunk = (carry + ("\n" if carry else "") + unit[start:start + remain]).strip()
                    if chunk:
                        pieces.append(chunk)
                        carry = _take_tail_overlap(chunk, overlap)
                    start += remain
                continue

            if len(carry) + buf_len + unit_len <= max_chars:
                if buf:
                    buf.append(unit)
                    buf_len += unit_len
                else:
                    buf.append(unit)
                    buf_len = unit_len
            else:
                flush()
                if len(carry) + len(unit) > max_chars:
                    start = 0
                    while start < len(unit):
                        remain = max_chars - len(carry) - (1 if carry else 0)
                        chunk = (carry + ("\n" if carry else "") + unit[start:start + remain]).strip()
                        if chunk:
                            pieces.append(chunk)
                            carry = _take_tail_overlap(chunk, overlap)
                        start += remain
                else:
                    buf.append(unit)
                    buf_len = len(unit)
        # Prefer to flush at paragraph boundaries when the buffer is fairly full
        if buf_len > 0 and (len(carry) + buf_len >= max_chars * 0.85):
            flush()
        else:
            if buf:
                buf.append("")
                buf_len += 1

    flush()

    if not pieces and s.strip():
        pieces = [s.strip()[:max_chars]]
    return pieces

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
