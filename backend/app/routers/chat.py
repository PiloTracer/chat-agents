# app/routers/chat.py
from __future__ import annotations
import asyncio
from typing import List
import re
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import httpx  # kept for compatibility; not strictly required after refactor

from app.auth import Principal
from app.config import settings
from app.db import get_db
from app.rag import search_chunks
from app.agents import route_question, build_system_prompt
from app.llm_provider import llm_provider
from app.models import Chunk, Document
from app.security import (
    ensure_agent_access,
    get_accessible_agents,
    require_role,
    Role,
)

router = APIRouter()


class AskPayload(BaseModel):
    provider: str | None = Field(default=None, description="'gpt'|'openai' or 'deepseek' or 'gemini'")
    question: str
    agent: str | None = None
    top_k: int = Field(default=settings.TOP_K, ge=1, le=settings.MAX_CANDIDATE_CHUNKS)
    extended: bool = Field(default=False, description="Enable extended context by including adjacent pages")
    page_window: int = Field(default=1, ge=0, le=5)


def _page_of(ref: str) -> int:
    m = re.search(r"(?:^|#)(page|slide|para|sec)=(\d+)", ref or "")
    return int(m.group(2)) if m else -1


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _extract_quoted_title(q: str) -> str | None:
    # Support ASCII and common curly quotes via \u escapes to avoid non-ASCII literals
    quotes = "\"'\u201c\u201d\u2018\u2019"
    pattern = "[" + quotes + "]" + "([^" + quotes + "]{3,120})" + "[" + quotes + "]"
    m = re.search(pattern, q or "")
    if m:
        return m.group(1).strip()
    return None


def _is_completeness_intent(question: str) -> bool:
    q = _normalize(question)
    # ASCII-only terms to avoid encoding issues
    terms = [
        "esta completo",
        "completo",
        "completitud",
        "integridad",
        "continuidad",
        "faltan paginas",
        "faltan capitulos",
        "incompleto",
        "libro completo",
        "obra completa",
        "verificar continuidad",
        "verificar si esta completo",
    ]
    return any(t in q for t in terms)


def _find_candidate_documents(db: Session, question: str, doc_ids_from_hits: List[int]) -> List[Document]:
    quoted = _extract_quoted_title(question)
    if quoted:
        title_norm = _normalize(quoted)
        docs = db.query(Document).all()
        ranked: List[tuple[int, Document]] = []
        for d in docs:
            fname = _normalize(d.filename or "")
            score = 0
            if title_norm in fname:
                score += 10
            if any(tok in fname for tok in title_norm.split()):
                score += 1
            if score > 0:
                ranked.append((score, d))
        ranked.sort(key=lambda t: t[0], reverse=True)
        if ranked:
            return [d for _, d in ranked]
    if doc_ids_from_hits:
        return list(db.query(Document).filter(Document.id.in_(doc_ids_from_hits)).all())
    return []


def _coverage_for_doc(db: Session, doc: Document) -> tuple[str, str, dict]:
    rows = (
        db.query(Chunk.id, Chunk.document_id, Chunk.text, Chunk.source_ref, Chunk.ord)
        .filter(Chunk.document_id == doc.id)
        .order_by(Chunk.ord.asc())
        .all()
    )
    pages = []
    for r in rows:
        p = _page_of(r[3] or "")
        if p >= 0:
            pages.append(p)
    unique_pages = sorted(set(pages))
    # Incorporate blank pages metadata if available
    meta = getattr(doc, "meta", None)
    blanks = []
    total = None
    try:
        if isinstance(meta, dict):
            blanks = list(meta.get("blank_pages") or [])
            total = meta.get("pages_total")
    except Exception:
        blanks = []
        total = None
    present = set(unique_pages) | set(blanks)
    missing_pages = []
    continuity_ok = True
    page_range = None
    if isinstance(total, int) and total > 0:
        page_range = (1, total)
        missing_pages = [n for n in range(1, total + 1) if n not in present]
        continuity_ok = len(missing_pages) == 0
    elif unique_pages:
        first = unique_pages[0]
        last = unique_pages[-1]
        page_range = (first, last)
        have = set(unique_pages)
        missing_pages = [n for n in range(first, last + 1) if n not in have]
        continuity_ok = len(missing_pages) == 0

    lines: List[str] = []
    lines.append("Coverage Report (computed):")
    lines.append(f"- document_id: {doc.id}")
    lines.append(f"- filename: {doc.filename}")
    lines.append(f"- total_chunks: {len(rows)}")
    if page_range:
        lines.append(f"- pages_covered: {page_range[0]}..{page_range[1]}")
        lines.append(f"- unique_pages: {len(unique_pages)}")
        if missing_pages:
            lines.append(f"- missing_pages: {', '.join(str(p) for p in missing_pages)}")
        else:
            lines.append("- missing_pages: none")
    else:
        lines.append("- pages_covered: unknown (no page markers); using chunk ord only")
    lines.append(f"- continuity_ok: {'yes' if continuity_ok else 'no'}")

    label = f"Coverage: {doc.filename or 'document'}"
    text = "\n".join(lines)
    meta_out = {
        "type": "coverage",
        "document_id": int(doc.id),
        "filename": doc.filename or "",
        "total_chunks": int(len(rows)),
        "unique_pages": unique_pages,
        "page_range": page_range,
        "missing_pages": missing_pages,
        "continuity_ok": continuity_ok,
    }
    return label, text, meta_out


@router.post("/ask")
async def ask(
    payload: AskPayload,
    principal: Principal = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
):
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question")

    agents = get_accessible_agents(db, principal)
    if not agents:
        raise HTTPException(status_code=403, detail="No agents available for this user")

    agent_map = {agent.slug: agent.description or agent.title for agent in agents}

    if payload.agent:
        agent_slug = payload.agent
        ensure_agent_access(db, principal, agent_slug)
    else:
        agent_slug = await route_question(question, agent_map)

    if agent_slug not in agent_map:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Intent and citation detection
    is_completeness = _is_completeness_intent(question)
    quoted_title = _extract_quoted_title(question)
    # Auto-extend when completeness intent OR when a book is cited OR generic 'libro' mention
    extended = payload.extended or is_completeness or (quoted_title is not None) or ("libro" in question.lower())

    top_k = max(1, min(payload.top_k, settings.MAX_CANDIDATE_CHUNKS))
    # Each hit: (chunk_id, document_id, text, source_ref, ord, filename)
    hits = await search_chunks(db, agent_slug, question, top_k)

    # Optionally expand context with adjacent pages for each hit
    if extended and hits:
        doc_ids = list({h[1] for h in hits})
        rows = (
            db.query(Chunk.id, Chunk.document_id, Chunk.text, Chunk.source_ref, Chunk.ord, Document.filename)
            .join(Document, Document.id == Chunk.document_id)
            .filter(Chunk.document_id.in_(doc_ids))
            .order_by(Chunk.ord.asc())
            .all()
        )
        by_doc_page: dict[int, dict[int, list[tuple[int, int, str, str, int, str]]]] = {}
        for r in rows:
            doc_id = int(r[1])
            page_no = _page_of(r[3] or "")
            d = by_doc_page.setdefault(doc_id, {})
            d.setdefault(page_no, []).append(
                (int(r[0]), int(r[1]), str(r[2]), str(r[3]) if r[3] else "", int(r[4] or 0), str(r[5]) if r[5] else "")
            )

        selected: dict[int, tuple[int, int, str, str, int, str]] = {h[0]: h for h in hits}
        for h in hits:
            doc_id = h[1]
            page_no = _page_of(h[3] or "")
            if doc_id not in by_doc_page:
                continue
            for p in range(page_no - payload.page_window, page_no + payload.page_window + 1):
                for row in by_doc_page.get(doc_id, {}).get(p, []):
                    selected[row[0]] = row
        hits = list(selected.values())

    # If there is a quoted title, prepend its coverage and boost its content
    coverage_blocks: List[tuple[str, str]] = []
    sources_meta: List[dict] = []
    candidate_docs = _find_candidate_documents(db, question, [h[1] for h in hits])
    priority_doc_ids = {d.id for d in candidate_docs}
    for d in candidate_docs:
        cov_label, cov_text, cov_meta = _coverage_for_doc(db, d)
        # Canonical fact when coverage is complete
        if cov_meta.get("continuity_ok") and not cov_meta.get("missing_pages"):
            fact_label = f"Hecho verificado: {d.filename}"
            fact_text = (
                f"El documento '{d.filename}' esta completo (missing_pages=none). "
                "No afirmes que falta contenido salvo que haya evidencia contraria."
            )
            coverage_blocks.append((fact_label, fact_text))
            sources_meta.append({"type": "fact", "document_id": int(d.id), "filename": d.filename or ""})
        coverage_blocks.append((cov_label, cov_text))
        sources_meta.append(cov_meta)
        if extended and d.id not in {h[1] for h in hits}:
            extra_rows = (
                db.query(Chunk.id, Chunk.document_id, Chunk.text, Chunk.source_ref, Chunk.ord, Document.filename)
                .join(Document, Document.id == Chunk.document_id)
                .filter(Chunk.document_id == d.id)
                .order_by(Chunk.ord.asc())
                .limit(max(12, top_k))
                .all()
            )
            for r in extra_rows:
                hits.append(
                    (
                        int(r[0]),
                        int(r[1]),
                        str(r[2]),
                        str(r[3]) if r[3] else "",
                        int(r[4] or 0),
                        str(r[5]) if r[5] else "",
                    )
                )

    # Group hits by document to maintain logical sequence within each file.
    groups: dict[int, list[tuple[int, int, str, str, int, str]]] = {}
    doc_order: list[int] = []
    for row in hits:
        doc_id = row[1]
        if doc_id not in groups:
            groups[doc_id] = []
            doc_order.append(doc_id)
        groups[doc_id].append(row)

    # Prioritize cited documents first in doc_order
    if priority_doc_ids:
        doc_order = sorted(doc_order, key=lambda did: 0 if did in priority_doc_ids else 1)

    # Build sequential context blocks by sorting each group by page asc then ord, and merge consecutive ords
    context_blocks: List[tuple[str, str]] = []
    response_sources: List[str] = []
    if coverage_blocks:
        context_blocks.extend(coverage_blocks)
        response_sources.extend([label for (label, _) in coverage_blocks])

    for doc_id in doc_order:
        rows = sorted(
            groups[doc_id],
            key=lambda r: (_page_of(r[3] or ""), int(r[4] or 0)),
        )
        if not rows:
            continue
        run_start = 0
        while run_start < len(rows):
            run_end = run_start
            while run_end + 1 < len(rows) and int(rows[run_end + 1][4] or 0) == int(rows[run_end][4] or 0) + 1:
                run_end += 1

            first = rows[run_start]
            last = rows[run_end]
            filename = first[5] or ""
            source_ref_first = first[3] or ""
            ord_first = int(first[4] or 0)
            ord_last = int(last[4] or 0)
            page_first = _page_of(source_ref_first)
            page_last = _page_of(last[3] or "")

            if run_start == run_end:
                label_parts: List[str] = []
                if filename:
                    label_parts.append(filename)
                if page_first >= 0:
                    label_parts.append(f"p. {page_first}")
                elif source_ref_first:
                    label_parts.append(source_ref_first)
                label_parts.append(f"chunk #{ord_first + 1}")
                label = " | ".join(label_parts)
                text = first[2]
            else:
                label_base: List[str] = []
                if filename:
                    label_base.append(filename)
                if page_first >= 0 and page_last >= 0:
                    if page_first == page_last:
                        label_base.append(f"p. {page_first}")
                    else:
                        label_base.append(f"p. {page_first}-{page_last}")
                else:
                    source_ref_last = last[3] or ""
                    if source_ref_first and source_ref_last and source_ref_first != source_ref_last:
                        label_base.append(f"{source_ref_first} -> {source_ref_last}")
                    elif source_ref_first:
                        label_base.append(source_ref_first)
                label_base.append(f"chunks #{ord_first + 1}-{ord_last + 1}")
                label = " | ".join(label_base)
                text = "\n\n".join(r[2] for r in rows[run_start : run_end + 1])

            context_blocks.append((label, text))
            response_sources.append(label)
            sources_meta.append({
                "type": "context",
                "document_id": int(doc_id),
                "filename": filename,
                "page_start": int(page_first) if page_first >= 0 else None,
                "page_end": int(page_last) if page_last >= 0 else None,
                "ord_start": ord_first,
                "ord_end": ord_last,
                "chunk_ids": [int(r[0]) for r in rows[run_start : run_end + 1]],
                "label": label,
            })
            run_start = run_end + 1

    system_prompt = build_system_prompt(agent_slug, context_blocks, agent_map)

    # ---------- LLM call via unified provider client ----------
    # Ensure a defined value even if client/fallback doesn't populate it
    used_provider = payload.provider or "openai"

    # Build messages for OpenAI/DeepSeek compatible chat
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    try:
        result = await llm_provider.chat_completion(
            messages=messages,
            provider=payload.provider,                 # "gpt"/"openai" or "deepseek"
            temperature=settings.CHAT_TEMPERATURE,
            max_tokens=settings.CHAT_MAX_TOKENS,
            retries=getattr(settings, "CHAT_MAX_RETRIES", 2),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    answer = (result.get("content") or "").strip()
    used_provider = result.get("provider", used_provider)
    used_model = result.get("model", settings.CHAT_MODEL)

    if not answer:
        raise HTTPException(status_code=502, detail="No answer from chat provider")

    return {
        "ok": True,
        "provider": used_provider,
        "model": used_model,
        "agent": agent_slug,
        "answer": answer,
        "sources": response_sources,
        "sources_meta": sources_meta,
    }
