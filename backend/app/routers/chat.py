# app/routers/chat.py
from __future__ import annotations
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import httpx

from app.auth import Principal
from app.config import settings
from app.db import get_db
from app.rag import search_chunks
from app.agents import route_question, build_system_prompt
from app.security import (
    ensure_agent_access,
    get_accessible_agents,
    require_role,
    Role,
)

router = APIRouter()


class AskPayload(BaseModel):
    question: str
    agent: str | None = None
    top_k: int = Field(default=settings.TOP_K, ge=1, le=settings.MAX_CANDIDATE_CHUNKS)


@router.post("/ask")
async def ask(
    payload: AskPayload,
    principal: Principal = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
):
    question = payload.question.strip()
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

    top_k = max(1, min(payload.top_k, settings.MAX_CANDIDATE_CHUNKS))
    # Each hit: (chunk_id, document_id, text, source_ref, ord, filename)
    hits = await search_chunks(db, agent_slug, question, top_k)

    # Group hits by document to maintain logical sequence within each file.
    groups: dict[int, list[tuple[int, int, str, str, int, str]]] = {}
    doc_order: list[int] = []
    for row in hits:
        doc_id = row[1]
        if doc_id not in groups:
            groups[doc_id] = []
            doc_order.append(doc_id)
        groups[doc_id].append(row)

    # Build sequential context blocks by sorting each group by ord asc
    # and merging consecutive ords into a single block for continuity.
    context_blocks: List[tuple[str, str]] = []
    response_sources: List[str] = []

    for doc_id in doc_order:
        rows = sorted(groups[doc_id], key=lambda r: int(r[4] or 0))
        if not rows:
            continue
        # Merge consecutive ord runs
        run_start = 0
        while run_start < len(rows):
            run_end = run_start
            while run_end + 1 < len(rows) and int(rows[run_end + 1][4] or 0) == int(rows[run_end][4] or 0) + 1:
                run_end += 1

            # Prepare label and combined text
            first = rows[run_start]
            last = rows[run_end]
            filename = first[5] or ""
            source_ref_first = first[3] or ""
            ord_first = int(first[4] or 0)
            ord_last = int(last[4] or 0)

            if run_start == run_end:
                label_parts: List[str] = []
                if filename:
                    label_parts.append(filename)
                if source_ref_first:
                    label_parts.append(source_ref_first)
                label_parts.append(f"chunk #{ord_first + 1}")
                label = " | ".join(label_parts)
                text = first[2]
            else:
                label_base: List[str] = []
                if filename:
                    label_base.append(filename)
                # Use first and last refs if available to hint range
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
            run_start = run_end + 1

    system_prompt = build_system_prompt(agent_slug, context_blocks, agent_map)

    headers = {"Content-Type": "application/json"}
    if settings.API_KEY:
        headers["Authorization"] = f"Bearer {settings.API_KEY}"

    completion_payload = {
        "model": settings.CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ],
        "temperature": settings.CHAT_TEMPERATURE,
        "top_p": settings.CHAT_TOP_P,
        "max_tokens": settings.CHAT_MAX_TOKENS,
    }

    async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{settings.CHAT_PROVIDER_BASE_URL}/chat/completions",
            headers=headers,
            json=completion_payload,
        )
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]

    return {
        "ok": True,
        "agent": agent_slug,
        "answer": answer,
        "sources": response_sources,
    }
