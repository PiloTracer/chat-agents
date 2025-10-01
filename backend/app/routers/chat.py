# app/routers/chat.py
from __future__ import annotations
from typing import Dict, List
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
import httpx

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
    principal: Dict[str, str] = Depends(require_role(Role.ADMIN)),
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
    hits = await search_chunks(db, agent_slug, question, top_k)
    context_blocks: List[tuple[str, str]] = []
    response_sources: List[str] = []
    for idx, (_, text, source_ref, ordinal, filename) in enumerate(hits, start=1):
        label_parts: List[str] = []
        if filename:
            label_parts.append(filename)
        if source_ref:
            label_parts.append(source_ref)
        if ordinal is not None:
            label_parts.append(f"chunk #{ordinal + 1}")
        label = " | ".join(label_parts) if label_parts else f"Chunk {idx}"
        context_blocks.append((label, text))
        response_sources.append(label)

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

