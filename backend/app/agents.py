# app/agents.py
from __future__ import annotations
from typing import Dict, List, Tuple

import httpx

from app.config import settings
from app.defaults import DEFAULT_AGENT_DEFINITIONS


DEFAULT_AGENT_DESCRIPTIONS: Dict[str, str] = {
    agent["slug"]: agent["description"] for agent in DEFAULT_AGENT_DEFINITIONS
}


def _router_system_prompt(agent_map: Dict[str, str]) -> str:
    lines = [
        "You are a router. Choose the single best agent slug for the user's question.",
        "Select only from the following options:",
    ]
    for slug, description in agent_map.items():
        lines.append(f"- {slug}: {description}")
    lines.append("Reply with ONLY the slug.")
    return "\n".join(lines)


async def route_question(question: str, agents: Dict[str, str]) -> str:
    if not agents:
        raise ValueError("No agents available for routing.")

    headers = {"Content-Type": "application/json"}
    if settings.API_KEY:
        headers["Authorization"] = f"Bearer {settings.API_KEY}"

    payload = {
        "model": settings.CHAT_MODEL,
        "messages": [
            {"role": "system", "content": _router_system_prompt(agents)},
            {"role": "user", "content": question},
        ],
        "temperature": 0,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.CHAT_PROVIDER_BASE_URL}/chat/completions",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        slug = response.json()["choices"][0]["message"]["content"].strip()

    return slug if slug in agents else next(iter(agents))


def build_system_prompt(
    agent_slug: str,
    context_blocks: List[Tuple[str, str]],
    agent_descriptions: Dict[str, str],
) -> str:
    role = agent_descriptions.get(agent_slug, DEFAULT_AGENT_DESCRIPTIONS.get(agent_slug, "Expert assistant"))
    if context_blocks:
        context = "\n\n".join(
            f"[{idx}] {source}\n{text}" for idx, (source, text) in enumerate(context_blocks, start=1)
        )
    else:
        context = "No supporting context was retrieved."

    return (
        f"{role}\n"
        "Answer using ONLY the provided context and cite sources as [#].\n"
        "Context:\n"
        f"{context}\n"
        "If unsure, say you don't know."
    )
