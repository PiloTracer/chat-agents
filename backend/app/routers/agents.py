# app/routers/agents.py
from __future__ import annotations
from typing import Dict, List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.security import get_accessible_agents, require_role, Role

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/")
def list_agents(
    principal: Dict[str, str] = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
) -> List[Dict[str, str]]:
    agents = get_accessible_agents(db, principal)
    return [
        {
            "id": agent.id,
            "slug": agent.slug,
            "title": agent.title,
            "description": agent.description,
        }
        for agent in agents
    ]
