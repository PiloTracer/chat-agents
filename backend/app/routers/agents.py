# app/routers/agents.py
from __future__ import annotations
from typing import List

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.auth import Principal
from app.db import get_db
from app.schemas import AgentOut
from app.security import get_accessible_agents, require_role, Role

router = APIRouter(prefix="/agents", tags=["agents"])


@router.get("/", response_model=List[AgentOut])
@router.get("", response_model=List[AgentOut], include_in_schema=False)
def list_agents(
    principal: Principal = Depends(require_role(Role.ADMIN)),
    db: Session = Depends(get_db),
) -> List[AgentOut]:
    agents = get_accessible_agents(db, principal)
    return [
        AgentOut(
            id=agent.id,
            slug=agent.slug,
            title=agent.title,
            description=agent.description,
        )
        for agent in agents
    ]
