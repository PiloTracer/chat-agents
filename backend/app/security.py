from __future__ import annotations
from typing import Dict, List

from fastapi import Header, HTTPException
from sqlalchemy.orm import Session

from app.models import Agent, AgentACL


class Role:
    OWNER = "owner"
    ADMIN = "admin"


_ROLE_RANK = {Role.ADMIN: 1, Role.OWNER: 2}


def require_role(required: str):
    min_rank = _ROLE_RANK.get(required, 0)

    async def dep(
        x_role: str = Header(default=Role.ADMIN, alias="X-Role"),
        x_user: str = Header(default="demo", alias="X-User"),
    ) -> Dict[str, str]:
        role = (x_role or Role.ADMIN).lower()
        user = x_user or "demo"
        if role not in _ROLE_RANK:
            raise HTTPException(status_code=403, detail="Invalid role")
        if _ROLE_RANK[role] < min_rank:
            raise HTTPException(status_code=403, detail="Insufficient role")
        return {"role": role, "user": user}

    return dep


def ensure_agent_access(db: Session, principal: Dict[str, str], agent_slug: str) -> Agent:
    agent = db.query(Agent).filter(Agent.slug == agent_slug).first()
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")
    if principal.get("role") == Role.OWNER:
        return agent
    allowed = (
        db.query(AgentACL)
        .filter(AgentACL.username == principal.get("user"), AgentACL.agent_slug == agent_slug)
        .first()
    )
    if allowed is None:
        raise HTTPException(status_code=403, detail="Agent not permitted")
    return agent


def get_accessible_agents(db: Session, principal: Dict[str, str]) -> List[Agent]:
    query = db.query(Agent)
    if principal.get("role") == Role.OWNER:
        return query.order_by(Agent.slug).all()
    return (
        query.join(AgentACL, Agent.slug == AgentACL.agent_slug)
        .filter(AgentACL.username == principal.get("user"))
        .order_by(Agent.slug)
        .distinct()
        .all()
    )
