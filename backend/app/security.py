from __future__ import annotations

from typing import List

from fastapi import Header, HTTPException, status
from sqlalchemy.orm import Session

from app.auth import Principal, get_auth_service
from app.models import Agent, AgentACL


class Role:
    OWNER = "owner"
    ADMIN = "admin"


_ROLE_RANK = {Role.ADMIN: 1, Role.OWNER: 2}


def require_role(required: str):
    min_rank = _ROLE_RANK.get(required, 0)

    async def dep(
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> Principal:
        service = get_auth_service()
        principal = service.verify_bearer(authorization)
        if not principal:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")

        role = principal.role.lower()
        if role not in _ROLE_RANK:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid role")
        if _ROLE_RANK[role] < min_rank:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return Principal(username=principal.username, role=role)

    return dep


def ensure_agent_access(db: Session, principal: Principal, agent_slug: str) -> Agent:
    agent = db.query(Agent).filter(Agent.slug == agent_slug).first()
    if agent is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")
    if principal.role == Role.OWNER:
        return agent
    allowed = (
        db.query(AgentACL)
        .filter(AgentACL.username == principal.username, AgentACL.agent_slug == agent_slug)
        .first()
    )
    if allowed is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Agent not permitted")
    return agent


def get_accessible_agents(db: Session, principal: Principal) -> List[Agent]:
    query = db.query(Agent)
    if principal.role == Role.OWNER:
        return query.order_by(Agent.slug).all()
    return (
        query.join(AgentACL, Agent.slug == AgentACL.agent_slug)
        .filter(AgentACL.username == principal.username)
        .order_by(Agent.slug)
        .distinct()
        .all()
    )
