from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.auth import Principal, get_auth_service
from app.db import get_db
from app.security import get_accessible_agents

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str
    password: str


class PrincipalOut(BaseModel):
    username: str
    role: str

    @classmethod
    def from_principal(cls, principal: Principal) -> "PrincipalOut":
        return cls(username=principal.username, role=principal.role)


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user: PrincipalOut


class MeResponse(BaseModel):
    user: PrincipalOut
    agents: list[str]


def _extract_principal(authorization: str | None) -> Principal:
    service = get_auth_service()
    principal = service.verify_bearer(authorization)
    if not principal:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return principal


@router.post("/login", response_model=LoginResponse)
def login(payload: LoginRequest) -> LoginResponse:
    service = get_auth_service()
    result = service.login(payload.username, payload.password)
    if not result:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token, principal = result
    return LoginResponse(access_token=token, token_type="bearer", user=PrincipalOut.from_principal(principal))


@router.post("/logout")
def logout() -> dict[str, bool]:
    # Stateless tokens cannot be revoked server-side without additional storage.
    # Clients should discard the token on logout.
    return {"ok": True}


@router.get("/me", response_model=MeResponse)
def who_am_i(
    authorization: str | None = Header(default=None, alias="Authorization"),
    db: Session = Depends(get_db),
) -> MeResponse:
    principal = _extract_principal(authorization)
    agents = [agent.slug for agent in get_accessible_agents(db, principal)]
    return MeResponse(user=PrincipalOut.from_principal(principal), agents=agents)
